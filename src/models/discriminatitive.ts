import {
    AutoConfig,
    AutoTokenizer,
    ModelRegistry,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
    Tensor,
    type DataType,
    type DeviceType,
    type ProgressInfo,
} from "@huggingface/transformers";
import type { FilteringModel, ModelState, ProgressCallback } from "./modelTypes";

export class D1OmniModel implements FilteringModel {
    // LiquidAI d1-omni-600M, converted to ONNX for the browser.
    // Original model + Python reference code: https://huggingface.co/LiquidAI/d1-omni-600M
    private static readonly REPO = "onnx-community/d1-omni-600M-ONNX";

    // Special tokens d1 uses to lay out a request (from d1's prompt.py).
    private static readonly TOKENS = {
        state: "<|reserved_7|>",
        question: "<|reserved_8|>",
        option: "<|reserved_9|>",
        optionEnd: "<|reserved_10|>",
        decide: "<|reserved_11|>",
        marker: "<|mask|>",
    };

    private static readonly NOUL_QTYPE = 2; // d1's question types: choice = 0, score = 1, noul (yes/no) = 2
    private static readonly PER_OPTION = 24; // token budget per option (prompt.py default)

    private _state: ModelState = "unknown";
    private config: PretrainedConfig | null = null;
    private tokenizer: PreTrainedTokenizer | null = null;
    private model: PreTrainedModel | null = null;
    private pending: Promise<void> | null = null; // the download/load currently running, if any

    private readonly question: string;
    private readonly criteria: { true?: string; false?: string };
    private readonly threshold: number;
    private readonly device: DeviceType;
    private readonly dtype: Extract<DataType, "q8" | "q4">;

    constructor(options: {
        /** The yes/no filter question, e.g. "Is this text about politics?" */
        question: string;
        /** Optional descriptions of what "yes" and "no" mean. d1 has generic defaults. */
        criteria?: { true?: string; false?: string };
        /** P(yes) at or above this means "filter it". */
        threshold?: number;
        /** "webgpu" in the browser; "cpu" when testing in Node. */
        device?: DeviceType;
        /** "q8" = model_quantized.onnx (0.41 GB), "q4" = model_q4.onnx (0.24 GB, less accurate). */
        dtype?: Extract<DataType, "q8" | "q4">;
    }) {
        this.question = options.question;
        this.criteria = options.criteria ?? {};
        this.threshold = options.threshold ?? 0.5;
        this.device = options.device ?? "webgpu";
        this.dtype = options.dtype ?? "q8";
    }

    get state(): ModelState {
        return this._state;
    }

    // ------------------------------------------------------------------ lifecycle

    async checkState(): Promise<ModelState> {
        // While something is running (or the model is in memory) we already know the state.
        if (this.pending || this._state === "ready") {
            return this._state;
        }
        try {
            const { files } = await ModelRegistry.is_cached_files(D1OmniModel.REPO, {
                config: await this.getConfig(),
                dtype: this.dtype,
            });
            // preprocessor_config.json is only used for images, so text-only use never downloads it.
            const cached = files.every((f) => f.cached || f.file === "preprocessor_config.json");
            this._state = cached ? "downloaded" : "not-downloaded";
        } catch (e) {
            this._state = "error";
            throw e;
        }
        return this._state;
    }

    /**
     * Puts the files in the cache without keeping the model in memory.
     * transformers.js has no "download only" call, so this loads the model once and frees it again.
     */
    async download(onProgress?: ProgressCallback): Promise<void> {
        if (this._state === "downloaded" || this._state === "ready") return;
        if (this.pending) return this.pending;

        this.pending = (async () => {
            const model = await this.loadModel(onProgress);
            await model.dispose();
            this._state = "downloaded";
        })();
        return this.run();
    }

    /** Downloads (if needed) and loads the model, after which request() works. */
    async load(onProgress?: ProgressCallback): Promise<void> {
        // If a download or load is already running, let it finish first.
        while (this.pending) await this.pending.catch(() => {});
        if (this._state === "ready") return;

        this.pending = (async () => {
            this.model = await this.loadModel(onProgress);
            this._state = "ready";
        })();
        return this.run();
    }

    // ------------------------------------------------------------------ inference

    async request(input: string): Promise<boolean> {
        return (await this.probability(input)) >= this.threshold;
    }

    /** P(yes) for this model's question over `input`, between 0 and 1. */
    async probability(input: string): Promise<number> {
        if (this._state !== "ready" || !this.model || !this.tokenizer || !this.config) {
            throw new Error(`D1OmniModel is not loaded (state: ${this._state}); call load() first`);
        }

        const { ids, markers } = this.encode(input);
        const int64 = (values: number[]) =>
            new Tensor("int64", BigInt64Array.from(values, BigInt), [1, values.length]);

        const { logits } = await this.model({
            input_ids: int64(ids),
            attention_mask: int64(ids.map(() => 1)),
            prefix_len: new Tensor("int64", BigInt64Array.from([0n]), [1]), // 0 = no image/audio
            marker_pos: int64(markers),
            marker_mask: int64(markers.map(() => 1)),
            qtype: new Tensor("int64", BigInt64Array.from([BigInt(D1OmniModel.NOUL_QTYPE)]), [1]),
        });

        // One score per option, in the order [false, true].
        const [noScore, yesScore] = Array.from(logits.data as Float32Array).slice(0, 2);

        // Calibration temperature from config.json (d1 picks it by question type and option count).
        const temperatures = (this.config as any).temperatures ?? {};
        const t = temperatures["noul:2"] ?? temperatures["noul"] ?? 1;

        // softmax over the two options, keeping P(true) = P(yes)
        return 1 / (1 + Math.exp((noScore - yesScore) / t));
    }

    // ------------------------------------------------------------------ internals

    /** Port of d1's prompt.py `encode()` for one yes/no question. */
    private encode(state: string): { ids: number[]; markers: number[] } {
        const tok = this.tokenizer!;
        const TOKENS = D1OmniModel.TOKENS;
        const maxLen: number = (this.config as any).max_length ?? 16384;
        const id = (token: string) => tok.convert_tokens_to_ids(token);
        // `<|name|>` -> `<¦name¦>`, so page text can never produce one of d1's special tokens.
        const escape = (text: string) => text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
        const enc = (text: string) =>
            Array.from(tok(escape(text), { add_special_tokens: false }).input_ids.data as BigInt64Array, Number);

        // A yes/no question is read as two options, "false" first (prompt.py render_options).
        const options = [
            "false: " + (this.criteria.false || "no, the statement does not hold"),
            "true: " + (this.criteria.true || "yes, the statement holds"),
        ];

        // Room for the question + options; the page text gets what is left.
        const budget = Math.max(96, Math.min(options.length * D1OmniModel.PER_OPTION + 32, Math.floor(maxLen / 2)));
        const perOption = Math.max(2, Math.floor((budget - 3 * options.length) / options.length));

        //   <q> question <opt> <mask> option_0 </opt> <opt> <mask> option_1 </opt> <decide>
        let question = [id(TOKENS.question), ...enc(this.question)].slice(0, Math.max(16, budget));
        let markers: number[] = [];
        for (const text of options) {
            markers.push(question.length + 1); // position of this option's <mask>
            question = [
                ...question,
                id(TOKENS.option),
                id(TOKENS.marker),
                ...enc(" " + text).slice(0, perOption),
                id(TOKENS.optionEnd),
            ];
        }
        question.push(id(TOKENS.decide));

        //   <bos> <state> page text ...question part
        const room = Math.max(0, maxLen - question.length - 2);
        const stateIds = [id(TOKENS.state), ...enc(state).slice(0, room)];
        const ids = [tok.bos_token_id, ...stateIds, ...question].slice(0, maxLen);
        markers = markers.map((m) => m + 1 + stateIds.length); // shift past <bos> and the state

        return { ids, markers };
    }

    private async loadModel(onProgress?: ProgressCallback): Promise<PreTrainedModel> {
        const config = await this.getConfig();
        this.tokenizer ??= await AutoTokenizer.from_pretrained(D1OmniModel.REPO);

        this._state = "loading";
        return await PreTrainedModel.from_pretrained(D1OmniModel.REPO, {
            config,
            dtype: this.dtype,
            device: this.device,
            use_external_data_format: true,
            progress_callback: (info: ProgressInfo) => {
                if (info.status !== "progress_total") return;
                // While bytes are still coming in we are downloading; after that, loading.
                this._state = info.progress < 100 ? "downloading" : "loading";
                onProgress?.(info.progress);
            },
        });
    }

    /** Runs the pending download/load and keeps the state right if it fails. */
    private async run(): Promise<void> {
        try {
            await this.pending;
        } catch (e) {
            this._state = "error";
            throw e;
        } finally {
            this.pending = null;
        }
    }

    /**
     * config.json, with one fix: it doesn't say the weights live in a separate
     * `.onnx_data` file, so transformers.js would miss that file when checking the cache
     * and when computing download progress.
     */
    private async getConfig(): Promise<PretrainedConfig> {
        if (!this.config) {
            const config = await AutoConfig.from_pretrained(D1OmniModel.REPO);
            const c = config as any;
            c["transformers.js_config"] = { ...c["transformers.js_config"], use_external_data_format: true };
            this.config = config;
        }
        return this.config;
    }
}
