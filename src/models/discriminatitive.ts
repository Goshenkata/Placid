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
import type { DecisionRuntime, LoadableModel, ModelState, ProgressCallback } from "./modelTypes";

/** LiquidAI d1-omni-600M's files and lifecycle: is it downloaded, and loading it. Reference: https://huggingface.co/LiquidAI/d1-omni-600M */
export class D1OmniModel implements LoadableModel<D1OmniRuntime> {
    private static readonly REPO = "onnx-community/d1-omni-600M-ONNX";

    private _state: ModelState = "unknown";
    /** d1's config.json from its Hugging Face repo (not Placid's settings). */
    private modelRepoConfig: PretrainedConfig | null = null;
    private sharedLoad: Promise<D1OmniRuntime> | null = null; // started by the first load(); later calls wait for it

    private readonly device: DeviceType;
    private readonly dtype: Extract<DataType, "q8" | "q4">;

    constructor(options: {
        /** Where the model runs: "webgpu" (GPU, default) or "wasm" (CPU, in the browser). */
        device?: DeviceType;
        /** "q4" (0.24 GB, default) or "q8" (0.41 GB). */
        dtype?: Extract<DataType, "q8" | "q4">;
    } = {}) {
        this.device = options.device ?? "webgpu";
        this.dtype = options.dtype ?? "q4";
    }

    get state(): ModelState {
        return this._state;
    }

    /** Startup only: looks in the browser cache to replace "unknown" with "downloaded" or "not-downloaded". */
    async checkState(): Promise<ModelState> {
        if (this._state === "unknown") {
            this._state = (await this.modelFilesInBrowserCache()) ? "downloaded" : "not-downloaded";
        }
        return this._state;
    }

    /** Downloads the model files if needed and loads the model into memory. */
    async load(onProgress?: ProgressCallback): Promise<D1OmniRuntime> {
        this.sharedLoad ??= this.downloadAndLoad(onProgress);
        try {
            const runtime = await this.sharedLoad;
            this._state = "ready";
            return runtime;
        } catch (error) {
            this.sharedLoad = null; // so the next load() starts over
            this._state = "error";
            throw error;
        }
    }

    /** True when every file the model needs is already in the browser's download cache. */
    private async modelFilesInBrowserCache(): Promise<boolean> {
        const textOnly = { include_processor: false }; // skip d1's image/audio settings, which we never download (missing from transformers.js's types)
        return ModelRegistry.is_cached(D1OmniModel.REPO, { config: await this.getModelRepoConfig(), dtype: this.dtype, ...textOnly });
    }

    /** Downloads any missing model files, then loads the model into memory. */
    private async downloadAndLoad(onProgress?: ProgressCallback): Promise<D1OmniRuntime> {
        this._state = "loading";
        const needsDownload = !(await this.modelFilesInBrowserCache());
        const modelRepoConfig = await this.getModelRepoConfig();
        const tokenizer = await AutoTokenizer.from_pretrained(D1OmniModel.REPO);
        const model = await PreTrainedModel.from_pretrained(D1OmniModel.REPO, {
            config: modelRepoConfig,
            dtype: this.dtype,
            device: this.device,
            use_external_data_format: true,
            progress_callback: (info: ProgressInfo) => {
                if (info.status !== "progress_total") return;
                // transformers.js reports progress for cache reads too, so only a real download is "downloading".
                this._state = needsDownload && info.progress < 100 ? "downloading" : "loading";
                onProgress?.(info.progress);
            },
        });

        const d1Fields = modelRepoConfig as any; // d1-specific fields that PretrainedConfig doesn't type
        return new D1OmniRuntime(
            model,
            tokenizer,
            d1Fields.max_length ?? 16384,
            // d1 keeps one temperature per question type and option count; ours is yes/no with 2 options.
            d1Fields.temperatures?.["noul:2"] ?? d1Fields.temperatures?.["noul"] ?? 1,
        );
    }

    /** d1's config.json is missing the flag saying its weights are in a separate .onnx_data file; we add it so the cache check and progress include that file. */
    private async getModelRepoConfig(): Promise<PretrainedConfig> {
        if (!this.modelRepoConfig) {
            const fromRepo = (await AutoConfig.from_pretrained(D1OmniModel.REPO)) as any;
            fromRepo["transformers.js_config"] = { ...fromRepo["transformers.js_config"], use_external_data_format: true };
            this.modelRepoConfig = fromRepo as PretrainedConfig;
        }
        return this.modelRepoConfig;
    }
}

/** A loaded d1-omni-600M that answers yes/no questions about texts. Get one from D1OmniModel.load(). */
export class D1OmniRuntime implements DecisionRuntime {
    /** Section markers d1 was trained on: <bos> <state> text <q> question <opt> <mask> option </opt> … <decide> */
    private static readonly TOKENS = {
        state: "<|reserved_7|>", // starts the text being judged ("state" in d1's terms)
        question: "<|reserved_8|>", // starts the question
        option: "<|reserved_9|>", // opens one answer option
        optionEnd: "<|reserved_10|>", // closes it
        decide: "<|reserved_11|>", // ends the request: "now decide"
        marker: "<|mask|>", // one per option: where the decision head reads that option's score
    };

    /** d1's question types: choice = 0, score = 1, yes/no ("noul") = 2. */
    private static readonly YES_NO_QUESTION_TYPE = 2;

    /** prompt.py's token limits, worked out for a yes/no question (2 options). */
    private static readonly QUESTION_MAX_TOKENS = 96;
    private static readonly OPTION_MAX_TOKENS = 45;

    private readonly model: PreTrainedModel;
    private readonly tokenizer: PreTrainedTokenizer;
    private readonly maxLength: number; // longest request the model accepts, in tokens
    private readonly temperature: number; // calibration: divides the scores before they become a probability

    constructor(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, maxLength: number, temperature: number) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.temperature = temperature;
    }

    /**
     * P(yes) for `question` about `text`, between 0 and 1.
     * `criteria` optionally describes what "yes" and "no" mean; d1 has generic defaults.
     */
    async probability(text: string, question: string, criteria?: { true?: string; false?: string }): Promise<number> {
        // A yes/no question is read as two options, "false" first (prompt.py render_options).
        const options = [
            "false: " + (criteria?.false || "no, the statement does not hold"),
            "true: " + (criteria?.true || "yes, the statement holds"),
        ];
        const { ids, markers } = this.encode(text, question, options);
        const int64 = D1OmniRuntime.int64Tensor;

        const { logits } = await this.model({
            input_ids: int64(ids),
            attention_mask: int64(ids.map(() => 1)), // 1 = look at this token (all of them)
            prefix_len: int64([0], [1]), // number of image/audio tokens before the text: none
            marker_pos: int64(markers),
            marker_mask: int64(markers.map(() => 1)),
            qtype: int64([D1OmniRuntime.YES_NO_QUESTION_TYPE], [1]),
        });

        // One score per option, in option order: [false, true].
        const [noScore, yesScore] = logits.data as Float32Array;

        // Softmax over two options = sigmoid of the score difference.
        return 1 / (1 + Math.exp((noScore - yesScore) / this.temperature));
    }

    /** Port of d1's prompt.py `encode()` for one yes/no question. */
    private encode(text: string, questionText: string, options: string[]) {
        const TOKENS = D1OmniRuntime.TOKENS;
        const tokenId = (token: string) => this.tokenizer.convert_tokens_to_ids(token);
        const tokenize = (s: string) => this.tokenize(s);

        //   <q> question <opt> <mask> false: … </opt> <opt> <mask> true: … </opt> <decide>
        const question = [tokenId(TOKENS.question), ...tokenize(questionText)].slice(0, D1OmniRuntime.QUESTION_MAX_TOKENS);
        const markerOffsets: number[] = []; // where each <mask> sits inside `question`
        for (const option of options) {
            question.push(tokenId(TOKENS.option));
            markerOffsets.push(question.length);
            question.push(tokenId(TOKENS.marker), ...tokenize(" " + option).slice(0, D1OmniRuntime.OPTION_MAX_TOKENS), tokenId(TOKENS.optionEnd));
        }
        question.push(tokenId(TOKENS.decide));

        //   <bos> <state> page text — cut so the whole request fits in maxLength
        const room = this.maxLength - question.length - 2;
        const page = [this.tokenizer.bos_token_id, tokenId(TOKENS.state), ...tokenize(text).slice(0, room)];

        return { ids: [...page, ...question], markers: markerOffsets.map((offset) => page.length + offset) };
    }

    /** Text -> token ids, with `<|name|>` escaped to `<¦name¦>` so page text can't fake a special token. */
    private tokenize(text: string): number[] {
        const escaped = text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
        return Array.from(this.tokenizer(escaped, { add_special_tokens: false }).input_ids.data as BigInt64Array, Number);
    }

    private static int64Tensor(values: number[], dims = [1, values.length]): Tensor {
        return new Tensor("int64", BigInt64Array.from(values, BigInt), dims);
    }
}
