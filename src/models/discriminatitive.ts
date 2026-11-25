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

/**
 * LiquidAI d1-omni-600M: an encoder that answers a yes/no question about a text.
 * ONNX export: https://huggingface.co/onnx-community/d1-omni-600M-ONNX
 * Original model + Python reference code (prompt.py): https://huggingface.co/LiquidAI/d1-omni-600M
 */
export class D1OmniModel implements FilteringModel {
    private static readonly REPO = "onnx-community/d1-omni-600M-ONNX";

    /**
     * Special tokens that mark the parts of a request (from d1's prompt.py).
     *
     * The `<|reserved_N|>` tokens are spare slots in the LFM2 vocabulary: they exist in the
     * tokenizer but never come out of normal text. LiquidAI gave them a meaning while training
     * d1, so the model learned to read them as section markers. A yes/no request looks like:
     *
     *   <bos> <state> page text <q> question <opt> <mask> false: … </opt> <opt> <mask> true: … </opt> <decide>
     *
     * d1 is an encoder, so every token sees every other token. The vector at each option's
     * `<mask>` has therefore "read" the page text, the question and that option, and d1's
     * decision head turns that vector into one score per option.
     */
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

    /**
     * Token limits for the question and for each option. prompt.py derives them from the
     * number of options; for a yes/no question (2 options) its formula gives these values.
     */
    private static readonly QUESTION_MAX_TOKENS = 96;
    private static readonly OPTION_MAX_TOKENS = 45;

    private _state: ModelState = "unknown";
    /**
     * d1's own config.json from its Hugging Face repo (architecture, max length, calibration
     * temperatures), with our external-data fix applied. Not Placid's settings.
     * Needed before the model is loaded, to check the cache.
     */
    private modelRepoConfig: PretrainedConfig | null = null;
    /** Everything probability() needs; null until load() finishes. */
    private inference: {
        model: PreTrainedModel;
        tokenizer: PreTrainedTokenizer;
        maxLength: number; // longest request the model accepts, in tokens
        temperature: number; // calibration: divides the scores before they become a probability
    } | null = null;
    private pending: Promise<void> | null = null; // the download/load currently running, if any

    private readonly question: string;
    private readonly options: [no: string, yes: string];
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
        /** Where the model runs: "webgpu" (GPU, default) or "wasm" (CPU, in the browser). */
        device?: DeviceType;
        /**
         * Which published quantization to load:
         * "q4" = model_q4.onnx (0.24 GB), "q8" = model_quantized.onnx (0.41 GB).
         */
        dtype?: Extract<DataType, "q8" | "q4">;
    }) {
        this.question = options.question;
        // A yes/no question is read as two options, "false" first (prompt.py render_options).
        this.options = [
            "false: " + (options.criteria?.false || "no, the statement does not hold"),
            "true: " + (options.criteria?.true || "yes, the statement holds"),
        ];
        this.threshold = options.threshold ?? 0.5;
        this.device = options.device ?? "webgpu";
        this.dtype = options.dtype ?? "q4";
    }

    get state(): ModelState {
        return this._state;
    }

    // ------------------------------------------------------------------ lifecycle

    async checkState(): Promise<ModelState> {
        // While something is running (or the model is in memory) we already know the state.
        if (this.pending || this._state === "ready") return this._state;

        try {
            this._state = (await this.isCached()) ? "downloaded" : "not-downloaded";
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

        return this.startJob(async () => {
            const { model } = await this.loadFromHub(onProgress);
            await model.dispose();
            this._state = "downloaded";
        });
    }

    /** Downloads (if needed) and loads the model, after which request() works. */
    async load(onProgress?: ProgressCallback): Promise<void> {
        // If a download or load is already running, let it finish first.
        while (this.pending) await this.pending.catch(() => {});
        if (this._state === "ready") return;

        return this.startJob(async () => {
            this.inference = await this.loadFromHub(onProgress);
            this._state = "ready";
        });
    }

    private async isCached(): Promise<boolean> {
        const { files } = await ModelRegistry.is_cached_files(D1OmniModel.REPO, {
            config: await this.getModelRepoConfig(),
            dtype: this.dtype,
        });
        // preprocessor_config.json is only used for images, so text-only use never downloads it.
        return files.every((f) => f.cached || f.file === "preprocessor_config.json");
    }

    /** Runs a download/load as the one pending job, and marks the model "error" if it fails. */
    private async startJob(job: () => Promise<void>): Promise<void> {
        this.pending = job();
        try {
            await this.pending;
        } catch (e) {
            this._state = "error";
            throw e;
        } finally {
            this.pending = null;
        }
    }

    /** Downloads what is missing from the cache, then loads the model into memory. */
    private async loadFromHub(onProgress?: ProgressCallback): Promise<NonNullable<D1OmniModel["inference"]>> {
        const modelRepoConfig = await this.getModelRepoConfig();
        const tokenizer = await AutoTokenizer.from_pretrained(D1OmniModel.REPO);

        this._state = "loading";
        const model = await PreTrainedModel.from_pretrained(D1OmniModel.REPO, {
            config: modelRepoConfig,
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

        const d1Fields = modelRepoConfig as any; // d1-specific fields that PretrainedConfig doesn't type
        return {
            model,
            tokenizer,
            maxLength: d1Fields.max_length ?? 16384,
            // d1 keeps one temperature per question type and option count; ours is yes/no with 2 options.
            temperature: d1Fields.temperatures?.["noul:2"] ?? d1Fields.temperatures?.["noul"] ?? 1,
        };
    }

    /**
     * d1's config.json, with one fix: it doesn't say the weights live in a separate `.onnx_data`
     * file, so transformers.js would miss that file when checking the cache and when
     * computing download progress.
     */
    private async getModelRepoConfig(): Promise<PretrainedConfig> {
        if (!this.modelRepoConfig) {
            const fromRepo = (await AutoConfig.from_pretrained(D1OmniModel.REPO)) as any;
            fromRepo["transformers.js_config"] = { ...fromRepo["transformers.js_config"], use_external_data_format: true };
            this.modelRepoConfig = fromRepo as PretrainedConfig;
        }
        return this.modelRepoConfig;
    }

    // ------------------------------------------------------------------ inference

    async request(text: string): Promise<boolean> {
        return (await this.probability(text)) >= this.threshold;
    }

    /** P(yes) for this model's question over `text`, between 0 and 1. */
    async probability(text: string): Promise<number> {
        if (!this.inference) {
            throw new Error(`D1OmniModel is not loaded (state: ${this._state}); call load() first`);
        }
        const { model, tokenizer, maxLength, temperature } = this.inference;
        const { ids, markers } = this.encode(text, tokenizer, maxLength);
        const int64 = D1OmniModel.int64Tensor;

        const { logits } = await model({
            input_ids: int64(ids),
            attention_mask: int64(ids.map(() => 1)), // 1 = look at this token (all of them)
            prefix_len: int64([0], [1]), // number of image/audio tokens before the text: none
            marker_pos: int64(markers),
            marker_mask: int64(markers.map(() => 1)),
            qtype: int64([D1OmniModel.YES_NO_QUESTION_TYPE], [1]),
        });

        // One score per option, in option order: [false, true].
        const [noScore, yesScore] = logits.data as Float32Array;

        // Softmax over two options = sigmoid of the score difference.
        return 1 / (1 + Math.exp((noScore - yesScore) / temperature));
    }

    /** Port of d1's prompt.py `encode()` for one yes/no question. */
    private encode(text: string, tokenizer: PreTrainedTokenizer, maxLength: number) {
        const TOKENS = D1OmniModel.TOKENS;
        const tokenId = (token: string) => tokenizer.convert_tokens_to_ids(token);
        const tokenize = (s: string) => D1OmniModel.tokenize(tokenizer, s);

        //   <q> question <opt> <mask> false: … </opt> <opt> <mask> true: … </opt> <decide>
        const question = [tokenId(TOKENS.question), ...tokenize(this.question)].slice(0, D1OmniModel.QUESTION_MAX_TOKENS);
        const markerOffsets: number[] = []; // where each <mask> sits inside `question`
        for (const option of this.options) {
            question.push(tokenId(TOKENS.option));
            markerOffsets.push(question.length);
            question.push(tokenId(TOKENS.marker), ...tokenize(" " + option).slice(0, D1OmniModel.OPTION_MAX_TOKENS), tokenId(TOKENS.optionEnd));
        }
        question.push(tokenId(TOKENS.decide));

        //   <bos> <state> page text — cut so the whole request fits in maxLength
        const room = maxLength - question.length - 2;
        const page = [tokenizer.bos_token_id, tokenId(TOKENS.state), ...tokenize(text).slice(0, room)];

        return { ids: [...page, ...question], markers: markerOffsets.map((offset) => page.length + offset) };
    }

    /** Text -> token ids, with `<|name|>` escaped to `<¦name¦>` so page text can't fake a special token. */
    private static tokenize(tokenizer: PreTrainedTokenizer, text: string): number[] {
        const escaped = text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
        return Array.from(tokenizer(escaped, { add_special_tokens: false }).input_ids.data as BigInt64Array, Number);
    }

    private static int64Tensor(values: number[], dims = [1, values.length]): Tensor {
        return new Tensor("int64", BigInt64Array.from(values, BigInt), dims);
    }
}

/**
 * Jared Palmer's Kev: Qwen3 + LoRA + a pointer head that scores each answer option.
 * Model card + reference code: https://huggingface.co/onnx-community/kev-4b-ONNX
 */
export class KevModel implements FilteringModel {
    private static readonly REPOS = {
        "0.6b": "onnx-community/kev-0.6b-ONNX", // ~0.34 GB (q4f16)
        "4b": "onnx-community/kev-4b-ONNX", // ~2.3 GB (q4f16)
    };

    /**
     * Special tokens that mark the parts of a request (from the model card).
     *
     * Kev reuses existing Qwen tokens meant for code fill-in-the-middle and image boxes. They
     * almost never appear in normal text, so fine-tuning could give them a new meaning:
     *
     *   <state> page text <q> question <opt> no </opt> <opt> yes </opt> <decide>
     *
     * Kev is a decoder, so each token only sees the tokens before it. An option's `</opt>` is
     * the first position that has read the text, the question and that whole option, so the
     * pointer head's score there is that option's score.
     */
    private static readonly TOKENS = {
        state: "<|fim_prefix|>", // starts the text being judged
        question: "<|fim_middle|>", // starts the question
        option: "<|box_start|>", // opens one answer option
        optionEnd: "<|box_end|>", // closes it; its score is the option's score
        decide: "<|fim_suffix|>", // ends the request
    };

    private static readonly YES_NO_OPTIONS = ["no", "yes"];
    private static readonly MAX_TEXT_TOKENS = 8192; // page text is cut to this (model card default)

    private _state: ModelState = "unknown";
    /** Everything probability() needs; null until load() finishes. */
    private inference: { model: PreTrainedModel; tokenizer: PreTrainedTokenizer } | null = null;
    private pending: Promise<void> | null = null; // the download/load currently running, if any

    private readonly repo: string;
    private readonly question: string;
    private readonly threshold: number;
    private readonly device: DeviceType;
    private readonly dtype: Extract<DataType, "q4f16" | "q4">;

    constructor(options: {
        /** The yes/no filter question, e.g. "Is this text about politics?" */
        question: string;
        /** P(yes) at or above this means "filter it". */
        threshold?: number;
        /** "4b" = more accurate, "0.6b" = smaller and faster. */
        size?: keyof typeof KevModel.REPOS;
        /** Where the model runs: "webgpu" (GPU, default) or "wasm" (CPU, in the browser). */
        device?: DeviceType;
        /**
         * Which published quantization to load. Both have 4-bit weights and the same accuracy.
         * "q4f16" = 16-bit maths: smaller and ~2x faster on GPU, needs WebGPU's shader-f16 feature.
         * "q4" = 32-bit maths: runs on any WebGPU device.
         */
        dtype?: Extract<DataType, "q4f16" | "q4">;
    }) {
        this.question = options.question;
        this.threshold = options.threshold ?? 0.5;
        this.repo = KevModel.REPOS[options.size ?? "4b"];
        this.device = options.device ?? "webgpu";
        this.dtype = options.dtype ?? "q4f16";
    }

    get state(): ModelState {
        return this._state;
    }

    // ------------------------------------------------------------------ lifecycle

    async checkState(): Promise<ModelState> {
        // While something is running (or the model is in memory) we already know the state.
        if (this.pending || this._state === "ready") return this._state;

        try {
            this._state = (await this.isCached()) ? "downloaded" : "not-downloaded";
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

        return this.startJob(async () => {
            const { model } = await this.loadFromHub(onProgress);
            await model.dispose();
            this._state = "downloaded";
        });
    }

    /** Downloads (if needed) and loads the model, after which request() works. */
    async load(onProgress?: ProgressCallback): Promise<void> {
        // If a download or load is already running, let it finish first.
        while (this.pending) await this.pending.catch(() => {});
        if (this._state === "ready") return;

        return this.startJob(async () => {
            this.inference = await this.loadFromHub(onProgress);
            this._state = "ready";
        });
    }

    private async isCached(): Promise<boolean> {
        const { files } = await ModelRegistry.is_cached_files(this.repo, { dtype: this.dtype });
        return files.every((f) => f.cached);
    }

    /** Runs a download/load as the one pending job, and marks the model "error" if it fails. */
    private async startJob(job: () => Promise<void>): Promise<void> {
        this.pending = job();
        try {
            await this.pending;
        } catch (e) {
            this._state = "error";
            throw e;
        } finally {
            this.pending = null;
        }
    }

    /** Downloads what is missing from the cache, then loads the model into memory. */
    private async loadFromHub(onProgress?: ProgressCallback): Promise<NonNullable<KevModel["inference"]>> {
        const tokenizer = await AutoTokenizer.from_pretrained(this.repo);

        this._state = "loading";
        const model = await PreTrainedModel.from_pretrained(this.repo, {
            dtype: this.dtype,
            device: this.device,
            progress_callback: (info: ProgressInfo) => {
                if (info.status !== "progress_total") return;
                // While bytes are still coming in we are downloading; after that, loading.
                this._state = info.progress < 100 ? "downloading" : "loading";
                onProgress?.(info.progress);
            },
        });
        return { model, tokenizer };
    }

    // ------------------------------------------------------------------ inference

    async request(text: string): Promise<boolean> {
        return (await this.probability(text)) >= this.threshold;
    }

    /** P(yes) for this model's question over `text`, between 0 and 1. */
    async probability(text: string): Promise<number> {
        if (!this.inference) {
            throw new Error(`KevModel is not loaded (state: ${this._state}); call load() first`);
        }
        const { model, tokenizer } = this.inference;
        const { ids, optionEnds } = this.encode(text, tokenizer);
        const int64 = (values: number[]) => new Tensor("int64", BigInt64Array.from(values, BigInt), [1, values.length]);

        // One score per token; only the scores at each option's </opt> mean anything.
        const { logits } = await model({ input_ids: int64(ids), attention_mask: int64(ids.map(() => 1)) });
        const scores = logits.to("float32").data as Float32Array;
        const [noScore, yesScore] = optionEnds.map((i) => scores[i]);

        // Softmax over two options = sigmoid of the score difference.
        return 1 / (1 + Math.exp(noScore - yesScore));
    }

    /** Port of the model card's prompt layout for one yes/no question. */
    private encode(text: string, tokenizer: PreTrainedTokenizer) {
        const TOKENS = KevModel.TOKENS;
        const tokenId = (token: string) => tokenizer.convert_tokens_to_ids(token);
        const tokenize = (s: string) => KevModel.tokenize(tokenizer, s);

        //   <state> page text <q> question <opt> no </opt> <opt> yes </opt> <decide>
        const ids = [
            tokenId(TOKENS.state),
            ...tokenize(text).slice(0, KevModel.MAX_TEXT_TOKENS),
            tokenId(TOKENS.question),
            ...tokenize(this.question),
        ];
        const optionEnds: number[] = []; // where each </opt> sits in `ids`
        for (const option of KevModel.YES_NO_OPTIONS) {
            ids.push(tokenId(TOKENS.option), ...tokenize(option), tokenId(TOKENS.optionEnd));
            optionEnds.push(ids.length - 1);
        }
        ids.push(tokenId(TOKENS.decide));

        return { ids, optionEnds };
    }

    /** Text -> token ids, with `<|name|>` escaped to `<¦name¦>` so page text can't fake a special token. */
    private static tokenize(tokenizer: PreTrainedTokenizer, text: string): number[] {
        const escaped = text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
        return Array.from(tokenizer(escaped, { add_special_tokens: false }).input_ids.data as BigInt64Array, Number);
    }
}
