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

/** LiquidAI d1-omni-600M, a yes/no decision encoder. Reference code: https://huggingface.co/LiquidAI/d1-omni-600M */
export class D1OmniModel implements FilteringModel {
    private static readonly REPO = "onnx-community/d1-omni-600M-ONNX";

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

    private _state: ModelState = "unknown";
    /** d1's config.json from its Hugging Face repo (not Placid's settings). */
    private modelRepoConfig: PretrainedConfig | null = null;
    /** Everything probability() needs; null until load() finishes. */
    private inference: {
        model: PreTrainedModel;
        tokenizer: PreTrainedTokenizer;
        maxLength: number; // longest request the model accepts, in tokens
        temperature: number; // calibration: divides the scores before they become a probability
    } | null = null;
    private downloadOrLoadInProgress: Promise<void> | null = null;

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
        /** "q4" (0.24 GB, default) or "q8" (0.41 GB). */
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

    /** Startup only: looks in the browser cache to replace "unknown" with "downloaded" or "not-downloaded". */
    async checkState(): Promise<ModelState> {
        if (this._state === "unknown") {
            this._state = (await this.modelFilesInBrowserCache()) ? "downloaded" : "not-downloaded";
        }
        return this._state;
    }

    /** Downloads the model files without keeping the model in memory. */
    async download(onProgress?: ProgressCallback): Promise<void> {
        // Let a download or load that is already running finish first.
        while (this.downloadOrLoadInProgress) await this.downloadOrLoadInProgress.catch(() => {});
        if (this._state === "downloaded" || this._state === "ready") return;

        await this.runDownloadOrLoad(async () => {
            const { model } = await this.downloadAndLoad(onProgress); // transformers.js can't download without loading
            await model.dispose();
            this._state = "downloaded";
        });
    }

    /** Downloads the model files if needed and loads the model into memory, after which request() works. */
    async load(onProgress?: ProgressCallback): Promise<void> {
        // Let a download or load that is already running finish first.
        while (this.downloadOrLoadInProgress) await this.downloadOrLoadInProgress.catch(() => {});
        if (this._state === "ready") return;

        await this.runDownloadOrLoad(async () => {
            this.inference = await this.downloadAndLoad(onProgress);
            this._state = "ready";
        });
    }

    /** True when every file the model needs is already in the browser's download cache. */
    private async modelFilesInBrowserCache(): Promise<boolean> {
        const { files } = await ModelRegistry.is_cached_files(D1OmniModel.REPO, {
            config: await this.getModelRepoConfig(),
            dtype: this.dtype,
        });
        // preprocessor_config.json is only used for images, so text-only use never downloads it.
        return files.every((f) => f.cached || f.file === "preprocessor_config.json");
    }

    /** Runs a download or load, remembering it as in progress; if it fails the state becomes "error". */
    private async runDownloadOrLoad(work: () => Promise<void>): Promise<void> {
        this.downloadOrLoadInProgress = work();
        try {
            await this.downloadOrLoadInProgress;
        } catch (e) {
            this._state = "error";
            throw e;
        } finally {
            this.downloadOrLoadInProgress = null;
        }
    }

    /** Downloads any missing model files, then loads the model into memory. */
    private async downloadAndLoad(onProgress?: ProgressCallback): Promise<NonNullable<D1OmniModel["inference"]>> {
        this._state = "loading";
        const modelRepoConfig = await this.getModelRepoConfig();
        const tokenizer = await AutoTokenizer.from_pretrained(D1OmniModel.REPO);
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

    /** d1's config.json forgets its weights live in `.onnx_data`; without this fix the cache check misses them. */
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

/** Jared Palmer's Kev: Qwen3 + a pointer head that scores each option. https://huggingface.co/onnx-community/kev-4b-ONNX */
export class KevModel implements FilteringModel {
    private static readonly REPOS = {
        "0.6b": "onnx-community/kev-0.6b-ONNX", // ~0.34 GB (q4f16)
        "4b": "onnx-community/kev-4b-ONNX", // ~2.3 GB (q4f16)
    };

    /** Section markers (reused Qwen tokens): <state> text <q> question <opt> no </opt> <opt> yes </opt> <decide> */
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
    private downloadOrLoadInProgress: Promise<void> | null = null;

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
        /** "q4f16" (default, faster, needs WebGPU shader-f16) or "q4" (runs on any WebGPU). */
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

    /** Startup only: looks in the browser cache to replace "unknown" with "downloaded" or "not-downloaded". */
    async checkState(): Promise<ModelState> {
        if (this._state === "unknown") {
            this._state = (await this.modelFilesInBrowserCache()) ? "downloaded" : "not-downloaded";
        }
        return this._state;
    }

    /** Downloads the model files without keeping the model in memory. */
    async download(onProgress?: ProgressCallback): Promise<void> {
        // Let a download or load that is already running finish first.
        while (this.downloadOrLoadInProgress) await this.downloadOrLoadInProgress.catch(() => {});
        if (this._state === "downloaded" || this._state === "ready") return;

        await this.runDownloadOrLoad(async () => {
            const { model } = await this.downloadAndLoad(onProgress); // transformers.js can't download without loading
            await model.dispose();
            this._state = "downloaded";
        });
    }

    /** Downloads the model files if needed and loads the model into memory, after which request() works. */
    async load(onProgress?: ProgressCallback): Promise<void> {
        // Let a download or load that is already running finish first.
        while (this.downloadOrLoadInProgress) await this.downloadOrLoadInProgress.catch(() => {});
        if (this._state === "ready") return;

        await this.runDownloadOrLoad(async () => {
            this.inference = await this.downloadAndLoad(onProgress);
            this._state = "ready";
        });
    }

    /** True when every file the model needs is already in the browser's download cache. */
    private async modelFilesInBrowserCache(): Promise<boolean> {
        const { files } = await ModelRegistry.is_cached_files(this.repo, { dtype: this.dtype });
        return files.every((f) => f.cached);
    }

    /** Runs a download or load, remembering it as in progress; if it fails the state becomes "error". */
    private async runDownloadOrLoad(work: () => Promise<void>): Promise<void> {
        this.downloadOrLoadInProgress = work();
        try {
            await this.downloadOrLoadInProgress;
        } catch (e) {
            this._state = "error";
            throw e;
        } finally {
            this.downloadOrLoadInProgress = null;
        }
    }

    /** Downloads any missing model files, then loads the model into memory. */
    private async downloadAndLoad(onProgress?: ProgressCallback): Promise<NonNullable<KevModel["inference"]>> {
        this._state = "loading";
        const tokenizer = await AutoTokenizer.from_pretrained(this.repo);
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
