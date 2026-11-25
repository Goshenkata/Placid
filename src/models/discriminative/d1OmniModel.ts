import {
    AutoConfig,
    AutoTokenizer,
    ModelRegistry,
    PreTrainedModel,
    PretrainedConfig,
    type DataType,
    type DeviceType,
    type ProgressInfo,
} from "@huggingface/transformers";
import type { Model, ModelState, ProgressCallback } from "../modelTypes";
import { D1OmniSession } from "./d1OmniSession.ts";

/** LiquidAI d1-omni-600M's files and lifecycle: is it downloaded, and loading it. Reference: https://huggingface.co/LiquidAI/d1-omni-600M */
export class D1OmniModel implements Model<D1OmniSession> {
    private static readonly REPO = "onnx-community/d1-omni-600M-ONNX";

    private _state: ModelState = "unknown";
    /** d1's config.json from its Hugging Face repo (not Placid's settings). */
    private modelRepoConfig: PretrainedConfig | null = null;
    private sharedLoad: Promise<D1OmniSession> | null = null; // started by the first load(); later calls wait for it
    private progressListeners = new Set<ProgressCallback>(); // onProgress of every load() call still waiting

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
    async load(onProgress?: ProgressCallback): Promise<D1OmniSession> {
        if (onProgress) this.progressListeners.add(onProgress);
        this.sharedLoad ??= this.downloadAndLoad();
        try {
            const session = await this.sharedLoad;
            this._state = "ready";
            return session;
        } catch (error) {
            this.sharedLoad = null; // so the next load() starts over
            this._state = "error";
            throw error;
        } finally {
            if (onProgress) this.progressListeners.delete(onProgress);
        }
    }

    /** Frees the model's GPU memory. Sessions from earlier load() calls stop working; the files stay cached. */
    async unload(): Promise<void> {
        const loading = this.sharedLoad;
        if (!loading) return;
        this.sharedLoad = null; // the next load() loads again
        const session = await loading;
        this._state = "downloaded";
        await session.dispose();
    }

    /** True when every file the model needs is already in the browser's download cache. */
    private async modelFilesInBrowserCache(): Promise<boolean> {
        const textOnly = { include_processor: false }; // skip d1's image/audio settings, which we never download (missing from transformers.js's types)
        return ModelRegistry.is_cached(D1OmniModel.REPO, { config: await this.getModelRepoConfig(), dtype: this.dtype, ...textOnly });
    }

    /** Downloads any missing model files, then loads the model into memory. */
    private async downloadAndLoad(): Promise<D1OmniSession> {
        const needsDownload = !(await this.modelFilesInBrowserCache());
        this._state = needsDownload ? "downloading" : "loading";
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
                for (const listener of this.progressListeners) {
                    listener(info.progress);
                }
            },
        });

        const d1Fields = modelRepoConfig as any; // d1-specific fields that PretrainedConfig doesn't type
        return new D1OmniSession(
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
