import {
    AutoConfig,
    AutoTokenizer,
    ModelRegistry,
    PreTrainedModel,
    PretrainedConfig,
    type ProgressInfo,
} from "@huggingface/transformers";
import { hasGpuAdapter } from "../../utils/webgpu.ts";
import type { Model, ModelState, ProgressCallback } from "../modelTypes";
import { D1OmniSession, type D1Forward } from "./d1OmniSession.ts";

/** LiquidAI d1-omni-600M's files and lifecycle: is it downloaded, and loading it. Reference: https://huggingface.co/LiquidAI/d1-omni-600M */
export class D1OmniModel implements Model<D1OmniSession> {
    private static readonly REPO = "onnx-community/d1-omni-600M-ONNX";

    private _state: ModelState = "unknown";
    /** d1's config.json from its Hugging Face repo (not Placid's settings). */
    private modelRepoConfig: PretrainedConfig | null = null;
    private sharedLoad: Promise<D1OmniSession> | null = null; // started by the first load(); later calls wait for it
    private progressListeners = new Set<ProgressCallback>(); // onProgress of every load() call still waiting
    private onGpu: boolean | null = null; // decided once by hasGpuAdapter(): which files we use

    // Progress of the current load, reset when it starts.
    private needsDownload = false; // are any files missing from the cache?
    private bytesPerFile = new Map<string, { loaded: number; total: number }>();
    private reportedPercent = 0; // highest percent reported so far

    private readonly dtype: "q4" | "q8"; // "q4" (0.24 GB) or "q8" (0.41 GB)

    constructor(dtype: "q4" | "q8" = "q4") {
        this.dtype = dtype;
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

    /** True when every file the model needs is already in the browser's download cache. */
    private async modelFilesInBrowserCache(): Promise<boolean> {
        const config = await this.getModelRepoConfig();
        const files = (await this.runsOnGpu()) ? [this.fullModelFile()] : this.splitModelFiles();
        for (const file of files) {
            // model_file_name and include_processor work but are missing from transformers.js's types.
            // include_processor: false skips d1's image/audio settings, which we never download.
            const options = { config, model_file_name: file.name, dtype: file.dtype, include_processor: false };
            if (!(await ModelRegistry.is_cached(D1OmniModel.REPO, options))) return false;
        }
        return true;
    }

    /** A real GPU runs d1's full model on WebGPU. Without one we use d1's two split files on the CPU: the full model's q4 embedding can't run there. */
    private async runsOnGpu(): Promise<boolean> {
        this.onGpu ??= await hasGpuAdapter();
        return this.onGpu;
    }

    /** GPU: d1's full model in one file. */
    private fullModelFile() {
        return { name: "model", dtype: this.dtype };
    }

    /** CPU: d1's embedding (token ids -> vectors), then the rest of the model. The embedding is fp16: its q4 version uses an operator the CPU backend lacks. */
    private splitModelFiles() {
        return [
            { name: "embed_tokens", dtype: "fp16" as const },
            { name: "decision", dtype: this.dtype },
        ];
    }

    /** Downloads any missing model files, then loads the model into memory. */
    private async downloadAndLoad(): Promise<D1OmniSession> {
        this.needsDownload = !(await this.modelFilesInBrowserCache());
        this.bytesPerFile.clear();
        this.reportedPercent = 0;
        this._state = this.needsDownload ? "downloading" : "loading";

        const modelRepoConfig = await this.getModelRepoConfig();
        const tokenizer = await AutoTokenizer.from_pretrained(D1OmniModel.REPO);
        const forward = await this.loadForward(modelRepoConfig);

        const d1Fields = modelRepoConfig as any; // d1-specific fields that PretrainedConfig doesn't type
        return new D1OmniSession(
            forward,
            tokenizer,
            d1Fields.max_length ?? 16384,
            // d1 keeps one temperature per question type and option count; ours is yes/no with 2 options.
            d1Fields.temperatures?.["noul:2"] ?? d1Fields.temperatures?.["noul"] ?? 1,
        );
    }

    /** GPU if there is one; if loading there fails, the CPU instead. */
    private async loadForward(config: PretrainedConfig): Promise<D1Forward> {
        if (await this.runsOnGpu()) {
            try {
                return await this.loadFullModel(config);
            } catch (error) {
                console.warn("d1 failed to load on WebGPU, using the CPU:", error);
                this.onGpu = false; // from now on we use the split files, checkState() included
            }
        }
        return this.loadSplitModel(config);
    }

    private async loadFullModel(config: PretrainedConfig): Promise<D1Forward> {
        const model = await this.loadFile(config, this.fullModelFile(), "webgpu");
        return model;
    }

    private async loadSplitModel(config: PretrainedConfig): Promise<D1Forward> {
        const [embedTokensFile, decisionFile] = this.splitModelFiles();
        // Both at once, so progress counts both files' bytes from the start.
        const [embedTokens, decision] = await Promise.all([
            this.loadFile(config, embedTokensFile, "wasm"),
            this.loadFile(config, decisionFile, "wasm"),
        ]);

        return async (inputs) => {
            const { input_ids, ...decisionInputs } = inputs;
            const { inputs_embeds } = await embedTokens({ input_ids });
            return decision({ ...decisionInputs, inputs_embeds });
        };
    }

    private loadFile(config: PretrainedConfig, file: { name: string; dtype: "fp16" | "q4" | "q8" }, device: "webgpu" | "wasm") {
        return PreTrainedModel.from_pretrained(D1OmniModel.REPO, {
            config,
            model_file_name: file.name,
            dtype: file.dtype,
            device,
            use_external_data_format: true,
            progress_callback: (info: ProgressInfo) => this.reportProgress(file.name, info),
        });
    }

    /** Adds up the bytes of every file being loaded and reports one 0–100 to every waiting load() caller. */
    private reportProgress(file: string, info: ProgressInfo): void {
        if (info.status !== "progress_total") return;
        this.bytesPerFile.set(file, { loaded: info.loaded, total: info.total });

        let loaded = 0;
        let total = 0;
        for (const bytes of this.bytesPerFile.values()) {
            loaded += bytes.loaded;
            total += bytes.total;
        }
        // Totals grow as each file's parts start downloading, so a fresh percent can dip; we never report less.
        const percent = total > 0 ? (loaded / total) * 100 : 0;
        this.reportedPercent = Math.max(this.reportedPercent, percent);

        // transformers.js reports progress for cache reads too, so only a real download is "downloading".
        this._state = this.needsDownload && this.reportedPercent < 100 ? "downloading" : "loading";
        for (const listener of this.progressListeners) {
            listener(this.reportedPercent);
        }
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
