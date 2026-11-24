export type ModelState =
    | "unknown"        // not checked yet (startup)
    | "not-downloaded"
    | "downloading"
    | "downloaded"     // in cache, not in memory
    | "loading"
    | "ready"          // in memory, request() works
    | "error";

export type ProgressCallback = (percent: number) => void;

export type FilteringModel = {
    readonly state: ModelState;
    checkState(): Promise<ModelState>;                  // asks the cache, fixes "unknown"
    download(onProgress?: ProgressCallback): Promise<void>; // no-op if already downloaded
    load(onProgress?: ProgressCallback): Promise<void>;     // downloads first if needed
    request(input: string): Promise<boolean>;           // only works when "ready"
}
