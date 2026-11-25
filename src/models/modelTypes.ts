export type ModelState =
    | "unknown"        // not checked yet (startup)
    | "not-downloaded"
    | "downloading"
    | "downloaded"     // in cache, not in memory
    | "loading"
    | "ready"          // in memory, load() has returned the runtime
    | "error";

export type ProgressCallback = (percent: number) => void;

/** A model's files and lifecycle: is it downloaded, and loading it. No inference. */
export type LoadableModel<Runtime> = {
    readonly state: ModelState;
    checkState(): Promise<ModelState>;                       // startup: replaces "unknown"
    load(onProgress?: ProgressCallback): Promise<Runtime>;   // downloads first if needed; concurrent calls share one load
};

/** A loaded yes/no decision model. One runtime answers any number of questions. */
export type DecisionRuntime = {
    probability(text: string, question: string): Promise<number>; // P(yes) for the question about the text, 0 to 1
};
