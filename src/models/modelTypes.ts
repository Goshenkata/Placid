export type ModelState =
    | "unknown"        // not checked yet (startup)
    | "not-downloaded"
    | "downloading"
    | "downloaded"     // in cache, not in memory
    | "loading"
    | "ready"          // in memory, load() has returned the session
    | "error";

export type ProgressCallback = (percent: number) => void;

/** A model's files and lifecycle: is it downloaded, and loading it. No inference. */
export type Model<Session> = {
    readonly state: ModelState;
    checkState(): Promise<ModelState>;                       // startup: replaces "unknown"
    load(onProgress?: ProgressCallback): Promise<Session>;   // downloads first if needed; concurrent calls share one load
    unload(): Promise<void>;                                 // frees memory; sessions from load() stop working, files stay cached
};

/** A loaded model that gives P(yes) for a yes/no question about a text. One session answers any number of questions. */
export type ProbabilitySession = {
    probability(text: string, question: string): Promise<number>; // P(yes) for the question about the text, 0 to 1
};
