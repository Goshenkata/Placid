import { test } from "node:test";
import assert from "node:assert/strict";
import { D1OmniModel } from "../src/models/discriminative/d1OmniModel.ts";

// These tests replace the private download/cache methods, so nothing touches the network or GPU.
const fakeSession = { probability: async () => 0.5 };

test("checkState replaces unknown with what's in the cache", async () => {
    const cached: any = new D1OmniModel();
    cached.modelFilesInBrowserCache = async () => true;
    assert.equal(cached.state, "unknown");
    assert.equal(await cached.checkState(), "downloaded");

    const notCached: any = new D1OmniModel();
    notCached.modelFilesInBrowserCache = async () => false;
    assert.equal(await notCached.checkState(), "not-downloaded");
});

test("concurrent load() calls share one load", async () => {
    const model: any = new D1OmniModel();
    let loads = 0;
    model.downloadAndLoad = async () => {
        loads++;
        return fakeSession;
    };

    const sessions = await Promise.all([model.load(), model.load(), model.load()]);

    assert.equal(loads, 1);
    assert.ok(sessions.every((session) => session === fakeSession));
    assert.equal(model.state, "ready");
});

test("a failed load sets error, and the next load() tries again", async () => {
    const model: any = new D1OmniModel();
    model.downloadAndLoad = async () => {
        throw new Error("network down");
    };
    await assert.rejects(model.load(), /network down/);
    assert.equal(model.state, "error");

    model.downloadAndLoad = async () => fakeSession;
    assert.equal(await model.load(), fakeSession);
    assert.equal(model.state, "ready");
});

test("every waiting load() caller gets progress", async () => {
    const model: any = new D1OmniModel();
    model.downloadAndLoad = async () => {
        await new Promise((resolve) => setTimeout(resolve, 0)); // real progress comes after a network wait
        for (const listener of model.progressListeners) listener(50); // what transformers.js's progress does
        return fakeSession;
    };
    const first: number[] = [];
    const second: number[] = [];

    await Promise.all([model.load((percent: number) => first.push(percent)), model.load((percent: number) => second.push(percent))]);

    assert.deepEqual(first, [50]);
    assert.deepEqual(second, [50]);
    assert.equal(model.progressListeners.size, 0); // removed once load() returns
});

test("progress adds up the bytes of every file being loaded", () => {
    const model: any = new D1OmniModel();
    const percents: number[] = [];
    model.progressListeners.add((percent: number) => percents.push(percent));
    model.needsDownload = true;
    const embedTokens = (info: object) => model.reportProgress("embed_tokens", info);
    const decision = (info: object) => model.reportProgress("decision", info);

    embedTokens({ status: "progress_total", loaded: 0, total: 100 });
    decision({ status: "progress_total", loaded: 0, total: 300 });
    embedTokens({ status: "progress_total", loaded: 100, total: 100 }); // first file done: 100 of 400 bytes
    assert.equal(percents.at(-1), 25);
    assert.equal(model.state, "downloading");

    decision({ status: "progress_total", loaded: 0, total: 900 }); // decision's big part starts: total grows, percent would dip to 10
    assert.equal(percents.at(-1), 25); // never reported lower

    decision({ status: "progress_total", loaded: 900, total: 900 });
    assert.equal(percents.at(-1), 100);
    assert.equal(model.state, "loading"); // downloaded, now loading into memory
});

test("checkState says not-downloaded when the cache check fails (offline)", async () => {
    const model: any = new D1OmniModel();
    model.modelFilesInBrowserCache = async () => {
        throw new Error("fetch failed");
    };
    assert.equal(await model.checkState(), "not-downloaded");
});

test("a failed load frees the files it already loaded", async () => {
    const model: any = new D1OmniModel();
    let disposed = 0;
    model.downloadAndLoad = async () => {
        model.loadedModels.push({ dispose: async () => disposed++ }); // one file loaded...
        throw new Error("second file failed"); // ...then the other failed
    };
    await assert.rejects(model.load(), /second file failed/);
    assert.equal(disposed, 1);
    assert.deepEqual(model.loadedModels, []);
});
