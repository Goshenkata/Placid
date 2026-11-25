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

test("unload() frees the session, and the next load() loads again", async () => {
    const model: any = new D1OmniModel();
    let disposed = 0;
    let loads = 0;
    model.downloadAndLoad = async () => {
        loads++;
        return { ...fakeSession, dispose: async () => { disposed++; } };
    };

    await model.load();
    await model.unload();
    assert.equal(disposed, 1);
    assert.equal(model.state, "downloaded");

    await model.load();
    assert.equal(loads, 2);
    assert.equal(model.state, "ready");
});
