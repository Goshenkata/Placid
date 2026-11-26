import { test } from "node:test";
import assert from "node:assert/strict";
import { hasGpuAdapter } from "../src/utils/webgpu.ts";

// Pretends to be the browser: navigator.gpu.requestAdapter() returns `adapter`.
function fakeNavigator(gpu: unknown) {
    Object.defineProperty(globalThis, "navigator", { value: { gpu }, configurable: true });
}

test("no WebGPU at all", async () => {
    fakeNavigator(undefined);
    assert.equal(await hasGpuAdapter(), false);
});

test("WebGPU without an adapter", async () => {
    fakeNavigator({ requestAdapter: async () => null });
    assert.equal(await hasGpuAdapter(), false);
});

test("a software fallback adapter (SwiftShader) is not a GPU", async () => {
    fakeNavigator({ requestAdapter: async () => ({ info: { isFallbackAdapter: true } }) });
    assert.equal(await hasGpuAdapter(), false);
});

test("a real GPU adapter", async () => {
    fakeNavigator({ requestAdapter: async () => ({ info: { isFallbackAdapter: false } }) });
    assert.equal(await hasGpuAdapter(), true);
});
