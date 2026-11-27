/** True when WebGPU can use a real GPU. A software fallback (like Chromium's SwiftShader) counts as none: it's slower than the CPU backend. */
export async function hasGpuAdapter(): Promise<boolean> {
    const adapter = await globalThis.navigator?.gpu?.requestAdapter();
    return adapter != null && !adapter.info.isFallbackAdapter;
}
