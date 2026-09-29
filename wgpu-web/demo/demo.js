"use strict";

// A classic script over the standalone build (../dist/wgpu_web.js), so the
// page also works when opened from disk.
const output = document.querySelector("#output");
const CACHE_KEY = "wgpu-fft.pipeline-cache.v1";
let contextPromise;

// The build is not checked in; say how to get it.
if (typeof wgpuWeb === "undefined") {
  for (const button of document.querySelectorAll("button")) button.disabled = true;
  output.textContent =
    "wgpu-web/dist/wgpu_web.js is missing. Build it from the repository root with\n\n" +
    "  python wgpu-web/build_standalone.py\n\n" +
    "or download wgpu_web.js from a release (or from the wgpu_web artifact of a CI run)\n" +
    "into wgpu-web/dist/, then reload this page.";
}

async function context() {
  if (!contextPromise) {
    contextPromise = (async () => {
      const { WgpuFft } = await wgpuWeb.load();
      const fft = await WgpuFft.init();
      const cached = readCache();
      if (cached) await fft.importSnapshot(cached);
      return fft;
    })();
  }
  return contextPromise;
}

// Storage can be unavailable, for example for some file:// pages.
function readCache() {
  try {
    return localStorage.getItem(CACHE_KEY);
  } catch {
    return null;
  }
}

function writeCache(snapshot) {
  try {
    localStorage.setItem(CACHE_KEY, snapshot);
  } catch {
    // The cache only speeds up the next visit.
  }
}

function asF32(bytes) {
  return new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
}

function describeContext(fft) {
  return {
    adapter: fft.adapterName,
    backend: fft.backend,
    limits: {
      maxBufferSize: fft.maxBufferSize,
      maxStorageBufferBindingSize: fft.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: fft.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: fft.maxComputeInvocationsPerWorkgroup,
    },
    df64Available: fft.df64Available,
    df64CanaryWords: fft.df64CanaryWords,
    df64CanaryError: fft.df64CanaryError,
  };
}

document.querySelector("#runFft").addEventListener("click", async () => {
  try {
    output.textContent = "Loading WebAssembly and WebGPU...";
    const { WebFftDirection, WebFftNormalization, WebFftPrecision } = await wgpuWeb.load();
    const fft = await context();
    const plan = await fft.createPlan(
      4,
      1,
      WebFftDirection.Forward,
      WebFftPrecision.F32,
      WebFftNormalization.None,
    );
    const gpuInput = fft.upload(new Float32Array([1, 0, 2, 0, 3, 0, 4, 0]));
    const gpuOutput = fft.createBuffer(plan.outputBytes);
    await plan.execute(gpuInput, gpuOutput);
    writeCache(fft.exportSnapshot());
    output.textContent = JSON.stringify(
      { ...describeContext(fft), transform: "C2C", route: plan.route,
        output: Array.from(asF32(await fft.download(gpuOutput))) },
      null,
      2,
    );
  } catch (error) {
    output.textContent = error?.stack ?? String(error);
  }
});

document.querySelector("#runNufft").addEventListener("click", async () => {
  try {
    output.textContent = "Planning a browser GPU NUFFT...";
    const { WebFftPrecision, WebNufftModeOrder } = await wgpuWeb.load();
    const fft = await context();
    const plan = await fft.createNufftType1Plan(
      new Uint32Array([8]),
      1,
      1,
      1e-4,
      1,
      WebNufftModeOrder.Centered,
      2.0,
      WebFftPrecision.F32,
    );
    const points = fft.upload(new Float32Array([0]));
    const strengths = fft.upload(new Float32Array([1, 0]));
    const gpuOutput = fft.createBuffer(plan.outputBytes);
    await plan.execute(points, strengths, gpuOutput, 1);

    let df64Split = null;
    if (fft.df64Available) {
      const splitBuffer = fft.uploadDf64(new Float64Array([1 / 3]));
      df64Split = Array.from(asF32(await fft.download(splitBuffer)));
    }
    output.textContent = JSON.stringify(
      { ...describeContext(fft), transform: plan.kind, dimensions: plan.dimensions,
        pointBytes: plan.pointBytes, inputBytes: plan.inputBytes,
        outputBytes: plan.outputBytes,
        output: Array.from(asF32(await fft.download(gpuOutput))), df64Split },
      null,
      2,
    );
  } catch (error) {
    output.textContent = error?.stack ?? String(error);
  }
});
