# wgpu-web

JavaScript bindings of `wgpu-fft` and `wgpu-nufft` for the browser, built
with `wasm-bindgen` on WebGPU. Plans, inputs, and outputs stay on the GPU
until an explicit `download`.

## Build and demo

From the `wgpuNUFFT` repository root:

```bash
wasm-pack build wgpu-web --target web --out-dir pkg
python -m http.server 8000 --directory wgpu-web
```

Then open <http://localhost:8000/demo/> in a browser with WebGPU. The
generated `pkg/` directory is not checked in.

## Example

A 1D type-1 transform of 1,000 points onto 64 Fourier modes:

```js
import init, { WebFftPrecision, WebNufftModeOrder, WgpuFft } from "./pkg/wgpu_web.js";

await init(); // loads the WebAssembly module
const gpu = await WgpuFft.init();

const pointCount = 1000;
const plan = await gpu.createNufftType1Plan(
  new Uint32Array([64]), // modes per dimension
  pointCount,
  1,                     // batch capacity
  1e-6,                  // eps
  1,                     // isign
  WebNufftModeOrder.Centered,
  2.0,                   // sigma
  WebFftPrecision.F32,
);

const coordinates = new Float32Array(pointCount).map(() => (2 * Math.random() - 1) * Math.PI);
const strengths = new Float32Array(2 * pointCount);
for (let j = 0; j < pointCount; j++) strengths[2 * j] = 1; // (re, im) = (1, 0)

const points = gpu.upload(coordinates);
const input = gpu.upload(strengths);
const output = gpu.createBuffer(plan.outputBytes);
await plan.execute(points, input, output, 1); // active batch

const bytes = await gpu.download(output);
const modes = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
// modes holds 64 interleaved (re, im) pairs.

for (const object of [points, input, output, plan, gpu]) object.free();
```

## API

| Call | Description |
|---|---|
| `WgpuFft.init()` | Opens WebGPU with the adapter's maximum limits, falling back to the defaults if the browser rejects them. |
| `WgpuFft.initWithDefaultLimits()` | Opens WebGPU with exactly the default limits and no optional features, the most conservative portable device. |
| `WgpuFft.initFallback()` | Opens the browser's software adapter, such as SwiftShader, where no GPU adapter is available. |
| `createPlan(length, batch, direction, precision, normalization)` | A reusable 1D C2C FFT plan; `execute(input, output)`. |
| `createNufftType1Plan(nModes, pointCount, batch, eps, isign, modeOrder, sigma, precision)` | A type-1 plan; `execute(points, strengths, output, activeBatch)`. |
| `createNufftType2Plan(nModes, pointCount, batch, eps, isign, modeOrder, sigma, precision)` | A type-2 plan; `execute(points, coefficients, output, activeBatch)`. |
| `createNufftType3Plan(sourceBounds, targetBounds, sourceCount, targetCount, batch, eps, isign, sigma, precision)` | A type-3 plan; `execute(sources, strengths, targets, output, activeBatch)`. |
| `upload(Float32Array)`, `uploadDf64(Float64Array)`, `createBuffer(byteLength)` | Create GPU buffers. |
| `download(buffer)` | Resolves to the buffer's bytes as a `Uint8Array`. |
| `exportSnapshot()`, `importSnapshot(json)` | Save the pipeline cache (validated shader sources and pipeline keys) as JSON, and prewarm a new context from it; the demo keeps it in `localStorage`. |

Plans report the byte sizes their buffers need (`pointBytes`, `inputBytes`,
and `outputBytes`, or `sourcePointBytes`, `targetPointBytes`, `strengthBytes`,
and `outputBytes` for type 3). An awaited `execute` covers encoding,
submission, and queue completion, so reusing its buffers makes it a clean span
to time.

## Data layout

- `F32` complex values are interleaved `re, im` words (8 bytes each).
- `Df64` complex values are `re_hi, re_lo, im_hi, im_lo` words (16 bytes
  each). `uploadDf64` splits every `f64` into an `f32` high word and the
  residual low word, so coordinates become `[x_hi, x_lo, ...]`.
- Coordinates are point-major, while strengths, coefficients, and outputs are
  transform-major.
- Type-1 and type-2 mode shapes are `Uint32Array`s. Type-3 bounds are
  `Float64Array`s of `[lower0, upper0, lower1, upper1, ...]`. The plan copies
  these arrays, but keep them unchanged until its creation Promise settles.

Each plan fixes its point counts and batch capacity; `execute` accepts any
active batch from 1 to that capacity. Plans and buffers must come from the
same context, and one buffer cannot serve as both input and output.

GPU buffers cannot be inspected when a plan executes, so the caller must keep
coordinates valid: type-1 and type-2 points finite and inside
`[-3*pi, 3*pi]`, type-3 sources and targets finite and inside the bounds the
plan was created with.

## Precision

`WgpuFft.init()` runs `wgpu-fft`'s 96-word double-float arithmetic canary
suite on the browser's shader compiler. If a word fails, `F32` keeps working
and `Df64` plans are rejected with the canary failure; `df64Available`,
`df64CanaryWords`, and `df64CanaryError` report the outcome. Native `F64`
reaches `wgpu-fft`, which returns its `device-missing-shader-f64` error,
because WebGPU has no 64-bit float shaders. The browser test matrix covers
one to three dimensions.

## Releasing GPU memory

JavaScript garbage collection does not see GPU memory, so call `free()` on
objects you no longer need. Freeing a buffer destroys its GPU allocation at
once. A freed plan's internal buffers wait for garbage collection, but once
the context and every plan and buffer created from it have been freed, the
WebGPU device is destroyed and all of its memory is released.

Rust panics are reported through `console.error` before the browser raises
its `unreachable` trap.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
