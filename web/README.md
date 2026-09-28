# Browser test harness

Browser checks of `wgpu-fft`, `wgpu-nufft`, and the `wgpu-web` package in
Chrome's WebGPU implementation. Run them from the `wgpuNUFFT` repository
root:

| Runner | Checks |
|---|---|
| `web\run_browser_tests.cmd` | FFT smoke, correctness-matrix, df64-canary, and large-route tests, the NUFFT browser matrix, and a `wgpu-web` NUFFT smoke test. |
| `web\run_phase_c_browser.cmd` | Pipeline-cache persistence across documents, and a timing comparison with the JavaScript WebGPU-FFT library. |
| `web\run_phase_d_browser.cmd` | The JavaScript type-1, type-2, and type-3 NUFFT surface against an independent `f64` NDFT, at the WebGPU default limits. |

## Browser tests

The tests use `wasm-bindgen-test-runner` and ChromeDriver. Install the
`wasm32-unknown-unknown` target and the `wasm-bindgen-cli` version in
`Cargo.lock`, and point `CHROMEDRIVER` at a ChromeDriver matching the
installed Chrome (or put `chromedriver.exe` on `PATH`):

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
$env:CHROMEDRIVER = 'C:\path\to\chromedriver.exe'
web\run_browser_tests.cmd
```

The runner loads the checked-in root `webdriver.json`, which sets the
WebGPU-related Chrome flags, and forces ChromeDriver so that
`wasm-bindgen-test-runner` does not pick another WebDriver from `PATH`. The
FFT tests run through the `wgpuFFT/` submodule's manifest, and the NUFFT and
`wgpu-web` tests through the outer workspace. The large FFT cases allocate
about 1 GiB of transient GPU memory, so only this runner enables them
(`WGPU_FFT_RUN_BROWSER_LARGE_TESTS=1`); ordinary `cargo test` runs skip them.

## Pipeline cache and JavaScript comparison

The comparison needs the JavaScript
[WebGPU-FFT](https://github.com/MaximEremenko/WebGPU-FFT) repository checked
out next to this one, as `../WebGPU-FFT`, at the pinned revision:

```powershell
git clone https://github.com/MaximEremenko/WebGPU-FFT.git ..\WebGPU-FFT
git -C ..\WebGPU-FFT checkout fa45c93f524a69a96c9f55acfad865226bfccd29
web\run_phase_c_browser.cmd --mode all
```

`--mode` is `snapshot`, `bench`, or `all`. `--output <file>` writes the
complete result as JSON, for example `--output target\browser\comparison.json`.

The runner builds `wgpu-web`, runs `wasm-bindgen`, and serves this repository
and the sibling checkout from one origin. It refuses a sibling checkout at
another revision or with uncommitted changes, and records the `wgpuNUFFT`,
`wgpuFFT`, and WebGPU-FFT revisions. It first tries headless Chrome for up to
30 seconds, then falls back to a headed app window; `--headed-only` skips the
headless attempt on machines where headless WebGPU is unavailable.

- **Pipeline cache.** The page creates a plan, exports its versioned JSON
  snapshot to `localStorage` under `wgpu-fft.pipeline-cache.v1`, reads it back
  in a fresh same-origin document, imports it into a fresh WebGPU context,
  recreates the plan, and validates an impulse transform.
- **Comparison.** One out-of-place `f32` C2C forward transform of N = 4096
  with batch 1024. Plan creation, allocation, upload, and download are
  outside the timed region; each timed iteration covers command encoding, one
  queue submission, and `onSubmittedWorkDone`, and both libraries reuse their
  buffers. Five warm-ups precede three alternating 20-iteration blocks,
  reported as the mean milliseconds per transform with the standard error
  across blocks. Both libraries run under the same browser, shader compiler,
  and GPU, and first request the adapter's limits without optional features,
  falling back to the browser defaults only if Chrome rejects them. The result
  records the active limits and the Rust route. This compares the libraries
  end to end, not their generated shaders.

## Browser NUFFT matrix

```powershell
web\run_phase_d_browser.cmd --headed-only --output target\browser\nufft-matrix.json
```

The runner rebuilds `wgpu-web`, generates the browser module with
`wasm-bindgen`, and launches Chrome. The page runs batched type-1, type-2, and
type-3 plans in one to three dimensions, in `F32` and `Df64`, at the exact
WebGPU default limits, and compares every result with an independent
JavaScript `f64` NDFT. It also verifies all 96 df64 canary words, checks that
the `Df64` type-3 phase bound accepts 1024 and rejects 1025, and confirms
that native `F64` is unavailable in the browser. Without `--headed-only`, it
first tries headless Chrome for up to 30 seconds.
