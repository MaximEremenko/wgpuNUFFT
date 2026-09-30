# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GPU plans in six, seven, and eight dimensions: `MAX_GPU_NUFFT_DIMENSIONS`
  is now 8, like `MAX_NUFFT_DIMENSIONS`, for type 1, 2, and 3 in all three
  precisions. The README's new "Higher dimensions" section explains which
  sizes and tolerances are practical.
- `NufftError::F32ToleranceUnreachable`: `F32` plans, on the GPU and on the
  CPU, reject tolerances their rounding would miss by far after the
  deconvolution, which happens with an upsampling factor below 2 in two or
  more dimensions.
- `wgpu-web/build_standalone.py` builds `wgpu-web/dist/wgpu_web.js`, a
  standalone browser build: one classic script with the WebAssembly module
  embedded, loaded with `wgpuWeb.load()`. It works in pages opened from
  disk, so the demo opens without a server. CI builds it on every push and
  attaches it to releases; it is not checked in.
- wgpu-web: `adapterVendorName` and `adapterArchitecture` report the
  browser's names for the GPU, and `adapterName`, which browsers usually
  leave empty, falls back to them. The demo shows the accelerator on load.

### Changed

- The 4D and 5D GPU paths are rewritten, and serve 4D to 8D. Type 1 no
  longer lets every fine-grid cell walk its `8^d` neighbouring cells and
  refold each point it finds, which made it slower than the CPU backend and,
  with clustered points, able to run one dispatch past the Windows watchdog.
  Points are binned once in original order, and workgroups spread blocks of
  cells from the points whose support reaches them; blocks with too many
  points are split into parts added in a fixed order, so results stay
  bitwise repeatable. Type 2 interpolates the binned points with lanes that
  read contiguous rows.
- Type-1 GPU plans in `F64` and `Df64`, and on grids or devices the tuned
  `F32` spreaders of 1D to 3D cannot serve, spread through the same
  rank-generic block spreader, as does the type-3 outer spread. The per-cell
  gather they used made every fine-grid cell walk its points serially, so
  dense or clustered points were slow and could also run past the watchdog.
- One- and two-dimensional type-1 GPU plans in every precision now spread
  through the rank-generic spreader, which is faster than the former 1D and
  2D gathers on dense and clustered points. That spreader sizes the parts of
  every crowded block from its actual points, so a dense cluster spreads
  over many workgroups, and sizes its workgroups per rank and precision.
- `NufftPlan::set_points_gpu` now prepares 4D and higher point sets once,
  so later executions skip the binning, as they do in 1D to 3D.
- Large executions of the rank-generic paths are recorded as several
  dispatches of bounded work.

### Removed

- The `nd_prototype/` research prototype, whose design now ships as the
  rank-generic GPU paths.

## [0.2.0] - 2026-09-28

A CPU backend for all three transform types, in Rust and Python, with point
sets prepared once for repeated transforms.

### Added

- `CpuNufftPlan` (types 1 and 2) and `CpuNufftType3Plan`: plans in host
  memory, behind the default `cpu` feature. They take the same
  configurations and layouts as the GPU plans, support one to eight
  dimensions and all three precisions, check every coordinate, and give
  bitwise-identical results for any thread count. They transform their fine
  grid in place, on a `wgpu-fft` whose CPU backend runs strided axes in place
  and long lines as a parallel four-step decomposition.
- Prepared CPU point sets: `prepare_points` checks and sorts a point set
  once (`CpuNufftPoints`), and `execute_prepared` runs any number of
  transforms on it with results bitwise identical to `execute`. One set
  serves the type-1 and type-2 plans of a configuration. Type-3 plans
  prepare a source and a target set together (`CpuNufftType3Points`),
  including their rescaling and phases, which makes repeated type-3
  transforms about twice as fast.
- `NufftError::PointSetMismatch` for a prepared point set passed to a plan
  it does not fit.
- Python: `CpuType1Plan`, `CpuType2Plan`, and `CpuType3Plan` run the CPU
  plans on NumPy arrays without a GPU, in `complex64` or `complex128`
  (`dtype`, `complex128` by default), with a `threads` option. They read
  contiguous inputs in place, release the GIL, and run concurrently on one
  plan. `prepare_points` returns a `CpuPoints` or `CpuType3Points` set that
  `execute` takes in place of the point arrays. CI runs their tests.

### Changed

- The `wgpu-fft` submodule is pinned to its v0.1.0 release.

## [0.1.0] - 2026-09-28

First tagged release of `wgpu-nufft`, its Python binding, and the `wgpu-web`
browser wrapper, built on `wgpu` 30 and the pinned `wgpu-fft` submodule.

### Added

- Type-1, type-2, and type-3 NUFFT plans that run on any `wgpu` backend,
  including browser WebGPU, and record into a caller's command encoder with
  caller-owned buffers.
- GPU plans in one to five dimensions: dedicated 1D, 2D, and 3D paths and a
  rank-generic path for 4D and 5D (`MAX_GPU_NUFFT_DIMENSIONS`). Direct `f64`
  reference transforms cover up to eight dimensions.
- `f32`, native `f64` (with `SHADER_F64`), and portable double-float (`Df64`)
  precision.
- Many-vector batching over a shared point set, and `set_points_gpu` /
  `execute_type{1,2}_gpu[_batch]` to prepare a point set once for repeated
  transforms.
- Bitwise-repeatable results without floating-point atomics: coarse-bin
  spreaders and interpolators for `f32` (1D segments, 2D tiles, 3D blocks, and
  dense type-3 grids), and a per-cell gather elsewhere.
- Parallel sorting of large bins and heavy gather passes, so tightly clustered
  points stay fast and deterministic.
- One compute pass per execution, fine-grid FFTs included, through
  `wgpu-fft`'s `FftRecorder`; the pinned `wgpu-fft` also runs long
  power-of-two FFTs faster.
- GPU stage profiling with timestamp queries behind the `gpu-profiling`
  feature.
- A PyO3 + NumPy Python binding and a `wasm-bindgen` browser wrapper.
- Re-exports of `wgpu` and `wgpu_fft`, so consumers can use the exact versions
  the plans are built on.
- CI for formatting, lints, tests, the minimum Rust version, WebAssembly,
  documentation, and the Python binding, and compiled README examples.

### Changed

These affect code written against earlier development snapshots.

- `encode_type1_gpu_batch` and `encode_type2_gpu_batch` take `active_batch`
  before `point_count`, like the type-3 methods.
- `required_type1_strength_buffer_size_bytes` and
  `required_type2_output_buffer_size_bytes` are plan methods that size the
  configured batch in the configured precision. The `f32`-only static
  `required_type1_point_buffer_size_bytes` and
  `required_type2_point_buffer_size_bytes` are removed; use
  `required_point_buffer_size_bytes`.
- GPU plans reject more than five dimensions with
  `NufftError::GpuDimensionsUnsupported`.

### Fixed

- A buffer or device from another `wgpu` device returns
  `NufftError::GpuDeviceMismatch` instead of panicking.
- A huge `sigma` returns `NufftError::FineGridTooLarge` instead of scanning for
  a fine-grid length indefinitely.
- Coordinates outside `[-3*pi, 3*pi]`, NaN included, can no longer corrupt
  memory or other points' results.
- DX12 shader compilation no longer stalls on workgroup zero-initialization.
- Python raises `WgpuNufftError` instead of `PanicException`, and the source
  distribution builds with a single license file.
- In the browser, `free()` destroys a buffer's GPU memory at once and the
  device once every object of its context is freed, and Rust panics are
  reported through `console.error`.

[Unreleased]: https://github.com/MaximEremenko/wgpuNUFFT/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/MaximEremenko/wgpuNUFFT/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MaximEremenko/wgpuNUFFT/releases/tag/v0.1.0
