# wgpuNUFFT

[![CI](https://github.com/MaximEremenko/wgpuNUFFT/actions/workflows/ci.yml/badge.svg)](https://github.com/MaximEremenko/wgpuNUFFT/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 1.92+](https://img.shields.io/badge/rust-1.92%2B-orange.svg)](Cargo.toml)

Nonuniform fast Fourier transforms (NUFFTs) on the GPU through `wgpu`
(Vulkan, DX12, Metal) and in the browser through WebGPU. Results are bitwise
repeatable: no floating-point atomics, and every output is a sum in a fixed
order.

The Cargo package is `wgpu-nufft`, imported as `wgpu_nufft`. Its FFTs come
from [wgpu-fft](https://github.com/MaximEremenko/wgpuFFT), pinned here as the
`wgpuFFT/` Git submodule.

## Features

- **All three transform types**: type 1 (nonuniform points to Fourier modes),
  type 2 (modes to points), and type 3 (nonuniform points to nonuniform
  frequencies).
- **One to eight dimensions on the GPU**, in all three precisions: type 1
  spreads through one rank-generic spreader in every rank, and type 2
  interpolates through tuned 1D, 2D, and 3D paths and a rank-generic path
  for 4D to 8D.
  The direct `f64` reference transforms cover eight dimensions too. See
  [Higher dimensions](#higher-dimensions) for what is practical where.
- **Three precisions**: `f32`, native `f64` (devices with `SHADER_F64`), and
  portable double-float (`Df64`, about 44-48 significant bits on any device).
- **Deterministic**: repeated executions give bitwise-identical results, also
  for tightly clustered points.
- **GPU-resident**: plans record into your command encoder and read and write
  your buffers, so data never has to leave the GPU.
- **Batches and point reuse**: many vectors over one point set, and a point
  set prepared once for repeated transforms.
- **CPU backend**: `CpuNufftPlan` and `CpuNufftType3Plan` run all three types
  on the CPU in one to eight dimensions, with the same configuration and
  layouts and the same bitwise repeatability, for any thread count, and
  prepare point sets once for repeated transforms.
- **Python and JavaScript**: a PyO3 + NumPy binding with the GPU and CPU
  plans, and a `wasm-bindgen` browser package.

## Installation

`wgpu-nufft` is distributed through GitHub rather than crates.io:

```toml
[dependencies]
wgpu-nufft = { git = "https://github.com/MaximEremenko/wgpuNUFFT", tag = "v0.2.0" }
```

Cargo checks out the `wgpuFFT` submodule automatically. The default `cpu`
feature adds the [CPU backend](#cpu-backend); `default-features = false`
leaves it out. Use the re-exported
`wgpu_nufft::wgpu` and `wgpu_nufft::wgpu_fft` crates (or `wgpu = "30"`), so
your types match the ones the plans take. A separate Git dependency on
`wgpu-fft` would add a second, incompatible copy. Rust 1.92 or newer is
required.

To work on a local checkout, clone it with its submodule and depend on it by
path:

```bash
git clone --recurse-submodules https://github.com/MaximEremenko/wgpuNUFFT.git
# In an existing clone: git submodule update --init --recursive
```

```toml
[dependencies]
wgpu-nufft = { path = "../wgpuNUFFT" }
```

## Quick start

The direct `f64` reference transform runs on the CPU, which makes it a quick
check that the dependency works:

```rust
use wgpu_nufft::{reference_type1_f64, Complex64, NufftConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NufftConfig::new([8], 1.0e-6);
    let points = [0.0, 0.25];
    let strengths = [Complex64::new(1.0, 0.0), Complex64::new(0.5, -0.25)];
    let modes = reference_type1_f64(&config, &points, &strengths)?;

    assert_eq!(modes.len(), 8);
    Ok(())
}
```

A GPU plan records its work into your command encoder and reads and writes
your buffers, so results can stay on the GPU. This 2D type-1 transform spreads
1,000 points onto 64 x 64 Fourier modes:

```rust,no_run
use wgpu_nufft::wgpu::{self, util::DeviceExt};
use wgpu_nufft::{NufftConfig, NufftPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Any wgpu device works; wgpu-fft can pick the default adapter.
    let context = pollster::block_on(wgpu_nufft::wgpu_fft::device::request_default_device())
        .ok_or("no GPU adapter")?;
    let (device, queue) = (&context.device, &context.queue);
    let plan = NufftPlan::type1_gpu(device, queue, NufftConfig::new([64, 64], 1.0e-6))?;

    // Point-major coordinates [x0, y0, x1, y1, ...] in [-pi, pi) and
    // interleaved complex strengths [re0, im0, re1, im1, ...].
    let point_count = 1000;
    let points: Vec<f32> = (0..2 * point_count)
        .map(|i| (i as f32 * 0.618).fract() * 6.28 - 3.14)
        .collect();
    let strengths = vec![1.0f32; 2 * point_count];
    let storage = |contents: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(contents),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let (points, strengths) = (storage(&points), storage(&strengths));
    let modes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("modes"),
        size: plan.required_type1_output_buffer_size_bytes()?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    plan.encode_type1_gpu(device, &mut encoder, point_count, &points, &strengths, &modes)?;
    queue.submit([encoder.finish()]);
    // `modes` now holds 64 * 64 interleaved complex values, axis zero fastest.
    Ok(())
}
```

Type-2 plans (`NufftPlan::type2_gpu`) and type-3 plans (`NufftType3Plan`)
follow the same pattern. Run `cargo doc --open` for the full API.

## CPU backend

`CpuNufftPlan` (types 1 and 2) and `CpuNufftType3Plan` run in host memory,
for machines without a usable GPU or for checking GPU results. They take the
same configurations and the same layouts as the GPU buffers, in one to eight
dimensions. `F32` plans compute in `f32`; `F64` and `Df64` plans compute in
`f64` and take `f64` words or `Df64` word pairs.

```rust
use wgpu_nufft::{CpuNufftPlan, NufftConfig};

fn main() -> Result<(), wgpu_nufft::NufftError> {
    // Type 2 on the CPU: 32 x 32 Fourier coefficients at 1,000 points.
    let plan = CpuNufftPlan::type2(NufftConfig::new([32, 32], 1.0e-6))?;
    let point_count = 1000;
    let points: Vec<f32> = (0..2 * point_count)
        .map(|i| (i as f32 * 0.618).fract() * 6.28 - 3.14)
        .collect();
    let coefficients = vec![1.0f32; plan.required_input_len(point_count)?];
    let mut values = vec![0.0f32; plan.required_output_len(point_count)?];
    plan.execute(&points, &coefficients, &mut values)?;
    Ok(())
}
```

`execute` checks and sorts its points on every call. To run many transforms
on one point set, as iterative solvers do, prepare the points once with
`prepare_points` and run `execute_prepared`. A prepared set serves the
type-1 and type-2 plans of one configuration, and gives results bitwise
identical to `execute`. `CpuNufftType3Plan::prepare_points` does the same
for a source and a target set, including their phases.

```rust
use wgpu_nufft::{CpuNufftPlan, NufftConfig};

fn main() -> Result<(), wgpu_nufft::NufftError> {
    let config = NufftConfig::new([32, 32], 1.0e-6);
    let forward = CpuNufftPlan::type2(config.clone())?;
    let adjoint = CpuNufftPlan::type1(config)?;
    let points: Vec<f32> = (0..2000)
        .map(|i| (i as f32 * 0.618).fract() * 6.28 - 3.14)
        .collect();
    let prepared = forward.prepare_points(&points)?;
    let mut image = vec![1.0f32; forward.required_input_len(prepared.len())?];
    let mut values = vec![0.0f32; forward.required_output_len(prepared.len())?];
    for _ in 0..10 {
        forward.execute_prepared(&prepared, &image, &mut values)?;
        adjoint.execute_prepared(&prepared, &values, &mut image)?;
    }
    Ok(())
}
```

The CPU plans sort the points into bins and split the fine grid into slabs
along its last axis. Each slab adds the points that reach it in a fixed
order, so results are bitwise identical for any thread count
(`with_threads` sets it; the default is the machine's parallelism). They
check every coordinate and return an error for one outside
`[-3*pi, 3*pi]`. The kernel is evaluated from the same piecewise
polynomials as the GPU `F64` and `Df64` paths, and the fine-grid FFT runs on
wgpu-fft's CPU backend. A plan keeps its fine-grid buffers between
executions. Type-3 plans compose these steps as on the GPU and check every
source and target against the plan's intervals.

## Data layout

| Data | Layout |
|---|---|
| Points | Point-major, in the plan's precision: one value per point in 1D, `[x0, y0, z0, x1, y1, z1, ...]` in 3D. |
| Complex values | Interleaved `(re, im)` pairs; `Df64` stores `(re_hi, re_lo, im_hi, im_lo)`. |
| Fourier modes | Dimension zero fastest, in centered order by default (`ModeOrder::Fft` selects FFT order). |
| Batches | Transform-major, `[transform][point or mode]`, with every transform sharing one point set. |

Every buffer needs `STORAGE` usage. Size buffers with the plan's
`required_*_buffer_size_bytes` methods, which follow its dimensionality,
precision, and batch; the `*_for_batch` variants size a smaller active batch.

Type-1 and type-2 coordinates are radians in `[-3*pi, 3*pi]`. Type-3 sources
and targets must stay inside the intervals given when the plan is created.
See [Conventions](#conventions) for the exact definitions.

## Python

[`python/`](python/README.md) builds a native NumPy extension with PyO3 and
maturin. It runs reusable type-1, type-2, and type-3 GPU plans on `float32`
coordinates and `complex64` values:

```python
import numpy as np
import wgpu_nufft

context = wgpu_nufft.Context()
plan = wgpu_nufft.Type1Plan(context, (16,), eps=1e-5)
points = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float32)
strengths = np.ones(32, dtype=np.complex64)
modes = plan.execute(points, strengths)  # shape (16,)
```

Multidimensional modes keep the NumPy shape `(*n_modes)`; the binding
converts between C order and the dimension-zero-fast storage of the Rust
core. See [python/README.md](python/README.md) for building, array shapes, and
batches.

## Browser

[`wgpu-web/`](wgpu-web/README.md) is a `wasm-bindgen` package that exposes
`wgpu-fft` and `wgpu-nufft` to JavaScript. Plans and buffers stay on the GPU
until an explicit download.

```bash
wasm-pack build wgpu-web --target web --out-dir pkg
```

`F32` and `Df64` plans run in browsers. Native `F64` does not, because WebGPU
has no 64-bit float shaders. Chrome tests at the exact WebGPU default limits
cover batched type 1, 2, and 3 in one to three dimensions in both precisions;
see [web/README.md](web/README.md) for the browser test harness.

## Precision

`NufftConfig::with_precision` and `NufftType3Config::with_precision` choose
the precision of coordinates, values, kernel, and FFT together:

| Precision | Scalar storage | Requirement | Use |
|---|---|---|---|
| `F32` (default) | `f32` | none | Every backend, including browsers. |
| `F64` | `f64` | `wgpu::Features::SHADER_F64` (Vulkan in wgpu 30) | Full double precision where the device supports it. |
| `Df64` | `hi + lo` pair of `f32` words | none | About 44-48 significant bits on DX12, Metal, and browsers. |

Plans return a `PrecisionUnsupported` error before creating shaders when the
device lacks `SHADER_F64`. In both `F64` and `Df64`, coordinates take 8 bytes
and complex values 16 bytes, which the size helpers account for. `Df64` keeps
the `f32` exponent range (about `1e-38` to `1e38`).

Native-`f64` shaders cannot call `exp`, and `Df64` uses no shader
transcendentals, so both evaluate the kernel from piecewise Horner
polynomials fitted on the host in `f64`, with the same polynomial for
spreading and deconvolution. Type-3 phases use a range-reduced polynomial
sine and cosine. `F64` plans reject phase magnitudes above `1e6` and `Df64`
plans above `1024`, since two-word range reduction loses absolute precision
as the phase grows.

`Df64` products use Dekker splits rather than fused multiply-add. Exact-word
canaries in `wgpu-fft` check each backend's arithmetic: Vulkan and DX12 are
tested, while Metal, whose shader compiler enables fast math by default, is
untested. In the browser, `wgpu-web` runs all 96 canary words at start-up and
disables only `Df64` if one fails.

## Higher dimensions

Work and memory grow exponentially with the dimension `d`. Every point
touches `w^d` fine-grid cells, where the kernel width `w` is 7 at
`eps = 1e-6` and 4 at `eps = 1e-3`, and the oversampled grid holds `2^d`
times as many cells as there are modes. So:

- 4D and 5D are practical at the tolerances used in lower dimensions.
- 6D suits moderate tolerances and grids of about ten modes per axis.
- 7D and 8D work, but only on small grids at tolerances near `1e-3`; with
  so few modes per axis a direct sum over the modes costs about as much.

Each axis costs at least `2w` fine-grid cells, however few modes it has. A
plan returns an error when its fine grid exceeds the device's buffer or
binding limits, which in a browser allow up to 32^4 modes in 4D at
`sigma = 2`.

`F32` plans with an upsampling factor below 2 lose accuracy quickly as the
dimension grows: the kernel's Fourier transform then falls steeply towards
the edge of the mode box, and deconvolution amplifies `f32` rounding by
that fall along every axis. At `sigma = 1.25` and `eps = 1e-6`, `f32`
reaches only about `2e-5` in 2D, `3e-4` in 3D, and `1e-2` in 4D, while
`F64` and `Df64` stay near the tolerance. `F32` plans, on the GPU and on the
CPU, therefore return `F32ToleranceUnreachable` when the estimated error of
this rounding exceeds both ten times the tolerance and `1e-5`; use
`sigma = 2`, a larger tolerance, or `F64` or `Df64` precision instead.

## How it works

**Type 2** deconvolves and zero-pads the modes onto an oversampled grid, runs
a `wgpu-fft` C2C transform on it, and interpolates the grid at every point
with the exponential-of-semicircle (ES) kernel. In 1D to 3D it visits the
points in bin order for cache locality.

**Type 1** reverses these steps: it spreads the points onto the fine grid,
transforms the grid, and deconvolves and truncates it into the mode buffer.
Points are binned with portable `u32` atomics, a prefix scan, and a sort that
restores input order inside every bin. Each fine-grid cell is then written by
exactly one invocation, which adds its points in that fixed order, so no
floating-point atomics are needed.

- **One spreader for every rank.** Every type-1 plan, from 1D to 8D and in
  every precision, spreads through the rank-generic block spreader
  described with the higher dimensions below.
- **Clustered points.** Bins of more than 64 points are sorted by a parallel
  merge sort. A block that reaches too many points goes to heavy passes that
  split its points across workgroups; partial sums are added in a fixed
  order. The spreader sizes each crowded block's parts from the points
  actually in its reach, so a single dense cluster still spreads over
  thousands of workgroups.

**Four to eight dimensions** use one rank-generic implementation, whose
shaders are generated per rank and precision; its type-1 spreader also
serves every lower-rank plan. Points are binned once per point set
into coarse bins, in original order for type 1, and every point's support
is prepared. For type 1, each workgroup owns a block of fine-grid cells:
every invocation keeps a run of cells along axis zero in registers, while
the workgroup streams the points whose support reaches the block through
workgroup memory and evaluates their kernel weights once for all
invocations. Blocks with too many points in reach are split into parts
added in a fixed order: a plan pass sizes the parts from the points listed
in every crowded block's reach, with an exact sum that does not depend on
the order in which the blocks were listed. Workgroup sizes and run lengths
are chosen per rank and precision. Type 2 scatters the deconvolved modes
onto the fine grid and interpolates the binned points with one lane per
axis-zero support offset, adding the lanes in lane order. Executions whose
work would be large are recorded as several dispatches of bounded work,
which keeps the operating system's GPU watchdog from firing on devices
that preempt only between dispatches.

**Type 3** rescales and pre-phases the sources, spreads them onto an outer
uniform grid, evaluates that grid through an inner type-2 plan, and applies
the kernel correction and post-phase at the targets. The small, densely
populated outer grids of `F32` 1D and 2D plans use a spreader that sums
fixed groups of sources into partial grids and adds the partial grids in
group order; the other outer grids use the rank-generic type-1 spreader.

A GPU plan cannot read device buffers while it is created, so
`NufftType3Config` takes conservative source and target intervals, and every
later coordinate must stay finite and inside them. The CPU reference checks
this; on the GPU it is the caller's guarantee. Space-bandwidth products or
rescalings too large for GPU indices, buffers, or the `Df64` phase range
return planning errors.

Plans own and reuse their fine-grid and binning scratch, growing point-sized
buffers only when a larger point set needs them, so executions of one plan
must stay in queue order.

### Batches

`NufftConfig::with_batch(n)` and `NufftType3Config::with_batch(n)` plan `n`
transforms over one shared point set. Binning, scanning, and sorting run once
for all of them, the 1D to 3D direct interpolation kernels reuse each
kernel weight for up to four vectors, and the fine-grid FFT uses
`wgpu-fft`'s batch dimension. The ordinary encode methods run all `n`
transforms; the `*_batch` methods run any active count from 1 to `n` on the
same plan. The FFT stays sized for `n`, so a smaller count does not reduce
FFT work proportionally.

### Reusing a point set

`NufftPlan::set_points_gpu` records the point-dependent work once (binning,
sorting, and per-point preparation on the binned `F32` paths of 1D-3D
type-1 and type-2 plans and on the rank-generic paths).
`execute_type1_gpu[_batch]` and
`execute_type2_gpu[_batch]` then transform new strengths or coefficients at
those points. The plan keeps a reference to the point buffer: its contents
must not change while executions use it, and executions must be submitted
after the commands recorded by `set_points_gpu`. The `encode_*` methods are
`set_points_gpu` followed by the matching `execute_*` call.

## Testing

```bash
cargo test --locked                                     # wgpu-nufft and wgpu-web
cargo test --manifest-path wgpuFFT/Cargo.toml --locked  # the wgpu-fft submodule
```

The workspace excludes `wgpuFFT/`, so the submodule builds on its own and
workspace commands never modify it.

GPU tests are opt-in. `WGPU_FFT_RUN_GPU_TESTS=1` runs them on the default
adapter; `WGPU_BACKEND` (`vulkan`, `dx12`, or `metal`) picks a backend and
`WGPU_ADAPTER_NAME` an adapter whose name contains the given text:

```bash
WGPU_FFT_RUN_GPU_TESTS=1 cargo test --release
```

```powershell
$env:WGPU_FFT_RUN_GPU_TESTS = '1'; cargo test --release
```

| Test | Covers |
|---|---|
| `gpu_nufft` | 1D at `eps` from `1e-2` to `1e-6`, both signs and mode orders, and random, boundary, clustered, and duplicate points: relative L2 error at most `4*eps` overall and `8*eps` for each point class. |
| `gpu_nufft_2d`, `gpu_nufft_3d` | The same in 2D (at most `20*eps`) and 3D (at most `32*eps`), with non-square grids, scratch reuse, determinism, and adjoint checks. |
| `gpu_nufft_nd` | 4D to 8D against the direct sums, in all three precisions, with batches, both signs and mode orders, adjoint checks, and bitwise point reuse. |
| `gpu_nufft_type3` | Type 3 in 1D to 4D, and 5D and 6D on small intervals (at most `100*eps`). |
| `gpu_nufft_batch` | Batched type 1, 2, and 3. |
| `gpu_nufft_precision`, `gpu_nufft_df64` | Native `f64` and `Df64` accuracy, layouts, and device capabilities. |
| `gpu_clustered_points` | Tightly clustered points, the large-bin sort, and the heavy passes, in 1D to 6D. |
| `gpu_invalid_points` | Coordinates outside the contract, NaN and infinities included, in 1D to 6D. |
| `gpu_device_mismatch` | Buffers and devices from another `wgpu` device. |
| `cpu_nufft`, `cpu_nufft_type3` | The CPU plans against the reference (types 1 and 2 in 1D to 5D, type 3 in 1D to 4D) in all three precisions, clustered and boundary points, bitwise repeatability across thread counts, and prepared point sets against one-shot executions. They run without a GPU. |

`WGPU_NUFFT_RUN_LARGE_GPU_TESTS=1` adds a 256^3 case, which needs
`--features gpu-profiling`. Browser tests are described in
[web/README.md](web/README.md).

### Features and benchmarks

| Feature | Effect |
|---|---|
| `cpu` (default) | The CPU backend, `CpuNufftPlan` and `CpuNufftType3Plan`. |
| `gpu-profiling` | Timestamp-query stage profiling (`NufftPlan::encode_type1_gpu_profiled` and friends). |
| `type1-2d-tile-prototype`, `type1-3d-tile-prototype` | Benchmark-only variants of the type-1 gather. |

The benchmarks in `benches/` list their options with `--help`, for example
`cargo bench --bench nufft_bench -- --help`. With `--reuse-points`,
`nufft_bench` times the execute-only path of a prepared point set, and
`nufft_cpu_bench` times the CPU plans, with and without prepared points.

## Conventions

For integer mode vectors `k`, nonuniform points `x_j`, and complex strengths
`c_j`, type 1 computes

```text
f_k = sum_j c_j exp(isign * i * dot(k, x_j))
```

and type 2 computes

```text
c_j = sum_k f_k exp(isign * i * dot(k, x_j)).
```

For nonuniform source points `x_j` and nonuniform target frequencies `s_k`,
type 3 computes

```text
f_k = sum_j c_j exp(isign * i * dot(s_k, x_j)).
```

- There is no normalization. A nonnegative `isign` selects the positive sign
  and a negative one the negative sign.
- Modes are in centered order by default, with dimension zero stored fastest.
  Type 3 has no integer-mode ordering.
- Type-1 and type-2 coordinates must satisfy `|x| <= 3*pi` and are periodic
  modulo `2*pi`. Type-3 sources and targets must lie inside the intervals
  given when the plan is created.
- GPU plans cannot check device buffers, so a coordinate that breaks the
  contract, NaN or an infinity included, is not reported. It makes the type-1
  or type-3 result that contains it undefined, but never corrupts memory,
  other points' type-2 outputs, or the plan's later executions.
- With the usual Hermitian inner product, type 2 with sign `s` is the adjoint
  of type 1 with sign `-s`. The adjoint of a type-3 plan swaps its source and
  target sets and reverses the sign.

## Repository layout

```text
wgpuNUFFT/
|-- src/, tests/, benches/  the wgpu-nufft crate
|-- wgpuFFT/                pinned wgpu-fft submodule
|-- wgpu-web/               browser package (wasm-bindgen)
|-- python/                 Python binding (PyO3 + NumPy)
`-- web/                    browser test harness
```

The Cargo packages are not published to crates.io (`publish = false`). See
[CHANGELOG.md](CHANGELOG.md) for release notes.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
