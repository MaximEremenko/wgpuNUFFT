# wgpuNUFFT

`wgpu-nufft` is a portable nonuniform FFT library built only on `wgpu-fft`'s
public API. The repository is named `wgpuNUFFT`, the Cargo package is
`wgpu-nufft`, and Rust code imports it as `wgpu_nufft`.

This is the upper-level repository. Its `wgpuFFT/` directory is a pinned Git
submodule rather than an internal copy of the FFT implementation. Planning is
separate from execution and plans are reusable. The Cargo packages are kept
local (`publish = false`) and are consumed from a Git checkout rather than
crates.io.

## Checkout and repository layout

Any fresh Git clone must include Git's `--recurse-submodules` option. If the
repository is already checked out, initialize its pinned FFT dependency with:

```powershell
git submodule update --init --recursive
```

```text
wgpuNUFFT/
|-- src/, tests/, benches/  wgpu-nufft package
|-- wgpuFFT/                pinned wgpu-fft submodule
|-- wgpu-web/               combined browser wrapper
|-- python/                 local PyO3 + NumPy binding
|-- web/                    browser integration harness
`-- nd_prototype/           standalone NUFFT research prototype
```

The outer Cargo workspace intentionally excludes `wgpuFFT/`. This keeps the
submodule independently buildable and prevents parent formatting or lockfile
operations from modifying it. Validate both repository boundaries explicitly:

```powershell
cargo test --locked
cargo test --manifest-path wgpuFFT/Cargo.toml --locked
```

Rust 1.92 or newer is required. GPU suites remain opt-in as documented by the
individual test targets.

## Local dependency and quick start

Add the checkout as a path dependency, with the path resolved relative to the
consumer's `Cargo.toml`:

```toml
[dependencies]
wgpu-nufft = { path = '../wgpuNUFFT' }
```

The direct `f64` transform is a small CPU-only way to verify the dependency
before creating a GPU plan:

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

## Python binding

`python/` builds a native CPython extension directly from the Rust library
with PyO3, NumPy, and maturin. It does not add a separate C ABI and it is not
published. The first supported Python data path is `float32` coordinates with
`complex64` values for reusable type-1, type-2, and type-3 GPU plans. The
pinned development tools below need Python 3.12 or newer (NumPy 2.5.2 has no
older wheels); the built package itself supports Python 3.10 and newer.

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "maturin==1.15.0" "numpy==2.5.2" "pytest==9.1.1"
$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
cargo test -p wgpu-nufft-python --locked
maturin develop --release --manifest-path python\Cargo.toml
python -m pytest python\tests -q
```

```python
import numpy as np
import wgpu_nufft

context = wgpu_nufft.Context()
plan = wgpu_nufft.Type1Plan(context, (16,), eps=1e-5)
points = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float32)
strengths = np.ones(32, dtype=np.complex64)
modes = plan.execute(points, strengths)
```

Plans retain their GPU context and serialize reuse of internal scratch
buffers. Inputs must be C-contiguous NumPy arrays. Multidimensional Fourier
arrays keep the logical Python shape `(*n_modes)`; the binding explicitly
converts that C-order layout to and from the Rust core's dimension-zero-fast
storage. See [python/README.md](python/README.md) for the complete array and
batch contract and the command that builds a local wheel.

## Current scope

The current implementation provides:

- Type-1, type-2, and type-3 definitions in arbitrary
  supported dimension: hand-tuned 1D, 2D, and 3D paths plus a rank-generic GPU
  path that serves every supported rank `d >= 4` for all three transform types
  in F32, native F64, and portable double-float;
- direct `f64` NDFT execution for every transform kind as the correctness oracle
  and tiny-problem fallback;
- `sigma = 2` default exponential-of-semicircle kernel planning;
- host-`f64` centered-grid kernel Fourier coefficients;
- even fine-grid selection through `wgpu-fft`'s public supported-length
  factorization;
- reusable 1D, 2D, and 3D type-2 `f32`, native-`f64`, and portable
  double-float GPU plans with caller-owned point, Fourier-coefficient, and
  output buffers, including many-vector batching;
- reusable 1D, 2D, and 3D type-1 plans in the same three precisions with
  caller-owned point, strength, and Fourier-mode output buffers, including
  many-vector batching;
- reusable 1D, 2D, and 3D type-3 plans in the same three precisions with
  caller-owned source points, strengths, target frequencies, and output
  buffers, also with many-vector batching;
- a `set_points_gpu` / `execute_*` split for type-1 and type-2 plans that
  prepares a point set once for repeated executions (see
  [Reusing a point set](#reusing-a-point-set));
- a local native Python extension with persistent GPU contexts, reusable
  type-1/type-2/type-3 plans, NumPy host-array transfer, and many-vector
  batching in F32/`complex64`.

The GPU type-2 route pre-deconvolves and zero-pads on the GPU, executes a public
`wgpu-fft` C2C plan on the oversampled grid, and gathers the ES interpolation
kernel at each nonuniform point. The caller records execution into its own
command encoder and can consume the caller-owned output buffer without a
readback. Plan-owned fine-grid buffers are reused between ordered executions.
Points are point-major values in the configured precision (scalar in 1D and, for example,
`[x0, y0, z0, x1, y1, z1, ...]` in 3D), while coefficients and outputs are
interleaved complex `(re, im)` pairs; all require `STORAGE` buffer usage.
Call `NufftPlan::required_point_buffer_size_bytes` when sizing a plan's
coordinate buffer; the older transform-specific static helpers retain their 1D
contract.

`NufftConfig::with_batch(ntr)` and `NufftType3Config::with_batch(ntr)` follow
the `ntransf` batching convention: every transform shares one point set, while
complex inputs and outputs are transform-major (`[transform][point or mode]`).
Point binning, scanning, and sorting run once per execution (or once per point
set, see [Reusing a point set](#reusing-a-point-set)) for all vectors together,
and the per-cell spread and direct interpolation kernels reuse each computed
support weight across blocks of up to four vectors. The oversampled-grid C2C plan uses `wgpu-fft`'s native batch
dimension. The ordinary encode methods execute the configured count; the
explicit `*_batch` methods may execute `1..=ntr` vectors on the same plan for
grow/shrink workflows. The embedded FFT remains capacity-sized in that case,
so a smaller active count is a reuse feature rather than a promise of
proportionally lower FFT work.

The 1D Vulkan accuracy matrix enforces relative L2 error at most `4*eps` over
the complete random-plus-adversarial vector and `8*eps` for each isolated
boundary, cluster, duplicate, and seeded-random subset. The 2D and 3D matrices
cover the same tolerance range, signs, mode orders, and adversarial point
classes at a `20*eps` float acceptance ceiling, plus
determinism, non-square/non-cubic grids, scratch reuse, and opposite-sign
adjoint checks.

The GPU type-1 route bins point indices with portable `u32` atomics, scans the
flattened 1D, 2D, or 3D bin counts hierarchically, restores a stable input order
within each bin, and assigns each fine-grid cell to exactly one gathering
invocation. It therefore needs no unavailable WGSL `f32` atomics, and repeated
executions accumulate strengths in deterministic point order. In 2D and 3D,
the ES spreading/interpolation weights and deconvolution amplitudes are tensor
products, with dimension zero stored fastest. Shared-memory tiles accelerate
the default multidimensional spreading path while retaining a global fallback
for device limits that cannot support the tile. The gathered grid passes
through a public `wgpu-fft` ND C2C plan before deconvolution and truncation into
the caller-owned mode buffer. The type-1/type-2 routes reuse plan-owned fine-grid
scratch. Type 1 also reuses fixed bin-count, cursor, and offset buffers and grows
its sorted-point-index buffer only when a larger point set requires it.
Executions on a given plan must remain in queue order.

F32 plans whose fine grid is large enough (at `eps = 1e-6`, at least 18 cells
per axis in 1D and 2D, and 32/32/22 cells in 3D) use coarse-bin paths instead.
The points are grouped into small bins once (integer-atomic counts, a prefix
scan, and, for type 1, a per-bin sort back into input order), and every point's
support start and df64 offset are prepared once, so the kernels evaluate every
weight as `(j + hi) + lo` without refolding coordinates:

- Type-1 spreading stays output-stationary. In 1D each invocation owns four
  consecutive fine-grid cells and in 2D a `4x4` tile, and walks only the few
  bins that can reach them, in a fixed order. In 3D each workgroup owns a
  `16x16x8` block whose rows stay in registers while the block's nearby points
  stream through workgroup memory. Every fine-grid cell is written exactly
  once, without atomics, so results remain bitwise repeatable.
- Type-2 interpolation evaluates the points in bin order for cache locality
  (in 2D and 3D, eight invocations share one point and read each fine-grid row
  as one run). Every output is still one fixed-order sum.
- The small, densely populated outer grids of 1D and 2D type-3 plans use a
  dense spreader that splits the sources by index into fixed groups, sums each
  group into a partial grid, and adds the partial grids in group order. The 3D
  type-3 outer grid uses the 3D block spreader.

F64/Df64 plans, smaller grids, and devices without the required workgroup
limits keep the per-cell gather and direct interpolation.

### Known limitation: clustered points

Type-1 plans cost roughly `n log n` in the largest number of points that share
one coarse bin. Two steps are serial for such a bin: restoring input order,
where one invocation heap-sorts every bin above 64 points, and the gather,
where each tile's invocation walks every point in reach. Uniformly spread
points put only a handful in each bin, but tightly clustered inputs, such as
displacement vectors near zero, do not. 65,536 points inside a
single bin take about 24 ms in 1D, 150 ms in 2D and 120 ms in 3D per
execution.

Around a million points in one bin, a single dispatch can run long enough for
the Windows TDR watchdog to reset the GPU driver, which ends every process
using the GPU. Until large bins are sorted and gathered in parallel, keep
heavily clustered type-1 inputs well below that size.

### Reusing a point set

`NufftPlan::set_points_gpu` records the point-dependent work (for the F32
coarse-bin paths of 1D-3D type-1 and type-2 plans: binning, sorting, and
per-point preparation) once, and
`execute_type1_gpu[_batch]` or `execute_type2_gpu[_batch]` then transform new
strengths or coefficients at those points, like a one-time point setup followed by
repeated `execute` calls. The plan keeps a reference to the point buffer; its
contents must not change while executions use it, and executions must be
submitted after the commands recorded by `set_points_gpu`. The existing
`encode_*` methods are equivalent to `set_points_gpu` followed by the matching
`execute_*` call. `cargo bench -p wgpu-nufft --bench nufft_bench --
--reuse-points` times this execute-only path.

The GPU type-3 route follows the standard rescaling composition: it rescales and
pre-phases nonuniform sources, spreads them to an outer uniform grid, evaluates
that grid through an inner type-2 plan, then applies the continuous ES-kernel
correction and post-phase at rescaled nonuniform target frequencies. A reusable
GPU plan cannot inspect arbitrary device buffers during construction, so
`NufftType3Config` requires conservative source and target intervals. Every
coordinate supplied to later GPU executions must remain finite and inside those
planned intervals. The direct CPU route validates this contract explicitly;
GPU execution treats it as a caller guarantee. Absurd space-bandwidth products
and rescalings that cannot fit the GPU index, buffer, or portable double-float
phase range return structured planning errors. Use the dimension-aware
`NufftType3Plan::required_*_buffer_size_bytes` methods to size each caller-owned
source, target, strength, and output buffer. Type-3 plans likewise reuse
grow-only internal scratch and require ordered execution on a given plan.

## Precision

`NufftConfig::with_precision` and `NufftType3Config::with_precision` select the
coordinate, complex-value, kernel, and fine-grid FFT precision together. `F32`
is the default and preserves the original shader path. `F64` uses scalar `f64`
coordinates and `vec2<f64>` complex values and requires a device created with
`wgpu::Features::SHADER_F64`; this is currently a Vulkan capability in wgpu.
Plans return a structured `PrecisionUnsupported` error before shader creation
when that feature is absent. `Df64` represents every scalar as an unevaluated
`hi + lo` pair of `f32` words and every complex value as
`(re_hi, re_lo, im_hi, im_lo)`. It requires no optional device feature and is
the portable high-precision route for DX12, Metal, and browser/WebGPU devices
that do not expose native shader `f64`.

The ES kernel cannot call `exp` in native-f64 SPIR-V, and df64 deliberately
contains no shader transcendental operations. Both high-precision routes fit
piecewise-Horner tables on the host in `f64` and use the same
polynomial for interpolation/spreading and Fourier deconvolution. Native-f64
tables store `f64` coefficients; df64 tables split each coefficient into its
high and low `f32` words. Type-3 phase factors use a transcendental-free,
range-reduced polynomial sine/cosine. Native f64 conservatively rejects planned
phase magnitudes above `1e6`; portable df64 uses the tighter, accuracy-driven
bound of `1024` because its two-f32 range reduction loses absolute precision as
the unreduced phase grows.

Df64 retains roughly 44-48 significant bits, but its exponent range remains
that of `f32` (approximately `1e-38` through `1e38`). Its split-based Dekker
products avoid relying on fused multiply-add contraction; exact-word arithmetic
canaries in `wgpu-fft` define the backend support contract. Vulkan and DX12 are
tested. Metal remains the riskiest untested backend because its
shader compiler enables fast-math transformations by default. Plan-aware
byte-size helpers account for 8-byte coordinates and 16-byte complex values in
both high-precision formats; legacy static helpers retain their documented
one-dimensional f32 ABI.

Both the crate and the `wgpu-web` JavaScript surface run on browser WebGPU.
Chrome/Tint testing at exact WebGPU defaults covers batched type 1/2/3 in one
through three dimensions for F32 and Df64. The browser compiler must first pass
all 96 exact df64 arithmetic words; otherwise only Df64 is disabled. Native F64
remains structurally unsupported in browsers because `SHADER_F64` is absent.
The browser wrapper keeps plans and caller buffers GPU-resident and exposes
explicit upload, awaited execution, and download operations.

## Mathematical conventions

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

There is no normalization. Nonnegative `isign` selects the positive sign;
negative values select the negative sign. Centered mode order is the default,
and dimension zero is stored fastest. Type 3 has no integer-mode ordering.
Type-1/type-2 coordinates follow the documented `|x| <= 3*pi` contract
and are periodic modulo `2*pi`; type-3 source and target domains are instead
the explicit intervals supplied at planning. With the conventional Hermitian
inner product, type-2 with sign `s` is adjoint to type-1 with sign `-s`; a
type-3 plan's adjoint swaps source and target sets and reverses the sign.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
