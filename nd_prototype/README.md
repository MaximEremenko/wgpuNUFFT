# ND NUFFT research prototype

A standalone research prototype of rank-generic N-dimensional NUFFTs in which
every dimension chooses how to pay its translation rank `r(eps)`. It is not
part of the `wgpu-fft` or `wgpu-nufft` crates. It is kept as a reference
oracle and as the record of the design behind `wgpu-nufft`'s 4D and 5D GPU
paths.

## Placements

| Placement | Mechanism | Fine grid | Cost per dimension |
|---|---|---|---|
| `SPREAD_S2` | ES-kernel spreading, sigma = 2 | 2N | w(eps) grid touches |
| `SPREAD_S125` | ES-kernel spreading, sigma = 1.25 | 1.25N | wider w, 1.6x less memory |
| `RT` | sigma = 1 Chebyshev-interpolated perturbation factor (Ruiz-Antolin/Townsend style) | N | K(eps) scaled FFT passes |

RT multiplicity multiplies across RT dimensions (K per dimension), so RT is
meant for one or two memory-critical dimensions, with spreading elsewhere.

## Files

- `nd_nufft.py`: the reference implementation (types 1 and 2, any d, both
  signs, f64). Kernel parameters follow the standard ES rules; RT factors come
  from barycentric Chebyshev interpolation in the sub-cell offset, with
  adaptive K.
- `validate.py`: an accuracy matrix against a direct NDFT plus the adjoint
  identity, for d = 1 to 5, all placements and mixes, both signs, and
  clustered points.
- `cost.py`: analytic work counters checked against instrumented execution,
  and the placement policy.
- `gpu_experiment/`: a standalone Rust crate, outside the workspace, that
  generates rank-generic WGSL spreading and interpolation kernels;
  `check.py` and `check2.py` validate the results it exports.

## Accuracy

`python validate.py` passes all 22 cases. Relative L2 errors track eps for
every placement mix; for example, 4D with three sigma = 1.25 dimensions and
one RT dimension at eps = 1e-6 gives 4.3e-6 for type 1 and 4.2e-6 for type 2.
Adjoint residuals are at machine precision (at most 7e-12), so the type-1 and
type-2 compositions are consistent, and spreading and RT placements compose
without measurable error inflation in this matrix.

## Cost model

With `python cost.py`, the analytic update counters match instrumented
execution exactly for every placement mix. Two consequences:

1. RT multiplicity K multiplies the spreading work
   (updates per point = prod(w_spread) * prod(K_rt)), so the work per point
   does not depend on the placement. Placement only decides where memory and
   FFT passes go.
2. Interpolation-based K is about 10 at eps = 1e-6, against an SVD floor of 7,
   which leaves room for SVD-derived factors.

The placement policy (`choose_placements`) takes sigma = 2 for d <= 3 and
sigma = 1.25 for d >= 4, and RT only under hard memory limits. RT is what
fits 64^4 into a 256 MiB browser budget (250 MiB); 96^4 and larger grids in a
browser need segmented execution such as `wgpu-fft`'s large routes.

## GPU spreading in any dimension

`gpu_experiment/` generates the gather-form spreading kernel in WGSL for any
dimension; d appears only in unrolled generated code. Each workgroup owns an
output tile with one register accumulator per cell, points arrive in
shared-memory batches, ES weights are per dimension with periodic wrap, and
there are no atomics. Host-side binning duplicates points into every tile
they touch, a shortcut that costs 21x duplication in 3D and up to 200x on
small 4D grids; production code reads neighbouring bins instead.

Vulkan results (`cargo run --release`, then `python check.py`):

| Case | Grid | Median | Throughput |
|---|---|---|---|
| d=3 256^3 s1.25, M=1e6, w=10 | 0.2 GiB | 8.99 ms | 1.11e11 upd/s |
| d=4 48^4 s1.25, M=5e5, w=10 | 0.1 GiB | 68.6 ms | 7.29e10 upd/s |
| d=4 64^4 s1.25, M=5e5, w=10 | 0.3 GiB | 73.2 ms | 6.83e10 upd/s |
| d=5 24^5 s1.25 eps=1e-3, M=2e5, w=5 | 0.2 GiB | 35.0 ms | 1.79e10 upd/s |

The exported 4D case has a relative L2 error of 1.11e-6 against an f64 NumPy
reference, the level of f32 accumulation error. Its throughput is in the
class of the tuned 2D kernel of the time (about 6.7e10 updates/s), so
rank-generic code generation cost essentially nothing in these measurements.

## End-to-end 4D on the GPU

Full chains ran on the GPU with `wgpu-fft` as the ND FFT through its public
API (a rank-4 plan over the 80^4 fine grid): type 1 as GPU spreading followed
by the forward FFT, and type 2 as host deconvolution and padding, the FFT, and
a rank-generic GPU interpolation kernel. At 64^4 with sigma = 1.25, type 1
took 79.5 ms and type 2 86.7 ms; spreading and interpolation dominate, as the
cost model predicts. `wgpu-fft`'s forward direction corresponds to
isign = -1.

On the exported 12^4 case (`python check2.py`), interior modes (|k| <= N/4)
reach 3.56e-7, clean f32, while the full mode box reaches 3.76e-4 (type 1)
and 2.07e-4 (type 2). The gap comes from deconvolution conditioning at
sigma = 1.25: 1/phi_hat amplifies f32 FFT rounding by about 50x per dimension
at the edge of the mode box, compounding toward the corners. It is invisible
in f64 but material in f32. Possible remedies are an f64 or df64 fine grid for
tight eps at sigma = 1.25, trimming the mode-box margin, or sigma = 2 where
memory allows.

## Outcome

The prototype led to the hybrid design in `wgpu-nufft`: generated
rank-generic kernels for 4D and 5D (`MAX_GPU_NUFFT_DIMENSIONS`), with the
tuned 1D, 2D, and 3D paths kept for lower ranks. The production path reuses
the crate's cell binning, scan, and sort instead of host duplication,
supports `f32`, native `f64`, and `Df64`, and uses sigma = 2 unless
`NufftConfig::with_sigma` selects another oversampling factor. The
measurements above describe the prototype, not the production code.
