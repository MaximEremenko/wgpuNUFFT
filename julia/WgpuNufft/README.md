# WgpuNufft.jl

Nonuniform fast Fourier transforms of types 1, 2, and 3 on the GPU (Vulkan,
DX12, Metal) or the CPU, from Julia, over the C interface of
[`capi/`](../../capi).

## Setup

The package loads the shared C library from `ENV["WGPU_NUFFT_C_LIBRARY"]`,
or from `target/release` of the wgpuNUFFT clone it lives in:

```sh
cargo build --release -p wgpu-nufft-c
julia --project=julia/WgpuNufft -e 'using Pkg; Pkg.test()'
```

In another project, `Pkg.develop(path = "path/to/wgpuNUFFT/julia/WgpuNufft")`.

## Use

```julia
using WgpuNufft

f = nufft2d1(x, y, c, 1, 1e-9, ms, mt)   # type 1: ms × mt modes
c = nufft2d2(x, y, 1, 1e-9, f)           # type 2: values at the points
f = nufft2d3(x, y, c, 1, 1e-9, s, t)     # type 3: values at s, t

plan = Plan(1, (ms, mt), 1, ntrans, 1e-9)
setpts!(plan, x, y)
F = execute(plan, C)    # C: M × ntrans, F: ms × mt × ntrans
```

- Modes are in centered order, frequency `-floor(n/2)` first along each
  axis; `modeord = 1` selects FFT order. Julia arrays are column-major, the
  library's own order, so modes pass through unchanged.
- `Float64` arrays run in native f64 on GPUs that support it and in Df64
  (about 44-48 bits) on others; `Float32` arrays run in f32.
- Keywords of every transform and of `Plan`: `backend` (`:auto`, `:gpu`,
  `:cpu`), `precision` (`:auto`, `:f64`, `:df64`, `:f32`), `modeord`,
  `threads`, `sigma`, and the GPU adapter: `adapter_name`,
  `adapter_pci_bus_id`, `adapter_index`.
- GPU plans run on the adapter wgpu picks unless the keywords select one, by
  name (such as the name CUDA reports for a device), by PCI address, and by
  index among the matching adapters; `WgpuNufft.adapters()` lists them. A
  selection that matches no adapter throws, never falls back to the CPU;
  [`capi/README.md`](../../capi/README.md#selecting-the-gpu) has the rules.

```julia
WgpuNufft.adapters()   # name, backend, device_type, pci_bus_id, is_default
f = nufft2d1(x, y, c, 1, 1e-9, ms, mt; adapter_name = "NVIDIA GeForce RTX 4090")
```
- Plans of more than three dimensions take a `dim × M` matrix:
  `setpts!(plan, X)`.
- The one-call functions keep their plans between calls with the same sizes;
  `WgpuNufft.shutdown()` releases them and the GPU device.
- Errors from the library are thrown as `WgpuNufftError`.
