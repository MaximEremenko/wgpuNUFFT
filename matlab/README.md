# MATLAB interface

Nonuniform fast Fourier transforms of types 1, 2, and 3 on the GPU or the
CPU, from MATLAB R2018a or later, over the C interface of [`../capi`](../capi).

## Building

With Rust (cargo) installed and a C compiler set up for MEX
(`mex -setup C`), run in MATLAB:

```matlab
cd wgpuNUFFT/matlab
build_wgpu_nufft      % compiles private/wgpu_nufft_mex
test_wgpunufft        % checks every interface against direct sums
addpath(pwd)
```

The MEX file links the C library statically, so it needs no other file.

## Use

```matlab
f = wgpunufft2d1(x, y, c, +1, 1e-9, ms, mt);   % type 1: ms-by-mt modes
c = wgpunufft2d2(x, y, +1, 1e-9, f);           % type 2: values at the points
f = wgpunufft2d3(x, y, c, +1, 1e-9, s, t);     % type 3: values at s, t

plan = wgpunufft_plan(1, [ms mt], +1, ntrans, 1e-9);
plan.setpts(x, y);
F = plan.execute(C);   % C: M-by-ntrans, F: ms-by-mt-by-ntrans
```

- Modes are in centered order, frequency `-floor(n/2)` first along each
  axis; `opts.modeord = 1` selects FFT order.
- Double arrays run in native `f64` on GPUs that support it and in `Df64`
  (about 44-48 bits) on others; single arrays run in `f32`.
- `opts.backend` is `'auto'` (the GPU when there is one), `'gpu'`, or
  `'cpu'`; see `help wgpunufft_plan` for every option.
- The one-call functions keep their plans, so repeated calls with the same
  sizes are fast; `wgpunufft_shutdown` releases them and the GPU device.
- `wgpunufft_info` names the GPU in use.
- GPU plans run on the adapter wgpu picks unless `opts.adapter_name`,
  `opts.adapter_pci_bus_id`, or `opts.adapter_index` select one;
  `wgpunufft_adapters` lists them. A selection that matches no adapter is an
  error, never a fallback to the CPU;
  [`../capi/README.md`](../capi/README.md#selecting-the-gpu) has the rules.

```matlab
adapters = wgpunufft_adapters();   % name, backend, device_type, pci_bus_id, is_default
f = wgpunufft2d1(x, y, c, +1, 1e-9, ms, mt, struct('adapter_name', 'NVIDIA GeForce RTX 4090'));
```
