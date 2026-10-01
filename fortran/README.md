# Fortran interface

[`wgpu_nufft.f90`](wgpu_nufft.f90) is a Fortran 2008 module over the C
interface of [`../capi`](../capi): nonuniform fast Fourier transforms of types
1, 2, and 3 on the GPU or the CPU, on ordinary Fortran arrays.

```fortran
use wgpu_nufft
complex(c_double_complex) :: f(ms, mt)
integer :: ier
call wgpu_nufft2d1(M, x, y, c, +1, 1d-9, ms, mt, f, ier)
if (ier /= WGPU_NUFFT_SUCCESS) print *, wgpu_nufft_error_message()
```

- Sizes (`M`, `N`, `ms`, `mt`, `mu`, `n_modes`) are `integer(8)`
  (`integer(c_int64_t)`); `ier` is the status, 0 on success.
- `f(ms, mt, mu)` stores frequency `-floor(n/2)` at index 1 along each axis
  (centered order); batches add a last dimension, `c(M, ntrans)` and
  `f(ms, mt, ntrans)`.
- Routines named `wgpu_nufft*` take double-precision arrays and
  `wgpu_nufftf*` single-precision ones. Double arrays run in native `f64` on
  GPUs that support it and in `Df64` on others.
- `type(wgpu_nufft_opts)` selects the backend (`WGPU_NUFFT_BACKEND_GPU`,
  `_CPU`, or automatic), the arithmetic, the mode order, CPU threads, and the
  upsampling factor; pass it as the last, optional argument.
- A plan sets its points once and runs batches:

```fortran
type(wgpu_nufft_plan) :: plan
call wgpu_nufft_makeplan(1, 2, [ms, mt], +1, ntrans, 1d-9, plan, ier)
call wgpu_nufft_setpts(plan, M, x, y, ier=ier)
call wgpu_nufft_execute(plan, c, f, ier)  ! c(M, ntrans), f(ms, mt, ntrans)
call wgpu_nufft_destroy(plan)
```

Plans of more than three dimensions take their points as an array
`points(dim, M)` through `wgpu_nufft_setpts_nd`.

GPU plans run on the adapter wgpu picks unless the options select one, by
name (such as the name CUDA reports for a device), by PCI address, and by
index among the matching adapters; `wgpu_nufft_adapters` lists them. A
selection that matches no adapter is an error, never a fallback to the CPU;
[`../capi/README.md`](../capi/README.md#selecting-the-gpu) has the rules.

```fortran
type(wgpu_nufft_adapter), allocatable :: adapters(:)
type(wgpu_nufft_opts) :: opts
call wgpu_nufft_adapters(adapters, ier)   ! name, backend, device_type, pci_bus_id, is_default
call wgpu_nufft_set_adapter(opts, name="NVIDIA GeForce RTX 4090")
call wgpu_nufft2d1(M, x, y, c, +1, 1d-9, ms, mt, f, ier, opts)
```

## Building and testing

```sh
python fortran/build.py --compiler gfortran --run cpu   # or --run gpu
```

builds the C library with cargo, compiles the module and
[`tests/test_wgpu_nufft.f90`](tests/test_wgpu_nufft.f90) into `fortran/build/`,
links them against the shared library and runs the test. On Windows, Intel
ifx works from a oneAPI environment (`--compiler ifx`). In your own build,
compile `wgpu_nufft.f90` with your sources and link the shared or static
library from `target/release`.
