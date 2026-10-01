# wgpu-nufft-c

A C interface to `wgpu-nufft`, declared by [`include/wgpu_nufft.h`](include/wgpu_nufft.h):
nonuniform fast Fourier transforms of types 1, 2, and 3 on the GPU (Vulkan,
DX12, Metal) or on the CPU, from host arrays. The Fortran module in
[`../fortran`](../fortran) and the MATLAB interface in [`../matlab`](../matlab)
build on it.

## Building

```sh
cargo build --release -p wgpu-nufft-c
```

This writes a shared library (`wgpu_nufft_c.dll` with the import library
`wgpu_nufft_c.dll.lib`, `libwgpu_nufft_c.so`, or `libwgpu_nufft_c.dylib`)
and a static library (`wgpu_nufft_c.lib` or `libwgpu_nufft_c.a`) to
`target/release`. A static link also needs the system libraries that

```sh
cargo rustc --release -p wgpu-nufft-c --crate-type staticlib -- --print native-static-libs
```

lists for the platform.

## Use

```c
#include <stdio.h>
#include "wgpu_nufft.h"

/* One call: 2D type 1 of M points into 32-by-32 modes. */
double f[2 * 32 * 32];
if (wgpu_nufft2d1(M, x, y, c, +1, 1e-9, 32, 32, f, NULL) != WGPU_NUFFT_SUCCESS)
    fprintf(stderr, "%s\n", wgpu_nufft_last_error());

/* A plan: points set once, batches of 4 transforms run many times. */
int64_t n_modes[2] = {32, 32};
wgpu_nufft_plan plan;
wgpu_nufft_makeplan(1, 2, n_modes, +1, 4, 1e-9, &plan, NULL);
wgpu_nufft_setpts(plan, M, x, y, NULL, 0, NULL, NULL, NULL);
wgpu_nufft_execute(plan, c4, f4); /* c4: M by 4, f4: 32 by 32 by 4 */
wgpu_nufft_destroy(plan);
```

- Complex arrays are interleaved `(re, im)` pairs. Modes are stored
  dimension zero fastest (column-major), in centered order: index `k` along
  an axis of `n` modes holds frequency `k - floor(n/2)`.
- `wgpu_nufft_*` functions take `double` arrays, `wgpu_nufftf_*` functions
  `float` arrays. Double arrays run in native `f64` on GPUs that support it
  and in `Df64` (about 44-48 bits) on others; `wgpu_nufft_opts` selects the
  backend, the arithmetic, the mode order, CPU threads, and the upsampling
  factor.
- GPU plans share one device per process, created on first use; the
  `WGPU_BACKEND` environment variable (`vulkan`, `dx12`, `metal`) picks a
  backend. Without a usable GPU, plans run on the CPU unless the options ask
  for the GPU. `wgpu_nufft_shutdown()` releases the device and the plans the
  one-call functions keep.
- A plan may be used from one thread at a time.

## Tests

```sh
cargo test -p wgpu-nufft-c                       # CPU
WGPU_FFT_RUN_GPU_TESTS=1 cargo test -p wgpu-nufft-c  # CPU and GPU
```
