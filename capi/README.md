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

## Selecting the GPU

Unless the options select an adapter, wgpu picks it, preferring a discrete
GPU. On a machine with several GPUs, three fields of `wgpu_nufft_opts`
select one, for instance the GPU a CUDA program already uses:

```c
/* The GPU of CUDA device `device`: its name, and its PCI address, which tells
 * identical cards apart. */
struct cudaDeviceProp properties;
cudaGetDeviceProperties(&properties, device);
wgpu_nufft_opts opts;
wgpu_nufft_default_opts(&opts);
snprintf(opts.adapter_name, sizeof opts.adapter_name, "%s", properties.name);
cudaDeviceGetPCIBusId(opts.adapter_pci_bus_id, sizeof opts.adapter_pci_bus_id, device);
```

- `adapter_name` matches an adapter's whole name, regardless of case and of
  surrounding spaces. NVIDIA's Vulkan and DX12 drivers report the name CUDA
  does.
- `adapter_pci_bus_id` matches the PCI address, `[domain:]bus:device[.function]`
  in hexadecimal, as `cudaDeviceGetPCIBusId()` and nvidia-smi write it.
  Vulkan reports every adapter's address; DX12 gives identical cards the
  address of the first, and Metal reports none, so an address selects
  reliably on Vulkan, the backend wgpu tries first.
- `adapter_index` = k takes the k-th adapter, in the order of
  `wgpu_nufft_list_adapters()`, among those the name and the address match,
  or among all of them; 0 leaves it unset, so a zero-filled `wgpu_nufft_opts`
  still means the defaults.
- Without an index, the matches of the first backend that lists any count:
  Windows lists a GPU under Vulkan and under DX12, and Vulkan comes first.
  They must be one adapter; identical cards need the address or an index.
- A selection that matches no adapter fails with
  `WGPU_NUFFT_ERROR_GPU_UNAVAILABLE`, whose message lists the adapters, also
  with the automatic backend: plans never fall back to the CPU for want of
  the selected adapter. CPU plans ignore the selection.
- The device is created on the adapter of the first GPU plan. Plans without
  a selection share it, and a plan that selects another adapter fails with
  `WGPU_NUFFT_ERROR_INVALID_ARGUMENT` until `wgpu_nufft_shutdown()`.
  `wgpu_nufft_gpu_name()` creates the device on the default adapter when
  there is none, so call it after the first plan.

`wgpu_nufft_list_adapters()` lists the adapters with their backend, device
type, PCI address, and which one is the default:

```c
int32_t count = 0;
wgpu_nufft_list_adapters(NULL, 0, &count);
wgpu_nufft_adapter *adapters = calloc(count, sizeof *adapters);
wgpu_nufft_list_adapters(adapters, count, &count);
for (int32_t i = 0; i < count; i++)
    printf("%d. %s (%s, %s) %s%s\n", i + 1, adapters[i].name, adapters[i].backend,
           adapters[i].device_type, adapters[i].pci_bus_id,
           adapters[i].is_default ? " default" : "");
```

On a workstation with one NVIDIA card and an Intel iGPU under Windows, that
prints:

```text
1. NVIDIA RTX 5000 Ada Generation (Vulkan, DiscreteGpu) 0000:01:00.0 default
2. Intel(R) UHD Graphics 770 (Vulkan, IntegratedGpu)
3. NVIDIA RTX 5000 Ada Generation (Dx12, DiscreteGpu) 0000:01:00.0
4. Intel(R) UHD Graphics 770 (Dx12, IntegratedGpu) 0000:00:02.0
5. Microsoft Basic Render Driver (Dx12, Cpu)
```

## Tests

```sh
cargo test -p wgpu-nufft-c                       # CPU
WGPU_FFT_RUN_GPU_TESTS=1 cargo test -p wgpu-nufft-c  # CPU and GPU
```

The GPU tests include adapter selection by name, index, and address, and a
selection of another adapter while the device exists.
