# wgpu-nufft for Python

Python binding of [wgpu-nufft](https://github.com/MaximEremenko/wgpuNUFFT):
portable GPU and CPU nonuniform FFTs for NumPy arrays. It is a native
CPython extension built with PyO3 and maturin, with no separate C ABI.

- Reusable type-1, type-2, and type-3 plans in one to eight dimensions, on
  the GPU and on the CPU.
- GPU plans take `float32` coordinates and `complex64` values. CPU plans
  take either those or `float64` coordinates and `complex128` values.
- A `Context` holds one GPU device, and GPU plans keep their context,
  pipelines, and scratch buffers between calls. Each call uploads its NumPy
  inputs and returns a new NumPy array.
- CPU plans need no GPU. They release the GIL while they work, give
  bitwise-identical results for any thread count, and prepare a point set
  once for repeated transforms.

## Example

```python
import numpy as np
import wgpu_nufft

context = wgpu_nufft.Context()
print(context.adapter_name, context.backend)
rng = np.random.default_rng(0)

# Type 1: nonuniform points to Fourier modes.
type1 = wgpu_nufft.Type1Plan(context, (16,), eps=1e-5)
points = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float32)
strengths = np.ones(32, dtype=np.complex64)
modes = type1.execute(points, strengths)  # shape (16,)

# Type 2: Fourier modes to nonuniform points, here in 2D.
type2 = wgpu_nufft.Type2Plan(context, (32, 24), eps=1e-5, isign=-1)
points_2d = rng.uniform(-np.pi, np.pi, (1000, 2)).astype(np.float32)
coefficients = np.ones((32, 24), dtype=np.complex64)
values = type2.execute(points_2d, coefficients)  # shape (1000,)

# Type 3: nonuniform points to nonuniform frequencies within fixed bounds.
type3 = wgpu_nufft.Type3Plan(
    context,
    source_bounds=np.array([[-1.0, 1.0]]),
    target_bounds=np.array([[-50.0, 50.0]]),
    eps=1e-5,
)
sources = rng.uniform(-1.0, 1.0, 500).astype(np.float32)
targets = rng.uniform(-50.0, 50.0, 200).astype(np.float32)
spectrum = type3.execute(sources, np.ones(500, np.complex64), targets)  # shape (200,)
```

The CPU plans work the same way without a `Context`, in `complex128` unless
`dtype` says otherwise. `prepare_points` checks and sorts a point set once;
`execute` then takes the prepared set in place of the point array, with
results bitwise identical to passing the array:

```python
import numpy as np
import wgpu_nufft

rng = np.random.default_rng(0)
points = rng.uniform(-np.pi, np.pi, (100_000, 2))
forward = wgpu_nufft.CpuType2Plan((128, 128), eps=1e-9, isign=-1)
adjoint = wgpu_nufft.CpuType1Plan((128, 128), eps=1e-9, isign=1)
prepared = forward.prepare_points(points)  # serves both plans

image = np.zeros((128, 128), np.complex128)
image[64, 64] = 1.0
for _ in range(10):
    image = adjoint.execute(prepared, forward.execute(prepared, image))
```

## API

| Name | Description |
|---|---|
| `Context(power_preference="high-performance", force_fallback=False)` | Opens a GPU device. `adapter_name`, `backend`, `device_type`, `driver`, and `driver_info` describe it. |
| `Type1Plan(context, n_modes, eps=1e-6, isign=1, mode_order="centered", batch=1)` | `execute(points, strengths)` returns the Fourier modes. |
| `Type2Plan(context, n_modes, eps=1e-6, isign=1, mode_order="centered", batch=1)` | `execute(points, coefficients)` returns the values at the points. |
| `Type3Plan(context, source_bounds, target_bounds, eps=1e-6, isign=1, batch=1)` | `execute(source_points, strengths, target_points)` returns the values at the target frequencies. |
| `CpuType1Plan(n_modes, eps=1e-6, isign=1, mode_order="centered", batch=1, dtype=None, threads=None)` | `execute(points, strengths)` returns the Fourier modes; `points` may be a `CpuPoints` from `prepare_points(points)`. |
| `CpuType2Plan(n_modes, eps=1e-6, isign=1, mode_order="centered", batch=1, dtype=None, threads=None)` | `execute(points, coefficients)` returns the values at the points; `points` may be a `CpuPoints`. |
| `CpuType3Plan(source_bounds, target_bounds, eps=1e-6, isign=1, batch=1, dtype=None, threads=None)` | `execute(source_points, strengths, target_points)` returns the values at the target frequencies. `prepare_points(source_points, target_points)` returns a `CpuType3Points`, then `execute(prepared, strengths)`. |
| `WgpuNufftError` | Raised when the GPU fails or cannot run a plan; a subclass of `RuntimeError`. |

`isign` must be `1` or `-1`, and `mode_order` is `"centered"` or `"fft"`.
A CPU plan's `dtype` is `complex64` or `complex128` (the default), in any
form `numpy.dtype` accepts, and `threads` defaults to every core. A
`CpuPoints` set serves every CPU type-1 and type-2 plan with the same modes,
`eps`, and `dtype`; a `CpuType3Points` set serves every CPU type-3 plan with
the same bounds, `eps`, `isign`, and `dtype`. Invalid arguments, shapes, and
coordinates raise `ValueError`, and arrays of the wrong dtype `TypeError`.
Plans can be shared between threads. Calls on one GPU plan run one at a
time; calls on one CPU plan run concurrently.

## Arrays

- GPU points are C-contiguous `float32` arrays shaped `(M, d)`; a 1D plan
  also accepts `(M,)`. CPU points have the same shapes and the plan's real
  dtype, `float32` or `float64`, in any memory order. Type-1 and type-2
  coordinates must be finite and lie in `[-3*pi, 3*pi]`, which both check.
- Values are `complex64` arrays (or `complex128` for CPU plans) shaped
  `(M,)` for one transform or `(B, M)` for a batch of `B` transforms. GPU
  plans need them C-contiguous.
- Type-1 output and type-2 input have the shape `(*n_modes)` for one
  transform and `(B, *n_modes)` for a batch.
- Type-3 bounds are `float64` arrays shaped `(d, 2)` with one
  `[lower, upper]` row per dimension. Every source and target coordinate must
  be finite and lie inside them, which the binding checks too.

A plan created with `batch=B` executes any batch of 1 to `B` transforms. The
binding converts between NumPy's C order and the dimension-zero-fast storage
of the Rust core, so multidimensional arrays keep their natural NumPy layout.
CPU plans read C-contiguous inputs in place, so do not modify an input array
from another thread while a call on it runs.

## Building

From the repository root, in PowerShell. The pinned development tools need
Python 3.12 or newer (NumPy 2.5.2 has no older wheels); the built package
supports Python 3.10 and newer.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "maturin==1.15.0" "numpy==2.5.2" "pytest==9.1.1"
$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
cargo test -p wgpu-nufft-python --locked
maturin develop --release --manifest-path python\Cargo.toml
python -m pytest python\tests -q
```

`python\tests\test_cpu.py` needs no GPU; the other tests open one.

To build a wheel:

```powershell
maturin build --release --manifest-path python\Cargo.toml `
  --interpreter .\.venv\Scripts\python.exe
```

The package is not published to PyPI. A wheel uses CPython's
version-specific extension ABI, so build one for each Python minor version,
operating system, and CPU architecture you need.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
