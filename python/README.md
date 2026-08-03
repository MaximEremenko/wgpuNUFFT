# wgpu-nufft for Python

This local Python package exposes the Rust `wgpu-nufft` implementation as a
native CPython extension. It uses only PyO3, NumPy, and maturin at the binding
boundary. A separate C ABI is not required.

The initial interface uses `float32` coordinates and `complex64` values. A
`Context` owns one persistent GPU device, and reusable type-1, type-2, and
type-3 plans retain that context. Each call currently uploads NumPy inputs and
downloads its result; plan and GPU pipeline setup are reused.

## Local build

From the repository root in PowerShell. The pinned development tools need
Python 3.12 or newer (NumPy 2.5.2 has no older wheels); the built package
itself supports Python 3.10 and newer.

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

Build a local wheel without publishing it:

```powershell
maturin build --release --manifest-path python\Cargo.toml `
  --interpreter .\.venv\Scripts\python.exe
```

## Array contract

- Points are C-contiguous `float32` arrays shaped `(M, d)`. For a 1D plan,
  `(M,)` is also accepted.
- Values are C-contiguous `complex64` arrays. A single transform omits the
  batch axis; batched arrays use `(B, M)`.
- Type-1 output and type-2 input use logical NumPy shape `(*n_modes,)` for one
  transform and `(B, *n_modes)` for a batch.
- Type-3 plans receive C-contiguous `float64` source and target bounds shaped
  `(d, 2)`.
- `isign` must be `1` or `-1`; `mode_order` is `"centered"` or `"fft"`.

The binding explicitly converts between NumPy C order and the Rust core's
dimension-zero-fast mode storage. This matters for multidimensional transforms.

## Example

```python
import numpy as np
import wgpu_nufft

ctx = wgpu_nufft.Context()
plan = wgpu_nufft.Type1Plan(ctx, (16,), eps=1e-5)

points = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float32)
strengths = np.ones(32, dtype=np.complex64)
modes = plan.execute(points, strengths)
print(ctx.adapter_name, modes.shape)
```

No package in this repository is configured for crates.io publication. The
wheel is a local binary artifact and embeds the locally built Rust libraries.
This initial wheel uses CPython's normal version-specific extension ABI, so
build one wheel for each Python minor version, operating system, and CPU
architecture that you need. That is expected for PyO3 and does not require a
separate hand-written C ABI; such an ABI can be added later for MATLAB, C, or
Fortran consumers.
