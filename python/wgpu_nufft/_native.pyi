from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt


Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
Complex64Array = npt.NDArray[np.complex64]


class WgpuNufftError(RuntimeError): ...


class Context:
    """Persistent native wgpu device shared by reusable NUFFT plans."""

    def __init__(
        self,
        power_preference: str = "high-performance",
        force_fallback: bool = False,
    ) -> None: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def device_type(self) -> str: ...

    @property
    def driver(self) -> str: ...

    @property
    def driver_info(self) -> str: ...


class Type1Plan:
    """Reusable f32 type-1 NUFFT plan."""

    def __init__(
        self,
        context: Context,
        n_modes: Sequence[int],
        eps: float = 1.0e-6,
        isign: int = 1,
        mode_order: str = "centered",
        batch: int = 1,
    ) -> None: ...

    def execute(self, points: Float32Array, strengths: Complex64Array) -> Complex64Array: ...

    @property
    def n_modes(self) -> list[int]: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def eps(self) -> float: ...

    @property
    def isign(self) -> int: ...

    @property
    def mode_order(self) -> str: ...

    @property
    def batch_capacity(self) -> int: ...


class Type2Plan:
    """Reusable f32 type-2 NUFFT plan."""

    def __init__(
        self,
        context: Context,
        n_modes: Sequence[int],
        eps: float = 1.0e-6,
        isign: int = 1,
        mode_order: str = "centered",
        batch: int = 1,
    ) -> None: ...

    def execute(self, points: Float32Array, coefficients: Complex64Array) -> Complex64Array: ...

    @property
    def n_modes(self) -> list[int]: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def eps(self) -> float: ...

    @property
    def isign(self) -> int: ...

    @property
    def mode_order(self) -> str: ...

    @property
    def batch_capacity(self) -> int: ...


class Type3Plan:
    """Reusable f32 type-3 NUFFT plan with fixed coordinate bounds."""

    def __init__(
        self,
        context: Context,
        source_bounds: Float64Array,
        target_bounds: Float64Array,
        eps: float = 1.0e-6,
        isign: int = 1,
        batch: int = 1,
    ) -> None: ...

    def execute(
        self,
        source_points: Float32Array,
        strengths: Complex64Array,
        target_points: Float32Array,
    ) -> Complex64Array: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def source_bounds(self) -> list[tuple[float, float]]: ...

    @property
    def target_bounds(self) -> list[tuple[float, float]]: ...

    @property
    def outer_grid_shape(self) -> list[int]: ...

    @property
    def eps(self) -> float: ...

    @property
    def isign(self) -> int: ...

    @property
    def batch_capacity(self) -> int: ...


__version__: str
