from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt


Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
Complex64Array = npt.NDArray[np.complex64]
FloatArray = npt.NDArray[np.float32] | npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex64] | npt.NDArray[np.complex128]


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



class CpuPoints:
    """A point set checked and sorted once by a CPU type-1 or type-2 plan."""

    def __len__(self) -> int: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def dtype(self) -> str: ...


class CpuType3Points:
    """Source and target sets prepared once by a CPU type-3 plan."""

    @property
    def source_count(self) -> int: ...

    @property
    def target_count(self) -> int: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def dtype(self) -> str: ...


class CpuType1Plan:
    """Reusable CPU type-1 NUFFT plan; `dtype` defaults to complex128."""

    def __init__(
        self,
        n_modes: Sequence[int],
        eps: float = 1.0e-6,
        isign: int = 1,
        mode_order: str = "centered",
        batch: int = 1,
        dtype: npt.DTypeLike | None = None,
        threads: int | None = None,
    ) -> None: ...

    def prepare_points(self, points: FloatArray) -> CpuPoints: ...

    def execute(self, points: FloatArray | CpuPoints, strengths: ComplexArray) -> ComplexArray: ...

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

    @property
    def dtype(self) -> str: ...

    @property
    def threads(self) -> int: ...

    @property
    def fine_grid_shape(self) -> list[int]: ...


class CpuType2Plan:
    """Reusable CPU type-2 NUFFT plan; `dtype` defaults to complex128."""

    def __init__(
        self,
        n_modes: Sequence[int],
        eps: float = 1.0e-6,
        isign: int = 1,
        mode_order: str = "centered",
        batch: int = 1,
        dtype: npt.DTypeLike | None = None,
        threads: int | None = None,
    ) -> None: ...

    def prepare_points(self, points: FloatArray) -> CpuPoints: ...

    def execute(
        self, points: FloatArray | CpuPoints, coefficients: ComplexArray
    ) -> ComplexArray: ...

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

    @property
    def dtype(self) -> str: ...

    @property
    def threads(self) -> int: ...

    @property
    def fine_grid_shape(self) -> list[int]: ...


class CpuType3Plan:
    """Reusable CPU type-3 NUFFT plan with fixed coordinate bounds."""

    def __init__(
        self,
        source_bounds: Float64Array,
        target_bounds: Float64Array,
        eps: float = 1.0e-6,
        isign: int = 1,
        batch: int = 1,
        dtype: npt.DTypeLike | None = None,
        threads: int | None = None,
    ) -> None: ...

    def prepare_points(
        self, source_points: FloatArray, target_points: FloatArray
    ) -> CpuType3Points: ...

    def execute(
        self,
        source_points: FloatArray | CpuType3Points,
        strengths: ComplexArray,
        target_points: FloatArray | None = None,
    ) -> ComplexArray: ...

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

    @property
    def dtype(self) -> str: ...

    @property
    def threads(self) -> int: ...


__version__: str
