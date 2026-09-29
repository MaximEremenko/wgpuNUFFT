//! NumPy bindings of the CPU plans.
//!
//! The plans take `float32` points with `complex64` values or `float64`
//! points with `complex128` values. Contiguous NumPy inputs are read in
//! place, and the transforms run without the GIL.

use std::any::Any;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, PoisonError};

use numpy::ndarray::{ArrayD, ArrayViewD, Axis, CowArray, IxDyn};
use numpy::{
    Complex32, Complex64, Element, IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use wgpu_nufft::{
    CpuNufftPlan, CpuNufftPoints, CpuNufftType3Plan, CpuNufftType3Points, FftPrecision, ModeOrder,
    NufftConfig, NufftError, NufftInterval, NufftType3Config,
};

use super::{
    checked_mode_count, copy_bounds, mode_batch, parse_mode_order, parse_sign, vector_batch,
    WgpuNufftError,
};

/// Value type of a CPU plan.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dtype {
    /// `complex64` values and `float32` points.
    Complex64,
    /// `complex128` values and `float64` points.
    Complex128,
}

impl Dtype {
    /// Accepts anything `numpy.dtype` does; `None` means `complex128`.
    fn parse(value: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(value) = value else {
            return Ok(Self::Complex128);
        };
        let name: String = value
            .py()
            .import("numpy")?
            .getattr("dtype")?
            .call1((value,))?
            .getattr("name")?
            .extract()?;
        match name.as_str() {
            "complex64" => Ok(Self::Complex64),
            "complex128" => Ok(Self::Complex128),
            _ => Err(PyValueError::new_err(
                "dtype must be complex64 or complex128",
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
        }
    }

    fn precision(self) -> FftPrecision {
        match self {
            Self::Complex64 => FftPrecision::F32,
            Self::Complex128 => FftPrecision::F64,
        }
    }
}

/// A real scalar of the CPU plans and its complex NumPy type.
trait Scalar: Element + bytemuck::Pod + Send + Sync {
    type Complex: Element + bytemuck::Pod + Send + Sync;
    const POINTS: &'static str;
    const VALUES: &'static str;

    fn execute(
        plan: &CpuNufftPlan,
        points: &[Self],
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError>;

    fn prepare(plan: &CpuNufftPlan, points: &[Self]) -> Result<CpuNufftPoints, NufftError>;

    fn execute_prepared(
        plan: &CpuNufftPlan,
        points: &CpuNufftPoints,
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError>;

    fn prepare_type3(
        plan: &CpuNufftType3Plan,
        sources: &[Self],
        targets: &[Self],
    ) -> Result<CpuNufftType3Points, NufftError>;

    fn execute_type3(
        plan: &CpuNufftType3Plan,
        points: &CpuNufftType3Points,
        strengths: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError>;
}

impl Scalar for f32 {
    type Complex = Complex32;
    const POINTS: &'static str = "float32";
    const VALUES: &'static str = "complex64";

    fn execute(
        plan: &CpuNufftPlan,
        points: &[Self],
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError> {
        plan.execute(points, input, output)
    }

    fn prepare(plan: &CpuNufftPlan, points: &[Self]) -> Result<CpuNufftPoints, NufftError> {
        plan.prepare_points(points)
    }

    fn execute_prepared(
        plan: &CpuNufftPlan,
        points: &CpuNufftPoints,
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError> {
        plan.execute_prepared(points, input, output)
    }

    fn prepare_type3(
        plan: &CpuNufftType3Plan,
        sources: &[Self],
        targets: &[Self],
    ) -> Result<CpuNufftType3Points, NufftError> {
        plan.prepare_points(sources, targets)
    }

    fn execute_type3(
        plan: &CpuNufftType3Plan,
        points: &CpuNufftType3Points,
        strengths: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError> {
        plan.execute_prepared(points, strengths, output)
    }
}

impl Scalar for f64 {
    type Complex = Complex64;
    const POINTS: &'static str = "float64";
    const VALUES: &'static str = "complex128";

    fn execute(
        plan: &CpuNufftPlan,
        points: &[Self],
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError> {
        plan.execute_f64(points, input, output)
    }

    fn prepare(plan: &CpuNufftPlan, points: &[Self]) -> Result<CpuNufftPoints, NufftError> {
        plan.prepare_points_f64(points)
    }

    fn execute_prepared(
        plan: &CpuNufftPlan,
        points: &CpuNufftPoints,
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError> {
        plan.execute_prepared_f64(points, input, output)
    }

    fn prepare_type3(
        plan: &CpuNufftType3Plan,
        sources: &[Self],
        targets: &[Self],
    ) -> Result<CpuNufftType3Points, NufftError> {
        plan.prepare_points_f64(sources, targets)
    }

    fn execute_type3(
        plan: &CpuNufftType3Plan,
        points: &CpuNufftType3Points,
        strengths: &[Self],
        output: &mut [Self],
    ) -> Result<(), NufftError> {
        plan.execute_prepared_f64(points, strengths, output)
    }
}

/// Maps a core error to `ValueError` for bad arguments and
/// `WgpuNufftError` otherwise.
fn core_error(error: NufftError) -> PyErr {
    match error {
        NufftError::NonFiniteCoordinate { point, axis, .. } => PyValueError::new_err(format!(
            "points contain a non-finite coordinate at point {point}, axis {axis}"
        )),
        NufftError::PointOutOfRange { point, axis, .. } => PyValueError::new_err(format!(
            "points coordinate at point {point}, axis {axis} is outside [-3*pi, 3*pi]"
        )),
        NufftError::Type3NonFiniteCoordinate { .. }
        | NufftError::Type3CoordinateOutOfBounds { .. }
        | NufftError::PointSetMismatch { .. } => PyValueError::new_err(error.to_string()),
        NufftError::InvalidBatch { .. }
        | NufftError::InvalidDimensions { .. }
        | NufftError::ZeroMode { .. }
        | NufftError::InvalidTolerance { .. }
        | NufftError::InvalidSigma { .. }
        | NufftError::KernelWidthUnsupported { .. }
        | NufftError::InvalidType3Interval { .. }
        | NufftError::Type3GridSizeUnsupported { .. }
        | NufftError::Type3GridVolumeUnsupported { .. }
        | NufftError::Type3RescalingUnsupported { .. }
        | NufftError::FineGridTooLarge { .. } => {
            PyValueError::new_err(format!("invalid NUFFT plan: {error}"))
        }
        error => WgpuNufftError::new_err(error.to_string()),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic")
}

/// Runs `work` without the GIL, reporting errors and panics as Python
/// exceptions.
fn detached<R: Send>(
    py: Python<'_>,
    work: impl FnOnce() -> Result<R, NufftError> + Send,
) -> PyResult<R> {
    py.detach(|| std::panic::catch_unwind(AssertUnwindSafe(work)))
        .map_err(|payload| {
            WgpuNufftError::new_err(format!(
                "CPU NUFFT panicked: {}",
                panic_message(payload.as_ref())
            ))
        })?
        .map_err(core_error)
}

/// A NumPy array of `T`, or a `TypeError` naming the expected dtype.
fn array<'py, T: Element>(
    name: &str,
    dtype: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    value
        .extract::<PyReadonlyArrayDyn<'py, T>>()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be a {dtype} NumPy array")))
}

/// `view` in C order, copied only when it is not already.
fn c_order<'a, T: Clone>(view: ArrayViewD<'a, T>) -> CowArray<'a, T, IxDyn> {
    if view.is_standard_layout() {
        view.into()
    } else {
        view.as_standard_layout().into_owned().into()
    }
}

/// The C-ordered words of `(M, d)` points (or `(M,)` in 1D) and `M`.
fn point_words<'a, T: Element + Copy>(
    name: &str,
    points: &'a PyReadonlyArrayDyn<'_, T>,
    dimensions: usize,
) -> PyResult<(CowArray<'a, T, IxDyn>, usize)> {
    let count = match points.shape() {
        [count] if dimensions == 1 => *count,
        [count, width] if *width == dimensions => *count,
        _ => {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {dimensions}){}",
                if dimensions == 1 { " or (M,)" } else { "" }
            )))
        }
    };
    Ok((c_order(points.as_array()), count))
}

/// A zeroed output of `len` values. Its pages are mapped as the transform
/// writes them, without a separate pass.
fn zeroed<C: bytemuck::Zeroable>(len: usize) -> PyResult<Vec<C>> {
    bytemuck::allocation::try_zeroed_vec(len)
        .map_err(|()| PyMemoryError::new_err(format!("could not allocate {len} output values")))
}

fn words<T: Scalar>(values: &[T::Complex]) -> &[T] {
    bytemuck::cast_slice(values)
}

fn words_mut<T: Scalar>(values: &mut [T::Complex]) -> &mut [T] {
    bytemuck::cast_slice_mut(values)
}

/// The permutation between NumPy's `(batch, *n_modes)` order and the core
/// order, whose dimension zero is fastest: the mode axes reversed.
fn reversed_mode_axes(dimensions: usize) -> Vec<usize> {
    std::iter::once(0).chain((1..=dimensions).rev()).collect()
}

/// Core-ordered modes of a NumPy `(batch, *n_modes)` array.
fn core_modes<T: Clone>(view: ArrayViewD<'_, T>) -> CowArray<'_, T, IxDyn> {
    let dimensions = view.ndim() - 1;
    c_order(view.permuted_axes(reversed_mode_axes(dimensions)))
}

/// A NumPy-ordered `(batch, *n_modes)` array of core-ordered `values`,
/// without the batch axis unless `explicit_batch`.
fn numpy_modes<T: Clone>(
    values: Vec<T>,
    batch: usize,
    n_modes: &[usize],
    explicit_batch: bool,
) -> PyResult<ArrayD<T>> {
    let shape = std::iter::once(batch)
        .chain(n_modes.iter().rev().copied())
        .collect::<Vec<_>>();
    let core = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|error| WgpuNufftError::new_err(format!("could not shape the modes: {error}")))?;
    let modes = core.permuted_axes(reversed_mode_axes(n_modes.len()));
    let modes = if modes.is_standard_layout() {
        modes
    } else {
        modes.as_standard_layout().into_owned()
    };
    Ok(if explicit_batch {
        modes
    } else {
        modes.index_axis_move(Axis(0), 0)
    })
}

fn point_values<T>(values: Vec<T>, count: usize, batch: usize, explicit_batch: bool) -> ArrayD<T> {
    let shape = if explicit_batch {
        vec![batch, count]
    } else {
        vec![count]
    };
    ArrayD::from_shape_vec(IxDyn(&shape), values).expect("one value per point and transform")
}

/// Plans of one configuration, one per batch size in use, so that a plan
/// created with `batch=B` runs any batch of 1 to `B` transforms.
struct Plans<P> {
    plans: Mutex<HashMap<usize, Arc<P>>>,
    make: Box<dyn Fn(usize) -> Result<P, NufftError> + Send + Sync>,
}

impl<P> Plans<P> {
    fn new(
        batch: usize,
        make: impl Fn(usize) -> Result<P, NufftError> + Send + Sync + 'static,
    ) -> PyResult<Self> {
        let plan = make(batch).map_err(core_error)?;
        Ok(Self {
            plans: Mutex::new(HashMap::from([(batch, Arc::new(plan))])),
            make: Box::new(make),
        })
    }

    fn get(&self, batch: usize) -> PyResult<Arc<P>> {
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(plan) = plans.get(&batch) {
            return Ok(Arc::clone(plan));
        }
        let plan = Arc::new((self.make)(batch).map_err(core_error)?);
        plans.insert(batch, Arc::clone(&plan));
        Ok(plan)
    }
}

fn configure(plan: CpuNufftPlan, threads: Option<usize>) -> CpuNufftPlan {
    match threads {
        Some(threads) => plan.with_threads(threads),
        None => plan,
    }
}

fn check_threads(threads: Option<usize>) -> PyResult<()> {
    if threads == Some(0) {
        return Err(PyValueError::new_err("threads must be at least 1"));
    }
    Ok(())
}

/// A point set checked and sorted once by `CpuType1Plan.prepare_points` or
/// `CpuType2Plan.prepare_points`, for repeated executions.
#[pyclass(name = "CpuPoints", module = "wgpu_nufft._native", frozen)]
struct PyCpuPoints {
    points: CpuNufftPoints,
    dtype: Dtype,
}

#[pymethods]
impl PyCpuPoints {
    fn __len__(&self) -> usize {
        self.points.len()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.points.dimensions()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "CpuPoints(len={}, dimensions={}, dtype={:?})",
            self.points.len(),
            self.points.dimensions(),
            self.dtype.name()
        )
    }
}

/// Source and target sets checked, rescaled, and sorted once by
/// `CpuType3Plan.prepare_points`, for repeated executions.
#[pyclass(name = "CpuType3Points", module = "wgpu_nufft._native", frozen)]
struct PyCpuType3Points {
    points: CpuNufftType3Points,
    dtype: Dtype,
}

#[pymethods]
impl PyCpuType3Points {
    #[getter]
    fn source_count(&self) -> usize {
        self.points.source_count()
    }

    #[getter]
    fn target_count(&self) -> usize {
        self.points.target_count()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.points.dimensions()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "CpuType3Points(source_count={}, target_count={}, dimensions={}, dtype={:?})",
            self.points.source_count(),
            self.points.target_count(),
            self.points.dimensions(),
            self.dtype.name()
        )
    }
}

/// The points of a type-1 or type-2 execution.
enum PointInput<'a, T: Clone> {
    Prepared(&'a CpuNufftPoints),
    /// C-ordered coordinates and the point count.
    Coordinates(CowArray<'a, T, IxDyn>, usize),
}

/// Checks that prepared points hold the plan's dtype.
fn check_dtype(prepared: Dtype, plan: Dtype) -> PyResult<()> {
    if prepared != plan {
        return Err(PyValueError::new_err(format!(
            "the points were prepared for {} values, but this plan takes {}",
            prepared.name(),
            plan.name()
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Type1,
    Type2,
}

/// The state that type-1 and type-2 plans share.
struct Type12 {
    kind: Kind,
    plans: Plans<CpuNufftPlan>,
    n_modes: Vec<usize>,
    mode_count: usize,
    eps: f64,
    isign: i32,
    mode_order: ModeOrder,
    batch_capacity: usize,
    dtype: Dtype,
}

impl Type12 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: Kind,
        n_modes: Vec<usize>,
        eps: f64,
        isign: i32,
        mode_order: &str,
        batch: usize,
        dtype: Option<&Bound<'_, PyAny>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let mode_count = checked_mode_count(&n_modes)?;
        let isign = parse_sign(isign)?;
        let order = parse_mode_order(mode_order)?;
        let dtype = Dtype::parse(dtype)?;
        check_threads(threads)?;
        let config = NufftConfig::new(n_modes.clone(), eps)
            .with_isign(isign)
            .with_mode_order(order)
            .with_precision(dtype.precision());
        let plans = Plans::new(batch, move |batch| {
            let config = config.clone().with_batch(batch);
            let plan = match kind {
                Kind::Type1 => CpuNufftPlan::type1(config),
                Kind::Type2 => CpuNufftPlan::type2(config),
            };
            plan.map(|plan| configure(plan, threads))
        })?;
        Ok(Self {
            kind,
            plans,
            n_modes,
            mode_count,
            eps,
            isign,
            mode_order: order,
            batch_capacity: batch,
            dtype,
        })
    }

    fn prepare_points(&self, py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<PyCpuPoints> {
        match self.dtype {
            Dtype::Complex64 => self.prepare_as::<f32>(py, points),
            Dtype::Complex128 => self.prepare_as::<f64>(py, points),
        }
    }

    fn prepare_as<T: Scalar>(
        &self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
    ) -> PyResult<PyCpuPoints> {
        let points = array::<T>("points", T::POINTS, points)?;
        let (coordinates, _) = point_words("points", &points, self.n_modes.len())?;
        let coordinates = coordinates.as_slice().expect("C-ordered points");
        let plan = self.plans.get(self.batch_capacity)?;
        let points = detached(py, || T::prepare(&plan, coordinates))?;
        Ok(PyCpuPoints {
            points,
            dtype: self.dtype,
        })
    }

    fn execute<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.dtype {
            Dtype::Complex64 => self.execute_as::<f32>(py, points, values),
            Dtype::Complex128 => self.execute_as::<f64>(py, points, values),
        }
    }

    fn execute_as<'py, T: Scalar>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // The points: a prepared set, or C-ordered coordinates.
        let point_array;
        let points = match points.cast::<PyCpuPoints>() {
            Ok(prepared) => {
                let prepared = prepared.get();
                check_dtype(prepared.dtype, self.dtype)?;
                PointInput::Prepared(&prepared.points)
            }
            Err(_) => {
                point_array = array::<T>("points", T::POINTS, points)?;
                let (words, count) = point_words("points", &point_array, self.n_modes.len())?;
                PointInput::Coordinates(words, count)
            }
        };
        let point_count = match &points {
            PointInput::Prepared(prepared) => prepared.len(),
            PointInput::Coordinates(_, count) => *count,
        };

        let name = match self.kind {
            Kind::Type1 => "strengths",
            Kind::Type2 => "coefficients",
        };
        let values = array::<T::Complex>(name, T::VALUES, values)?;
        let (batch, explicit_batch) = match self.kind {
            Kind::Type1 => vector_batch(name, values.shape(), point_count, self.batch_capacity)?,
            Kind::Type2 => mode_batch(name, values.shape(), &self.n_modes, self.batch_capacity)?,
        };
        let input = match self.kind {
            Kind::Type1 => c_order(values.as_array()),
            Kind::Type2 if explicit_batch => core_modes(values.as_array()),
            Kind::Type2 => core_modes(values.as_array().insert_axis(Axis(0))),
        };
        let input = words::<T>(input.as_slice().expect("C-ordered values"));
        let output_len = match self.kind {
            Kind::Type1 => batch * self.mode_count,
            Kind::Type2 => batch * point_count,
        };
        let mut output = zeroed::<T::Complex>(output_len)?;
        let plan = self.plans.get(batch)?;
        detached(py, || {
            let output = words_mut::<T>(&mut output);
            match &points {
                PointInput::Prepared(prepared) => {
                    T::execute_prepared(&plan, prepared, input, output)
                }
                PointInput::Coordinates(words, _) => {
                    let coordinates = words.as_slice().expect("C-ordered points");
                    T::execute(&plan, coordinates, input, output)
                }
            }
        })?;
        let output = match self.kind {
            Kind::Type1 => numpy_modes(output, batch, &self.n_modes, explicit_batch)?,
            Kind::Type2 => point_values(output, point_count, batch, explicit_batch),
        };
        Ok(output.into_pyarray(py).into_any())
    }

    fn mode_order(&self) -> &'static str {
        match self.mode_order {
            ModeOrder::Centered => "centered",
            ModeOrder::Fft => "fft",
        }
    }

    fn fine_grid_shape(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .plans
            .get(self.batch_capacity)?
            .fine_grid_shape()
            .to_vec())
    }

    fn threads(&self) -> PyResult<usize> {
        Ok(self.plans.get(self.batch_capacity)?.threads())
    }
}

/// Reusable CPU type-1 NUFFT plan: nonuniform points to Fourier modes.
#[pyclass(module = "wgpu_nufft._native", frozen)]
struct CpuType1Plan {
    inner: Type12,
}

#[pymethods]
impl CpuType1Plan {
    #[new]
    #[pyo3(signature = (n_modes, eps=1.0e-6, isign=1, mode_order="centered", batch=1, dtype=None, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_modes: Vec<usize>,
        eps: f64,
        isign: i32,
        mode_order: &str,
        batch: usize,
        dtype: Option<&Bound<'_, PyAny>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Type12::new(
                Kind::Type1,
                n_modes,
                eps,
                isign,
                mode_order,
                batch,
                dtype,
                threads,
            )?,
        })
    }

    /// Check and sort `points` once, for repeated executions.
    fn prepare_points(&self, py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<PyCpuPoints> {
        self.inner.prepare_points(py, points)
    }

    /// Execute one transform or a transform-major batch at `points`, an
    /// array or a prepared point set.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
        strengths: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.execute(py, points, strengths)
    }

    #[getter]
    fn n_modes(&self) -> Vec<usize> {
        self.inner.n_modes.clone()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.n_modes.len()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.eps
    }

    #[getter]
    fn isign(&self) -> i32 {
        self.inner.isign
    }

    #[getter]
    fn mode_order(&self) -> &'static str {
        self.inner.mode_order()
    }

    #[getter]
    fn batch_capacity(&self) -> usize {
        self.inner.batch_capacity
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype.name()
    }

    #[getter]
    fn threads(&self) -> PyResult<usize> {
        self.inner.threads()
    }

    #[getter]
    fn fine_grid_shape(&self) -> PyResult<Vec<usize>> {
        self.inner.fine_grid_shape()
    }
}

/// Reusable CPU type-2 NUFFT plan: Fourier modes to nonuniform points.
#[pyclass(module = "wgpu_nufft._native", frozen)]
struct CpuType2Plan {
    inner: Type12,
}

#[pymethods]
impl CpuType2Plan {
    #[new]
    #[pyo3(signature = (n_modes, eps=1.0e-6, isign=1, mode_order="centered", batch=1, dtype=None, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_modes: Vec<usize>,
        eps: f64,
        isign: i32,
        mode_order: &str,
        batch: usize,
        dtype: Option<&Bound<'_, PyAny>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Type12::new(
                Kind::Type2,
                n_modes,
                eps,
                isign,
                mode_order,
                batch,
                dtype,
                threads,
            )?,
        })
    }

    /// Check and sort `points` once, for repeated executions.
    fn prepare_points(&self, py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<PyCpuPoints> {
        self.inner.prepare_points(py, points)
    }

    /// Execute one transform or a transform-major batch at `points`, an
    /// array or a prepared point set.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
        coefficients: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.execute(py, points, coefficients)
    }

    #[getter]
    fn n_modes(&self) -> Vec<usize> {
        self.inner.n_modes.clone()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.n_modes.len()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.eps
    }

    #[getter]
    fn isign(&self) -> i32 {
        self.inner.isign
    }

    #[getter]
    fn mode_order(&self) -> &'static str {
        self.inner.mode_order()
    }

    #[getter]
    fn batch_capacity(&self) -> usize {
        self.inner.batch_capacity
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype.name()
    }

    #[getter]
    fn threads(&self) -> PyResult<usize> {
        self.inner.threads()
    }

    #[getter]
    fn fine_grid_shape(&self) -> PyResult<Vec<usize>> {
        self.inner.fine_grid_shape()
    }
}

/// Reusable CPU type-3 NUFFT plan with fixed source and target bounds.
#[pyclass(module = "wgpu_nufft._native", frozen)]
struct CpuType3Plan {
    plans: Plans<CpuNufftType3Plan>,
    source_bounds: Vec<NufftInterval>,
    target_bounds: Vec<NufftInterval>,
    eps: f64,
    isign: i32,
    batch_capacity: usize,
    dtype: Dtype,
}

impl CpuType3Plan {
    fn prepare_as<T: Scalar>(
        &self,
        py: Python<'_>,
        source_points: &Bound<'_, PyAny>,
        target_points: &Bound<'_, PyAny>,
    ) -> PyResult<PyCpuType3Points> {
        let dimensions = self.source_bounds.len();
        let sources = array::<T>("source_points", T::POINTS, source_points)?;
        let targets = array::<T>("target_points", T::POINTS, target_points)?;
        let (sources, _) = point_words("source_points", &sources, dimensions)?;
        let (targets, _) = point_words("target_points", &targets, dimensions)?;
        let (sources, targets) = (
            sources.as_slice().expect("C-ordered points"),
            targets.as_slice().expect("C-ordered points"),
        );
        let plan = self.plans.get(self.batch_capacity)?;
        let points = detached(py, || T::prepare_type3(&plan, sources, targets))?;
        Ok(PyCpuType3Points {
            points,
            dtype: self.dtype,
        })
    }

    fn execute_as<'py, T: Scalar>(
        &self,
        py: Python<'py>,
        points: &CpuNufftType3Points,
        strengths: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let strengths = array::<T::Complex>("strengths", T::VALUES, strengths)?;
        let (batch, explicit_batch) = vector_batch(
            "strengths",
            strengths.shape(),
            points.source_count(),
            self.batch_capacity,
        )?;
        let strengths = c_order(strengths.as_array());
        let strengths = strengths.as_slice().expect("C-ordered values");
        let mut output = zeroed::<T::Complex>(batch * points.target_count())?;
        let plan = self.plans.get(batch)?;
        detached(py, || {
            T::execute_type3(
                &plan,
                points,
                words::<T>(strengths),
                words_mut::<T>(&mut output),
            )
        })?;
        let output = point_values(output, points.target_count(), batch, explicit_batch);
        Ok(output.into_pyarray(py).into_any())
    }

    fn execute_with<'py, T: Scalar>(
        &self,
        py: Python<'py>,
        source_points: &Bound<'py, PyAny>,
        strengths: &Bound<'py, PyAny>,
        target_points: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(prepared) = source_points.cast::<PyCpuType3Points>() {
            let prepared = prepared.get();
            if target_points.is_some() {
                return Err(PyValueError::new_err(
                    "target_points must be omitted with prepared points",
                ));
            }
            check_dtype(prepared.dtype, self.dtype)?;
            return self.execute_as::<T>(py, &prepared.points, strengths);
        }
        let target_points = target_points.ok_or_else(|| {
            PyValueError::new_err("target_points is required unless the points are prepared")
        })?;
        let prepared = self.prepare_as::<T>(py, source_points, target_points)?;
        self.execute_as::<T>(py, &prepared.points, strengths)
    }
}

#[pymethods]
impl CpuType3Plan {
    #[new]
    #[pyo3(signature = (source_bounds, target_bounds, eps=1.0e-6, isign=1, batch=1, dtype=None, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_bounds: PyReadonlyArrayDyn<'_, f64>,
        target_bounds: PyReadonlyArrayDyn<'_, f64>,
        eps: f64,
        isign: i32,
        batch: usize,
        dtype: Option<&Bound<'_, PyAny>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let isign = parse_sign(isign)?;
        let source_bounds = copy_bounds("source_bounds", source_bounds)?;
        let target_bounds = copy_bounds("target_bounds", target_bounds)?;
        if source_bounds.len() != target_bounds.len() {
            return Err(PyValueError::new_err(
                "source_bounds and target_bounds must have the same number of rows",
            ));
        }
        let dtype = Dtype::parse(dtype)?;
        check_threads(threads)?;
        let config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), eps)
            .with_isign(isign)
            .with_precision(dtype.precision());
        let plans = Plans::new(batch, move |batch| {
            CpuNufftType3Plan::new(config.clone().with_batch(batch)).map(|plan| match threads {
                Some(threads) => plan.with_threads(threads),
                None => plan,
            })
        })?;
        Ok(Self {
            plans,
            source_bounds,
            target_bounds,
            eps,
            isign,
            batch_capacity: batch,
            dtype,
        })
    }

    /// Check, rescale, and sort source and target points once, for repeated
    /// executions.
    fn prepare_points(
        &self,
        py: Python<'_>,
        source_points: &Bound<'_, PyAny>,
        target_points: &Bound<'_, PyAny>,
    ) -> PyResult<PyCpuType3Points> {
        match self.dtype {
            Dtype::Complex64 => self.prepare_as::<f32>(py, source_points, target_points),
            Dtype::Complex128 => self.prepare_as::<f64>(py, source_points, target_points),
        }
    }

    /// Execute one transform or a transform-major batch, from source and
    /// target arrays or from prepared points (then without `target_points`).
    #[pyo3(signature = (source_points, strengths, target_points=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        source_points: &Bound<'py, PyAny>,
        strengths: &Bound<'py, PyAny>,
        target_points: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.dtype {
            Dtype::Complex64 => {
                self.execute_with::<f32>(py, source_points, strengths, target_points)
            }
            Dtype::Complex128 => {
                self.execute_with::<f64>(py, source_points, strengths, target_points)
            }
        }
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.source_bounds.len()
    }

    #[getter]
    fn source_bounds(&self) -> Vec<(f64, f64)> {
        self.source_bounds
            .iter()
            .map(|interval| (interval.lower(), interval.upper()))
            .collect()
    }

    #[getter]
    fn target_bounds(&self) -> Vec<(f64, f64)> {
        self.target_bounds
            .iter()
            .map(|interval| (interval.lower(), interval.upper()))
            .collect()
    }

    #[getter]
    fn outer_grid_shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.plans.get(self.batch_capacity)?.outer_grid_shape())
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.eps
    }

    #[getter]
    fn isign(&self) -> i32 {
        self.isign
    }

    #[getter]
    fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }

    #[getter]
    fn threads(&self) -> PyResult<usize> {
        Ok(self.plans.get(self.batch_capacity)?.threads())
    }
}

/// Adds the CPU classes to the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCpuPoints>()?;
    module.add_class::<PyCpuType3Points>()?;
    module.add_class::<CpuType1Plan>()?;
    module.add_class::<CpuType2Plan>()?;
    module.add_class::<CpuType3Plan>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_axes_round_trip_between_numpy_and_core_order() {
        // NumPy (1, 2, 3) modes in C order hold 0..6; the core order runs
        // over axis zero fastest.
        let numpy =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), (0..6).collect::<Vec<i32>>()).unwrap();
        let core = core_modes(numpy.view());
        assert_eq!(core.as_slice().unwrap(), &[0, 3, 1, 4, 2, 5]);
        let back = numpy_modes(core.iter().copied().collect(), 1, &[2, 3], false).unwrap();
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.as_slice().unwrap(), &[0, 1, 2, 3, 4, 5]);
    }
}
