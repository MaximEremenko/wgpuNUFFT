//! Host-memory plans; see [`CpuNufftPlan`] and [`CpuNufftType3Plan`].

mod grid;
mod modes;
mod points;
mod real;
mod spread;
mod type3;
mod workers;

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use wgpu_fft::{CpuFftPlan, FftConfig, FftDirection, FftPrecision, Normalization};

use crate::config::{NufftConfig, NufftSign};
use crate::error::{NufftError, Result};
use crate::kernel::{select_fine_grid_size, EsKernel};
use crate::plan::NufftKind;
use grid::{FineGrid, Weights};
use modes::{deconvolve, predeconvolve, AxisModes};
use points::{Slabs, SortedPoints};
use real::{Coordinate, Real};
use spread::{dense_groups, interpolate, spread, spread_dense, Points};
use workers::{available_threads, for_each_chunk_group};

pub use type3::CpuNufftType3Plan;

/// Largest coordinate magnitude of the type-1 and type-2 contract.
const MAX_COORDINATE_MAGNITUDE: f64 = 3.0 * std::f64::consts::PI;
/// Preferred slab thickness along the last fine-grid axis, in cells: in 1D
/// a slab is a stretch of the grid, in more dimensions a stack of planes.
const SLAB_CELLS_1D: usize = 4096;
const SLAB_PLANES: usize = 16;
/// Slabs to aim for in more than one dimension, to keep threads busy.
const MIN_SLABS: usize = 64;
/// Values per thread below which the output copy stays on one thread.
const MIN_VALUES_PER_THREAD: usize = 1 << 15;

/// Host-memory type-1 or type-2 NUFFT plan, for machines without a usable
/// GPU.
///
/// It mirrors [`NufftPlan`](crate::NufftPlan) on the CPU: it takes the same
/// [`NufftConfig`] and uses the same layouts as the GPU buffers, with
/// point-major coordinates, interleaved complex values,
/// `[re_hi, re_lo, im_hi, im_lo]` words for `Df64`, transform-major batches,
/// and Fourier modes with dimension zero fastest. Plans support one to
/// [`MAX_NUFFT_DIMENSIONS`](crate::MAX_NUFFT_DIMENSIONS) dimensions.
///
/// Results are bitwise repeatable for any thread count. Spreading splits the
/// fine grid into slabs along its last axis, and each slab adds the points
/// that reach it in a fixed order; every other pass computes each output
/// word in one place. The kernel is evaluated from the same piecewise
/// polynomials as the GPU `F64` and `Df64` paths, and the fine-grid FFT runs
/// on `wgpu-fft`'s CPU backend. `F32` plans compute in `f32`; `F64` and
/// `Df64` plans compute in `f64`. A plan keeps its two fine-grid buffers
/// between executions.
///
/// ```
/// use wgpu_nufft::{CpuNufftPlan, NufftConfig};
///
/// let plan = CpuNufftPlan::type1(NufftConfig::new([16], 1.0e-6))?;
/// // Three 1D points and their interleaved complex strengths.
/// let points = [0.5f32, 1.0, -2.0];
/// let strengths = [1.0f32, 0.0, 0.5, -0.5, 0.25, 0.0];
/// let mut modes = vec![0.0f32; plan.required_output_len(points.len())?];
/// plan.execute(&points, &strengths, &mut modes)?;
/// # Ok::<(), wgpu_nufft::NufftError>(())
/// ```
pub struct CpuNufftPlan {
    kind: NufftKind,
    config: NufftConfig,
    kernel: EsKernel,
    fine_grid_shape: Vec<usize>,
    threads: usize,
    engine: Engine,
}

enum Engine {
    F32(TypedEngine<f32>),
    /// Used by both `F64` and `Df64` plans.
    F64(TypedEngine<f64>),
}

struct TypedEngine<T> {
    weights: Weights<T>,
    axes: Vec<AxisModes<T>>,
    fft: CpuFftPlan,
    /// The fine grid and its transform, kept between executions. An
    /// execution takes them out, so concurrent executions of one plan
    /// allocate their own.
    scratch: Mutex<Option<(Vec<T>, Vec<T>)>>,
}

impl std::fmt::Debug for CpuNufftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuNufftPlan")
            .field("kind", &self.kind)
            .field("config", &self.config)
            .field("kernel", &self.kernel)
            .field("fine_grid_shape", &self.fine_grid_shape)
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl CpuNufftPlan {
    /// Creates a type-1 plan: nonuniform points to Fourier modes.
    pub fn type1(config: NufftConfig) -> Result<Self> {
        Self::new(NufftKind::Type1, config)
    }

    /// Creates a type-2 plan: Fourier modes to nonuniform points.
    pub fn type2(config: NufftConfig) -> Result<Self> {
        Self::new(NufftKind::Type2, config)
    }

    fn new(kind: NufftKind, config: NufftConfig) -> Result<Self> {
        config.validate()?;
        let kernel = EsKernel::for_tolerance(config.eps(), config.sigma())?;
        let fine_grid_shape = config
            .n_modes()
            .iter()
            .map(|&length| select_fine_grid_size(length, config.sigma(), kernel.width()))
            .collect::<Result<Vec<_>>>()?;
        let table = kernel.horner_table();
        let coefficients = fine_grid_shape
            .iter()
            .map(|&length| kernel.centered_fourier_coefficients_horner(length, &table))
            .collect::<Result<Vec<_>>>()?;
        let direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_precision = match config.precision() {
            FftPrecision::F32 => FftPrecision::F32,
            FftPrecision::F64 | FftPrecision::Df64 => FftPrecision::F64,
        };
        let fft = CpuFftPlan::c2c(
            FftConfig::new_nd(fine_grid_shape.clone())
                .with_batch(config.batch())
                .with_direction(direction)
                .with_normalization(Normalization::None)
                .with_precision(fft_precision),
        )
        .map_err(|source| NufftError::FftShapeUnsupported {
            stage: "CPU oversampled-grid C2C plan",
            source,
        })?;
        let engine = match config.precision() {
            FftPrecision::F32 => Engine::F32(TypedEngine::new(
                &config,
                &table,
                &fine_grid_shape,
                &coefficients,
                fft,
            )?),
            FftPrecision::F64 | FftPrecision::Df64 => Engine::F64(TypedEngine::new(
                &config,
                &table,
                &fine_grid_shape,
                &coefficients,
                fft,
            )?),
        };
        Ok(Self {
            kind,
            config,
            kernel,
            fine_grid_shape,
            threads: available_threads(),
            engine,
        })
    }

    /// Sets the number of threads that spreading, interpolation, and
    /// deconvolution use; the default is the machine's available
    /// parallelism. Results do not depend on it. The fine-grid FFT picks its
    /// own thread count.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn kind(&self) -> NufftKind {
        self.kind
    }

    pub fn config(&self) -> &NufftConfig {
        &self.config
    }

    pub fn kernel(&self) -> EsKernel {
        self.kernel
    }

    /// Length of the oversampled fine grid on every axis.
    pub fn fine_grid_shape(&self) -> &[usize] {
        &self.fine_grid_shape
    }

    /// Words of the point slice for `point_count` points: `f32` words for
    /// `F32` and `Df64` plans (two per `Df64` coordinate), `f64` words for
    /// `F64` plans.
    pub fn required_points_len(&self, point_count: usize) -> Result<usize> {
        checked_product(
            "CPU point words",
            &[point_count, self.config.dimensions(), self.scalar_words()],
        )
    }

    /// Words of the input slice: type-1 strengths or type-2 Fourier
    /// coefficients, for every transform of the batch.
    pub fn required_input_len(&self, point_count: usize) -> Result<usize> {
        match self.kind {
            NufftKind::Type1 => self.complex_words("CPU type-1 strength words", point_count),
            NufftKind::Type2 => {
                self.complex_words("CPU type-2 coefficient words", self.config.mode_count()?)
            }
        }
    }

    /// Words of the output slice: type-1 Fourier modes or type-2 values at
    /// the points, for every transform of the batch.
    pub fn required_output_len(&self, point_count: usize) -> Result<usize> {
        match self.kind {
            NufftKind::Type1 => {
                self.complex_words("CPU type-1 mode words", self.config.mode_count()?)
            }
            NufftKind::Type2 => self.complex_words("CPU type-2 value words", point_count),
        }
    }

    /// Executes an `F32` or `Df64` plan on `f32` words.
    ///
    /// The point count follows from `points`. `F64` plans must use
    /// [`Self::execute_f64`].
    pub fn execute(&self, points: &[f32], input: &[f32], output: &mut [f32]) -> Result<()> {
        match (&self.engine, self.config.precision()) {
            (Engine::F32(engine), _) => {
                let point_count = self.validate_lengths(points.len(), input.len(), output.len())?;
                self.run(engine, points, input, output, point_count)
            }
            (Engine::F64(engine), FftPrecision::Df64) => {
                let point_count = self.validate_lengths(points.len(), input.len(), output.len())?;
                let points = join_df64(points, self.threads)?;
                let input = join_df64(input, self.threads)?;
                let mut result = filled("CPU Df64 output values", output.len() / 2, 0.0f64)?;
                self.run(engine, &points, &input, &mut result, point_count)?;
                split_df64(&result, output, self.threads);
                Ok(())
            }
            (Engine::F64(_), requested) => Err(NufftError::PrecisionUnsupported {
                requested,
                stage: "CPU NUFFT execution",
                reason: "f64-plans-execute-with-execute_f64",
            }),
        }
    }

    /// Executes an `F64` plan on `f64` words.
    ///
    /// The point count follows from `points`.
    pub fn execute_f64(&self, points: &[f64], input: &[f64], output: &mut [f64]) -> Result<()> {
        match (&self.engine, self.config.precision()) {
            (Engine::F64(engine), FftPrecision::F64) => {
                let point_count = self.validate_lengths(points.len(), input.len(), output.len())?;
                self.run(engine, points, input, output, point_count)
            }
            (_, requested) => Err(NufftError::PrecisionUnsupported {
                requested,
                stage: "CPU NUFFT execution",
                reason: "execute_f64-requires-an-f64-plan",
            }),
        }
    }

    /// Storage words per real scalar.
    fn scalar_words(&self) -> usize {
        match self.config.precision() {
            FftPrecision::Df64 => 2,
            FftPrecision::F32 | FftPrecision::F64 => 1,
        }
    }

    fn complex_words(&self, context: &'static str, values: usize) -> Result<usize> {
        checked_product(
            context,
            &[values, self.config.batch(), 2 * self.scalar_words()],
        )
    }

    /// Checks the slice lengths and returns the point count.
    fn validate_lengths(&self, points: usize, input: usize, output: usize) -> Result<usize> {
        let dimensions = self.config.dimensions();
        let point_words = dimensions * self.scalar_words();
        if !points.is_multiple_of(point_words) {
            return Err(NufftError::CoordinateLength {
                actual: points / self.scalar_words(),
                dimensions,
            });
        }
        let point_count = points / point_words;
        if u32::try_from(point_count).is_err() {
            return Err(NufftError::LengthOverflow {
                context: "CPU point count",
            });
        }
        let (input_name, output_name) = match self.kind {
            NufftKind::Type1 => ("CPU type-1 strengths", "CPU type-1 output modes"),
            NufftKind::Type2 => (
                "CPU type-2 Fourier coefficients",
                "CPU type-2 output values",
            ),
        };
        for (name, expected, actual) in [
            (input_name, self.required_input_len(point_count)?, input),
            (output_name, self.required_output_len(point_count)?, output),
        ] {
            if actual != expected {
                return Err(NufftError::InputLength {
                    input: name,
                    expected,
                    actual,
                });
            }
        }
        Ok(point_count)
    }

    fn run<T: Real, P: Coordinate>(
        &self,
        engine: &TypedEngine<T>,
        points: &[P],
        input: &[T],
        output: &mut [T],
        point_count: usize,
    ) -> Result<()> {
        let job = Job {
            plan: self,
            engine,
            points,
            input,
            point_count,
        };
        match self.config.dimensions() {
            1 => job.run::<1>(output),
            2 => job.run::<2>(output),
            3 => job.run::<3>(output),
            4 => job.run::<4>(output),
            5 => job.run::<5>(output),
            6 => job.run::<6>(output),
            7 => job.run::<7>(output),
            8 => job.run::<8>(output),
            dimensions => unreachable!("validated dimension count {dimensions}"),
        }
    }
}

impl<T: Real> TypedEngine<T> {
    fn new(
        config: &NufftConfig,
        table: &crate::kernel::EsHornerTable,
        fine_grid_shape: &[usize],
        coefficients: &[Vec<f64>],
        fft: CpuFftPlan,
    ) -> Result<Self> {
        let axes = config
            .n_modes()
            .iter()
            .zip(fine_grid_shape)
            .zip(coefficients)
            .map(|((&modes, &fine_length), coefficients)| {
                AxisModes::new(config.mode_order(), modes, fine_length, coefficients)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights: Weights::new(table),
            axes,
            fft,
            scratch: Mutex::new(None),
        })
    }
}

/// One execution of a plan on validated inputs.
struct Job<'a, T, P> {
    plan: &'a CpuNufftPlan,
    engine: &'a TypedEngine<T>,
    points: &'a [P],
    input: &'a [T],
    point_count: usize,
}

impl<T: Real, P: Coordinate> Job<'_, T, P> {
    /// Runs with axis-zero runs of the kernel width rounded up to 4, 8, or
    /// 16 cells.
    fn run<const D: usize>(&self, output: &mut [T]) -> Result<()> {
        match self.plan.kernel.width() {
            0..=4 => self.run_with::<D, 4>(output),
            5..=8 => self.run_with::<D, 8>(output),
            _ => self.run_with::<D, 16>(output),
        }
    }

    fn run_with<const D: usize, const RUN: usize>(&self, output: &mut [T]) -> Result<()> {
        let plan = self.plan;
        let config = &plan.config;
        let threads = plan.threads;
        let grid = FineGrid::<D>::new(&plan.fine_grid_shape, plan.kernel);
        let batch = config.batch();
        let points = Points {
            grid: &grid,
            weights: &self.engine.weights,
            coordinates: self.points,
            count: self.point_count,
            batch,
        };
        let last_length = grid.shape[D - 1];
        let slabs = Slabs::new(
            last_length,
            plan.kernel.width(),
            slab_thickness::<D>(last_length),
        )?;
        // Sorting also validates the coordinates, before any other work.
        let sorted = match self.point_count {
            0 => None,
            _ => Some(SortedPoints::new(&grid, &slabs, self.points, threads)?),
        };

        let fine_words = checked_product("CPU fine-grid words", &[grid.cells(), batch, 2])?;
        let taken = self
            .engine
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let (mut fine, mut transformed) = match taken {
            Some(buffers) => buffers,
            None => (
                zeroed("CPU fine grid", fine_words)?,
                zeroed("CPU transformed fine grid", fine_words)?,
            ),
        };
        let fft = |input: &[T], output: &mut [T], stage| {
            T::fft(&self.engine.fft, input, output)
                .map_err(|source| NufftError::FftExecutionFailed { stage, source })
        };
        match plan.kind {
            NufftKind::Type1 => {
                match &sorted {
                    Some(sorted) => spread_points::<T, P, D, RUN>(
                        &points,
                        bytemuck::cast_slice(self.input),
                        &slabs,
                        sorted,
                        bytemuck::cast_slice_mut(&mut fine),
                        threads,
                    )?,
                    None => fine.fill(T::ZERO),
                }
                fft(&fine, &mut transformed, "CPU type-1 fine-grid FFT")?;
                deconvolve(
                    &self.engine.axes,
                    &grid,
                    config.n_modes(),
                    &transformed,
                    output,
                    threads,
                );
            }
            NufftKind::Type2 => {
                predeconvolve(
                    &self.engine.axes,
                    &grid,
                    config.n_modes(),
                    self.input,
                    &mut fine,
                    threads,
                );
                fft(&fine, &mut transformed, "CPU type-2 fine-grid FFT")?;
                if let Some(sorted) = &sorted {
                    let mut values = zeroed("CPU type-2 sorted values", output.len() / 2)?;
                    interpolate::<T, P, D, RUN>(
                        &points,
                        bytemuck::cast_slice(&transformed),
                        sorted,
                        &mut values,
                        threads,
                    );
                    unsort(
                        sorted,
                        &values,
                        batch,
                        bytemuck::cast_slice_mut(output),
                        threads,
                    );
                }
            }
        }
        let mut scratch = self
            .engine
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if scratch.is_none() {
            *scratch = Some((fine, transformed));
        }
        Ok(())
    }
}

/// Preferred slab thickness for a last fine-grid axis of `length` cells.
///
/// The slabs fix every cell's summation order, so they depend on the grid
/// alone, never on the thread count.
fn slab_thickness<const D: usize>(length: usize) -> usize {
    if D == 1 {
        SLAB_CELLS_1D
    } else {
        (length / MIN_SLABS).clamp(1, SLAB_PLANES)
    }
}

/// Spreads through private grid copies when the grid is small for its
/// points, and slab by slab otherwise.
fn spread_points<T: Real, P: Coordinate, const D: usize, const RUN: usize>(
    points: &Points<'_, T, P, D>,
    strengths: &[[T; 2]],
    slabs: &Slabs,
    sorted: &SortedPoints,
    fine: &mut [[T; 2]],
    threads: usize,
) -> Result<()> {
    let cells = points.grid.cells() * points.batch;
    match dense_groups(slabs.count(), points.count, cells) {
        Some(groups) => {
            spread_dense::<T, P, D, RUN>(points, strengths, sorted, groups, fine, threads)
        }
        None => {
            spread::<T, P, D, RUN>(points, strengths, slabs, sorted, fine, threads);
            Ok(())
        }
    }
}

/// Copies point-major `values` in sorted order into transform-major
/// `output` in input order.
fn unsort<T: Real>(
    sorted: &SortedPoints,
    values: &[[T; 2]],
    batch: usize,
    output: &mut [[T; 2]],
    threads: usize,
) {
    let ranks = sorted.ranks(threads);
    let count = ranks.len();
    for_each_chunk_group(
        threads,
        output,
        1,
        MIN_VALUES_PER_THREAD,
        |first, output| {
            for (offset, value) in output.iter_mut().enumerate() {
                let flat = first + offset;
                let (transform, point) = (flat / count, flat % count);
                let rank = ranks[point].load(Ordering::Relaxed) as usize;
                *value = values[rank * batch + transform];
            }
        },
    );
}

/// `Df64` `hi + lo` word pairs as `f64` values.
fn join_df64(words: &[f32], threads: usize) -> Result<Vec<f64>> {
    let mut values = zeroed::<f64>("CPU Df64 input values", words.len() / 2)?;
    for_each_chunk_group(
        threads,
        &mut values,
        1,
        MIN_VALUES_PER_THREAD,
        |first, values| {
            let pairs = &words.as_chunks::<2>().0[first..first + values.len()];
            for (value, &[hi, lo]) in values.iter_mut().zip(pairs) {
                *value = f64::from(hi) + f64::from(lo);
            }
        },
    );
    Ok(values)
}

/// `f32` words as `f64` values.
fn widen(words: &[f32], threads: usize) -> Result<Vec<f64>> {
    let mut values = zeroed::<f64>("CPU widened values", words.len())?;
    for_each_chunk_group(
        threads,
        &mut values,
        1,
        MIN_VALUES_PER_THREAD,
        |first, values| {
            for (value, &word) in values.iter_mut().zip(&words[first..]) {
                *value = f64::from(word);
            }
        },
    );
    Ok(values)
}

/// Splits `f64` values into `Df64` `hi + lo` word pairs.
fn split_df64(values: &[f64], words: &mut [f32], threads: usize) {
    for_each_chunk_group(threads, words, 2, MIN_VALUES_PER_THREAD, |first, words| {
        for (pair, &value) in words
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(&values[first..])
        {
            let hi = value as f32;
            *pair = [hi, (value - f64::from(hi)) as f32];
        }
    });
}

fn checked_product(context: &'static str, factors: &[usize]) -> Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .ok_or(NufftError::LengthOverflow { context })
}

/// A zeroed vector of `len` elements, or a structured error when the
/// allocation fails. Large zeroed allocations cost nothing until written.
fn zeroed<T: bytemuck::Zeroable>(buffer: &'static str, len: usize) -> Result<Vec<T>> {
    bytemuck::allocation::try_zeroed_vec(len).map_err(|()| NufftError::HostAllocationFailed {
        buffer,
        elements: len,
    })
}

/// A vector of `len` copies of `value`, or a structured error when the
/// allocation fails.
fn filled<T: Clone>(buffer: &'static str, len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer,
            elements: len,
        })?;
    values.resize(len, value);
    Ok(values)
}
