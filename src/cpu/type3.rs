//! Host-memory type-3 plans.

use std::f64::consts::PI;

use super::grid::FineGrid;
use super::real::Real;
use super::spread::Points;
use super::workers::{for_each_chunk_group, map_ranges, ranges};
use super::{
    checked_product, grid_slabs, join_df64, split_df64, spread_points, widen, zeroed, Coordinates,
    CpuNufftPlan, CpuNufftPoints, Engine, TypedEngine, Usage,
};
use crate::config::{ModeOrder, NufftConfig};
use crate::error::{NufftError, Result};
use crate::kernel::{gauss_legendre, EsHornerTable, EsKernel};
use crate::type3::{NufftInterval, NufftType3AxisMetadata, NufftType3Config, NufftType3Plan};
use wgpu_fft::FftPrecision;

/// Values per thread below which the elementwise passes stay on one thread.
const MIN_VALUES_PER_THREAD: usize = 1 << 13;
/// Samples of the Chebyshev fit of the kernel transform.
const CHEBYSHEV_SAMPLES: usize = 64;

/// Host-memory type-3 NUFFT plan, from nonuniform points to nonuniform
/// frequencies, with the configuration semantics and buffer layouts of
/// [`NufftType3Plan`].
///
/// It follows the GPU composition: sources are pre-phased and rescaled,
/// spread onto an outer grid as in [`CpuNufftPlan::type1`], evaluated at the
/// rescaled targets by an inner [`CpuNufftPlan::type2`], and corrected by the
/// kernel's Fourier transform and a post-phase. Results are bitwise
/// repeatable for any thread count, and every source and target coordinate
/// is checked against the plan's intervals.
///
/// [`Self::prepare_points`] does the point work once: it checks, rescales,
/// and sorts a source and a target set and computes their phases. Then
/// [`Self::execute_prepared`] transforms any number of strength vectors
/// between them, with results bitwise identical to [`Self::execute`].
///
/// ```
/// use wgpu_nufft::{CpuNufftType3Plan, NufftInterval, NufftType3Config};
///
/// let config = NufftType3Config::new(
///     vec![NufftInterval::new(-1.0, 1.0)],
///     vec![NufftInterval::new(-40.0, 40.0)],
///     1.0e-6,
/// );
/// let plan = CpuNufftType3Plan::new(config)?;
/// let sources = [-0.5f32, 0.25, 0.75];
/// let strengths = [1.0f32, 0.0, 0.5, 0.5, -1.0, 0.25];
/// let targets = [-30.0f32, 0.0, 12.5, 39.0];
/// let mut values = vec![0.0f32; plan.required_output_len(targets.len())?];
/// plan.execute(&sources, &strengths, &targets, &mut values)?;
///
/// // The same transform, with the point work done once.
/// let points = plan.prepare_points(&sources, &targets)?;
/// plan.execute_prepared(&points, &strengths, &mut values)?;
/// # Ok::<(), wgpu_nufft::NufftError>(())
/// ```
#[derive(Debug)]
pub struct CpuNufftType3Plan {
    config: NufftType3Config,
    kernel: EsKernel,
    axes: Vec<NufftType3AxisMetadata>,
    /// The type-2 transform that evaluates the outer grid at the rescaled
    /// targets; it shares the kernel and the thread count.
    inner: CpuNufftPlan,
    correction: KernelTransform,
}

/// Source and target sets of a [`CpuNufftType3Plan`], checked, rescaled,
/// and sorted once for repeated executions; see
/// [`CpuNufftType3Plan::prepare_points`].
///
/// It serves every type-3 plan with the configuration of the plan that
/// prepared it, whatever their batch sizes, and executions on several
/// threads can share it.
pub struct CpuNufftType3Points {
    /// The preparing plan's configuration, with a batch of one.
    config: NufftType3Config,
    source_count: usize,
    target_count: usize,
    /// Rescaled sources, sorted on the outer grid.
    sources: CpuNufftPoints,
    /// `exp(isign i D.x)` of every source, as `[cos, sin]`.
    source_phases: Vec<[f64; 2]>,
    /// Rescaled targets, prepared for the inner type-2 plan.
    targets: CpuNufftPoints,
    /// The post-phase and kernel correction of every target, as
    /// `[cos, sin, scale]`.
    target_factors: Vec<[f64; 3]>,
}

impl CpuNufftType3Points {
    pub fn source_count(&self) -> usize {
        self.source_count
    }

    pub fn target_count(&self) -> usize {
        self.target_count
    }

    pub fn dimensions(&self) -> usize {
        self.config.dimensions()
    }
}

impl std::fmt::Debug for CpuNufftType3Points {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuNufftType3Points")
            .field("config", &self.config)
            .field("source_count", &self.source_count)
            .field("target_count", &self.target_count)
            .finish_non_exhaustive()
    }
}

impl CpuNufftType3Plan {
    pub fn new(config: NufftType3Config) -> Result<Self> {
        // The host metadata of a GPU-less plan: kernel, rescaling, outer grid.
        let metadata = NufftType3Plan::new(config)?;
        let (config, kernel) = (metadata.config().clone(), metadata.kernel());
        let axes = metadata.axes().to_vec();
        let inner = CpuNufftPlan::type2(
            NufftConfig::new(metadata.outer_grid_shape(), config.eps())
                .with_sign(config.sign())
                .with_mode_order(ModeOrder::Centered)
                .with_sigma(config.sigma())
                .with_batch(config.batch())
                .with_precision(config.precision()),
        )?;
        // The largest rescaled target frequency the correction must cover.
        let limit = axes
            .iter()
            .zip(config.target_bounds())
            .flat_map(|(axis, bounds)| {
                [
                    axis.rescale_target(bounds.lower()).abs(),
                    axis.rescale_target(bounds.upper()).abs(),
                ]
            })
            .fold(0.0, f64::max);
        let correction = KernelTransform::new(kernel, &kernel.horner_table(), limit);
        Ok(Self {
            config,
            kernel,
            axes,
            inner,
            correction,
        })
    }

    /// Sets the number of threads; see [`CpuNufftPlan::with_threads`].
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.inner = self.inner.with_threads(threads);
        self
    }

    pub fn threads(&self) -> usize {
        self.inner.threads()
    }

    pub fn config(&self) -> &NufftType3Config {
        &self.config
    }

    pub fn kernel(&self) -> EsKernel {
        self.kernel
    }

    /// Rescaling metadata of every axis.
    pub fn axes(&self) -> &[NufftType3AxisMetadata] {
        &self.axes
    }

    pub fn outer_grid_shape(&self) -> Vec<usize> {
        self.axes
            .iter()
            .map(|axis| axis.outer_grid_length())
            .collect()
    }

    /// Words of the source slice for `source_count` points.
    pub fn required_source_points_len(&self, source_count: usize) -> Result<usize> {
        self.coordinate_words("CPU type-3 source point words", source_count)
    }

    /// Words of the target slice for `target_count` frequencies.
    pub fn required_target_points_len(&self, target_count: usize) -> Result<usize> {
        self.coordinate_words("CPU type-3 target point words", target_count)
    }

    /// Words of the strength slice, for every transform of the batch.
    pub fn required_strengths_len(&self, source_count: usize) -> Result<usize> {
        self.complex_words("CPU type-3 strength words", source_count)
    }

    /// Words of the output slice, for every transform of the batch.
    pub fn required_output_len(&self, target_count: usize) -> Result<usize> {
        self.complex_words("CPU type-3 output words", target_count)
    }

    /// Executes an `F32` or `Df64` plan on `f32` words. The source and target
    /// counts follow from `sources` and `targets`.
    pub fn execute(
        &self,
        sources: &[f32],
        strengths: &[f32],
        targets: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let points = self.prepare_words(
            sources,
            targets,
            "CPU type-3 NUFFT execution",
            "f64-plans-execute-with-execute_f64",
        )?;
        self.execute_prepared(&points, strengths, output)
    }

    /// Executes an `F64` plan on `f64` words. The source and target counts
    /// follow from `sources` and `targets`.
    pub fn execute_f64(
        &self,
        sources: &[f64],
        strengths: &[f64],
        targets: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        self.require_f64(
            "CPU type-3 NUFFT execution",
            "execute_f64-requires-an-f64-plan",
        )?;
        let points = self.prepare(sources, targets)?;
        self.execute_prepared_f64(&points, strengths, output)
    }

    /// Checks, rescales, and sorts the source and target points of an `F32`
    /// or `Df64` plan once, for any number of executions with
    /// [`Self::execute_prepared`]. `sources` and `targets` hold the same
    /// words as in [`Self::execute`].
    pub fn prepare_points(&self, sources: &[f32], targets: &[f32]) -> Result<CpuNufftType3Points> {
        self.prepare_words(
            sources,
            targets,
            "CPU type-3 point preparation",
            "f64-plans-prepare-points-with-prepare_points_f64",
        )
    }

    /// Prepares the `f64` source and target points of an `F64` plan once;
    /// see [`Self::prepare_points`].
    pub fn prepare_points_f64(
        &self,
        sources: &[f64],
        targets: &[f64],
    ) -> Result<CpuNufftType3Points> {
        self.require_f64(
            "CPU type-3 point preparation",
            "prepare_points_f64-requires-an-f64-plan",
        )?;
        self.prepare(sources, targets)
    }

    /// Executes an `F32` or `Df64` plan on points from
    /// [`Self::prepare_points`]. `strengths` and `output` are as in
    /// [`Self::execute`].
    pub fn execute_prepared(
        &self,
        points: &CpuNufftType3Points,
        strengths: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.check_prepared(points)?;
        let threads = self.threads();
        match (&self.inner.engine, self.config.precision()) {
            (Engine::F32(engine), _) => {
                self.validate_values(points, strengths.len(), output.len())?;
                self.run(
                    engine,
                    points,
                    bytemuck::cast_slice(strengths),
                    bytemuck::cast_slice_mut(output),
                )
            }
            (Engine::F64(engine), FftPrecision::Df64) => {
                self.validate_values(points, strengths.len(), output.len())?;
                let strengths = join_df64(strengths, threads)?;
                let mut result = zeroed::<f64>("CPU Df64 type-3 output values", output.len() / 2)?;
                self.run(
                    engine,
                    points,
                    bytemuck::cast_slice(&strengths),
                    bytemuck::cast_slice_mut(&mut result),
                )?;
                split_df64(&result, output, threads);
                Ok(())
            }
            (Engine::F64(_), requested) => Err(NufftError::PrecisionUnsupported {
                requested,
                stage: "CPU type-3 NUFFT execution",
                reason: "f64-plans-execute-with-execute_prepared_f64",
            }),
        }
    }

    /// Executes an `F64` plan on points from [`Self::prepare_points_f64`].
    /// `strengths` and `output` are as in [`Self::execute_f64`].
    pub fn execute_prepared_f64(
        &self,
        points: &CpuNufftType3Points,
        strengths: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        self.check_prepared(points)?;
        match (&self.inner.engine, self.config.precision()) {
            (Engine::F64(engine), FftPrecision::F64) => {
                self.validate_values(points, strengths.len(), output.len())?;
                self.run(
                    engine,
                    points,
                    bytemuck::cast_slice(strengths),
                    bytemuck::cast_slice_mut(output),
                )
            }
            (_, requested) => Err(NufftError::PrecisionUnsupported {
                requested,
                stage: "CPU type-3 NUFFT execution",
                reason: "execute_prepared_f64-requires-an-f64-plan",
            }),
        }
    }

    fn scalar_words(&self) -> usize {
        match self.config.precision() {
            FftPrecision::Df64 => 2,
            FftPrecision::F32 | FftPrecision::F64 => 1,
        }
    }

    fn coordinate_words(&self, context: &'static str, count: usize) -> Result<usize> {
        checked_product(
            context,
            &[count, self.config.dimensions(), self.scalar_words()],
        )
    }

    fn complex_words(&self, context: &'static str, count: usize) -> Result<usize> {
        checked_product(
            context,
            &[count, self.config.batch(), 2 * self.scalar_words()],
        )
    }

    fn require_f64(&self, stage: &'static str, reason: &'static str) -> Result<()> {
        match self.config.precision() {
            FftPrecision::F64 => Ok(()),
            requested => Err(NufftError::PrecisionUnsupported {
                requested,
                stage,
                reason,
            }),
        }
    }

    /// Prepares the `f32` words of an `F32` or `Df64` plan.
    fn prepare_words(
        &self,
        sources: &[f32],
        targets: &[f32],
        stage: &'static str,
        reason: &'static str,
    ) -> Result<CpuNufftType3Points> {
        let threads = self.threads();
        let (sources, targets) = match self.config.precision() {
            FftPrecision::F32 => (widen(sources, threads)?, widen(targets, threads)?),
            FftPrecision::Df64 => {
                self.validate_word_lengths(sources.len(), targets.len())?;
                (join_df64(sources, threads)?, join_df64(targets, threads)?)
            }
            requested => {
                return Err(NufftError::PrecisionUnsupported {
                    requested,
                    stage,
                    reason,
                })
            }
        };
        self.prepare(&sources, &targets)
    }

    /// Checks that `Df64` coordinate slices hold whole word pairs.
    fn validate_word_lengths(&self, sources: usize, targets: usize) -> Result<()> {
        let dimensions = self.config.dimensions();
        for (set, words) in [("source", sources), ("target", targets)] {
            if !words.is_multiple_of(2 * dimensions) {
                return Err(NufftError::Type3CoordinateLength {
                    set,
                    actual: words / 2,
                    dimensions,
                });
            }
        }
        Ok(())
    }

    /// Checks that `points` came from a plan with this plan's configuration.
    fn check_prepared(&self, points: &CpuNufftType3Points) -> Result<()> {
        if points.config != self.config.clone().with_batch(1) {
            return Err(NufftError::PointSetMismatch {
                reason: "it was prepared by a type-3 plan with another configuration",
            });
        }
        Ok(())
    }

    /// Checks the strength and output lengths, given in storage words.
    fn validate_values(
        &self,
        points: &CpuNufftType3Points,
        strengths: usize,
        output: usize,
    ) -> Result<()> {
        for (input, expected, actual) in [
            (
                "CPU type-3 strengths",
                self.required_strengths_len(points.source_count)?,
                strengths,
            ),
            (
                "CPU type-3 output values",
                self.required_output_len(points.target_count)?,
                output,
            ),
        ] {
            if actual != expected {
                return Err(NufftError::InputLength {
                    input,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Checks the coordinates against the plan's intervals, then rescales,
    /// sorts, and phases them.
    fn prepare(&self, sources: &[f64], targets: &[f64]) -> Result<CpuNufftType3Points> {
        let config = &self.config;
        let threads = self.threads();
        let source_count = check_points("source", sources, config.source_bounds(), threads)?;
        let target_count = check_points("target", targets, config.target_bounds(), threads)?;
        for (context, count) in [
            ("CPU type-3 source count", source_count),
            ("CPU type-3 target count", target_count),
        ] {
            if u32::try_from(count).is_err() {
                return Err(NufftError::LengthOverflow { context });
            }
        }
        match config.dimensions() {
            1 => self.prepare_rank::<1>(sources, targets),
            2 => self.prepare_rank::<2>(sources, targets),
            3 => self.prepare_rank::<3>(sources, targets),
            4 => self.prepare_rank::<4>(sources, targets),
            5 => self.prepare_rank::<5>(sources, targets),
            6 => self.prepare_rank::<6>(sources, targets),
            7 => self.prepare_rank::<7>(sources, targets),
            8 => self.prepare_rank::<8>(sources, targets),
            dimensions => unreachable!("validated dimension count {dimensions}"),
        }
    }

    fn prepare_rank<const D: usize>(
        &self,
        sources: &[f64],
        targets: &[f64],
    ) -> Result<CpuNufftType3Points> {
        let threads = self.threads();
        let sign = self.config.sign().multiplier();
        let axes = &self.axes;
        let (source_count, target_count) = (sources.len() / D, targets.len() / D);

        // Sources rescale to (x - C) / gamma, sorted on the outer grid, and
        // pre-phase by exp(isign i D.x).
        let rescaled = rescale::<D>(
            "CPU type-3 rescaled sources",
            sources,
            axes,
            threads,
            |axis, coordinate| axis.rescale_source(coordinate),
        )?;
        let mut source_phases = zeroed::<[f64; 2]>("CPU type-3 source phases", source_count)?;
        for_each_chunk_group(
            threads,
            &mut source_phases,
            1,
            MIN_VALUES_PER_THREAD,
            |first, chunk| {
                for (offset, phase) in chunk.iter_mut().enumerate() {
                    let source = &sources[(first + offset) * D..(first + offset + 1) * D];
                    let angle = source
                        .iter()
                        .zip(axes)
                        .map(|(&x, axis)| axis.target_center() * x)
                        .sum::<f64>();
                    let (sin, cos) = (sign * angle).sin_cos();
                    *phase = [cos, sin];
                }
            },
        );
        let sorted_sources = CpuNufftPoints::new(
            self.config.precision(),
            &self.outer_grid_shape(),
            self.kernel,
            Coordinates::F64(rescaled),
            source_count,
            threads,
            Usage::Spread,
        )?;

        // Targets rescale to h gamma (s - D) for the inner type-2 plan, and
        // post-phase by exp(isign i C.(s - D)) over the kernel's Fourier
        // transform at the rescaled target.
        let rescaled = rescale::<D>(
            "CPU type-3 rescaled targets",
            targets,
            axes,
            threads,
            |axis, frequency| axis.rescale_target(frequency),
        )?;
        let mut target_factors = zeroed::<[f64; 3]>("CPU type-3 target corrections", target_count)?;
        for_each_chunk_group(
            threads,
            &mut target_factors,
            1,
            MIN_VALUES_PER_THREAD,
            |first, chunk| {
                for (offset, factor) in chunk.iter_mut().enumerate() {
                    let target = first + offset;
                    let frequencies = &targets[target * D..(target + 1) * D];
                    let rescaled = &rescaled[target * D..(target + 1) * D];
                    let mut angle = 0.0;
                    let mut transform = 1.0;
                    for ((&frequency, &rescaled), axis) in
                        frequencies.iter().zip(rescaled).zip(axes)
                    {
                        angle += axis.source_center() * (frequency - axis.target_center());
                        transform *= self.correction.evaluate(rescaled);
                    }
                    let (sin, cos) = (sign * angle).sin_cos();
                    *factor = [cos, sin, 1.0 / transform];
                }
            },
        );
        let inner = &self.inner;
        let prepared_targets = CpuNufftPoints::new(
            inner.config.precision(),
            &inner.fine_grid_shape,
            inner.kernel,
            Coordinates::F64(rescaled),
            target_count,
            threads,
            if D == 1 {
                Usage::InputOrder
            } else {
                Usage::Any
            },
        )?;
        Ok(CpuNufftType3Points {
            config: self.config.clone().with_batch(1),
            source_count,
            target_count,
            sources: sorted_sources,
            source_phases,
            targets: prepared_targets,
            target_factors,
        })
    }

    fn run<T: Real>(
        &self,
        engine: &TypedEngine<T>,
        points: &CpuNufftType3Points,
        strengths: &[[T; 2]],
        output: &mut [[T; 2]],
    ) -> Result<()> {
        match self.config.dimensions() {
            1 => self.run_rank::<T, 1>(engine, points, strengths, output),
            2 => self.run_rank::<T, 2>(engine, points, strengths, output),
            3 => self.run_rank::<T, 3>(engine, points, strengths, output),
            4 => self.run_rank::<T, 4>(engine, points, strengths, output),
            5 => self.run_rank::<T, 5>(engine, points, strengths, output),
            6 => self.run_rank::<T, 6>(engine, points, strengths, output),
            7 => self.run_rank::<T, 7>(engine, points, strengths, output),
            8 => self.run_rank::<T, 8>(engine, points, strengths, output),
            dimensions => unreachable!("validated dimension count {dimensions}"),
        }
    }

    fn run_rank<T: Real, const D: usize>(
        &self,
        engine: &TypedEngine<T>,
        points: &CpuNufftType3Points,
        strengths: &[[T; 2]],
        output: &mut [[T; 2]],
    ) -> Result<()> {
        match self.kernel.width() {
            0..=4 => self.run_with::<T, D, 4>(engine, points, strengths, output),
            5..=8 => self.run_with::<T, D, 8>(engine, points, strengths, output),
            _ => self.run_with::<T, D, 16>(engine, points, strengths, output),
        }
    }

    fn run_with<T: Real, const D: usize, const RUN: usize>(
        &self,
        engine: &TypedEngine<T>,
        points: &CpuNufftType3Points,
        strengths: &[[T; 2]],
        output: &mut [[T; 2]],
    ) -> Result<()> {
        let batch = self.config.batch();
        let threads = self.threads();
        let (source_count, target_count) = (points.source_count, points.target_count);
        if target_count == 0 {
            return Ok(());
        }

        // Pre-phase the strengths and spread them at the rescaled sources
        // onto the outer grid.
        let grid = FineGrid::<D>::new(&self.outer_grid_shape(), self.kernel);
        let mut outer = zeroed::<[T; 2]>("CPU type-3 outer grid", grid.cells() * batch)?;
        if let (Some(sorted), Coordinates::F64(coordinates)) =
            (&points.sources.sorted, &points.sources.coordinates)
        {
            let mut phased = zeroed::<[T; 2]>("CPU type-3 phased strengths", strengths.len())?;
            for_each_chunk_group(
                threads,
                &mut phased,
                1,
                MIN_VALUES_PER_THREAD,
                |first, chunk| {
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        let flat = first + offset;
                        let [cos, sin] = points.source_phases[flat % source_count];
                        *value = rotate_by(strengths[flat], cos, sin, 1.0);
                    }
                },
            );
            let slabs = grid_slabs(&grid, self.kernel.width())?;
            let sources = Points {
                grid: &grid,
                weights: &engine.weights,
                coordinates: coordinates.as_slice(),
                count: source_count,
                batch,
            };
            spread_points::<T, f64, D, RUN>(
                &sources, &phased, &slabs, sorted, &mut outer, threads,
            )?;
        }

        // Evaluate the outer grid at the rescaled targets, then post-phase
        // and correct.
        self.inner.run_prepared(
            engine,
            &points.targets,
            bytemuck::cast_slice(&outer),
            bytemuck::cast_slice_mut(output),
        )?;
        for_each_chunk_group(threads, output, 1, MIN_VALUES_PER_THREAD, |first, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                let [cos, sin, scale] = points.target_factors[(first + offset) % target_count];
                *value = rotate_by(*value, cos, sin, scale);
            }
        });
        Ok(())
    }
}

/// `rescale(axis, coordinate)` for every coordinate of point-major `points`.
fn rescale<const D: usize>(
    buffer: &'static str,
    points: &[f64],
    axes: &[NufftType3AxisMetadata],
    threads: usize,
    rescale: impl Fn(NufftType3AxisMetadata, f64) -> f64 + Sync,
) -> Result<Vec<f64>> {
    let mut values = zeroed::<f64>(buffer, points.len())?;
    for_each_chunk_group(
        threads,
        &mut values,
        D,
        MIN_VALUES_PER_THREAD,
        |first, chunk| {
            let points = &points[first * D..first * D + chunk.len()];
            for (value, (&coordinate, &axis)) in
                chunk.iter_mut().zip(points.iter().zip(axes.iter().cycle()))
            {
                *value = rescale(axis, coordinate);
            }
        },
    );
    Ok(values)
}

/// Checks every coordinate of a point set against its intervals, in
/// parallel, reporting the first failure, and returns the point count.
fn check_points(
    set: &'static str,
    coordinates: &[f64],
    bounds: &[NufftInterval],
    threads: usize,
) -> Result<usize> {
    let dimensions = bounds.len();
    if !coordinates.len().is_multiple_of(dimensions) {
        return Err(NufftError::Type3CoordinateLength {
            set,
            actual: coordinates.len(),
            dimensions,
        });
    }
    let count = coordinates.len() / dimensions;
    map_ranges(
        &ranges(count, threads, MIN_VALUES_PER_THREAD),
        |_, range| {
            for point in range {
                let point_coordinates = &coordinates[point * dimensions..(point + 1) * dimensions];
                for (axis, (&value, interval)) in point_coordinates.iter().zip(bounds).enumerate() {
                    if !value.is_finite() {
                        return Err(NufftError::Type3NonFiniteCoordinate {
                            set,
                            point,
                            axis,
                            value,
                        });
                    }
                    if !interval.contains(value) {
                        return Err(NufftError::Type3CoordinateOutOfBounds {
                            set,
                            point,
                            axis,
                            value,
                            lower: interval.lower(),
                            upper: interval.upper(),
                        });
                    }
                }
            }
            Ok(())
        },
    )
    .into_iter()
    .collect::<Result<Vec<()>>>()?;
    Ok(count)
}

/// `value * (cos + i sin) * scale`, computed in `f64`.
fn rotate_by<T: Real>(value: [T; 2], cos: f64, sin: f64, scale: f64) -> [T; 2] {
    let (re, im) = (value[0].to_f64(), value[1].to_f64());
    [
        T::from_f64((re * cos - im * sin) * scale),
        T::from_f64((re * sin + im * cos) * scale),
    ]
}

/// The continuous Fourier transform of the kernel the plans spread with.
///
/// It is the Gauss-Legendre quadrature over the half-support of the GPU
/// path, fitted once by a Chebyshev series on `[0, limit]`, which holds every
/// rescaled target; the series replaces a cosine per quadrature node with a
/// few multiply-adds.
#[derive(Debug)]
struct KernelTransform {
    distances: Vec<f64>,
    weights: Vec<f64>,
    /// Chebyshev coefficients of the transform on `[0, limit]`.
    coefficients: Vec<f64>,
    limit: f64,
}

impl KernelTransform {
    fn new(kernel: EsKernel, table: &EsHornerTable, limit: f64) -> Self {
        let positive = kernel.width() + 2;
        let (nodes, weights) = gauss_legendre(2 * positive);
        let half_width = kernel.half_width();
        let distances = nodes[positive..]
            .iter()
            .map(|&node| half_width * node)
            .collect::<Vec<_>>();
        let weights = distances
            .iter()
            .zip(&weights[positive..])
            .map(|(&distance, &weight)| 2.0 * half_width * weight * table.evaluate(distance))
            .collect();
        let mut transform = Self {
            distances,
            weights,
            coefficients: Vec::new(),
            limit: if limit > 0.0 { limit } else { PI },
        };
        let count = CHEBYSHEV_SAMPLES;
        let angle = |j: usize, k: usize| PI * j as f64 * (k as f64 + 0.5) / count as f64;
        let samples = (0..count)
            .map(|k| transform.quadrature(0.5 * transform.limit * (1.0 + angle(1, k).cos())))
            .collect::<Vec<_>>();
        transform.coefficients = (0..count)
            .map(|j| {
                let sum = samples
                    .iter()
                    .enumerate()
                    .map(|(k, &sample)| sample * angle(j, k).cos())
                    .sum::<f64>();
                sum * if j == 0 { 1.0 } else { 2.0 } / count as f64
            })
            .collect();
        transform
    }

    fn quadrature(&self, frequency: f64) -> f64 {
        self.distances
            .iter()
            .zip(&self.weights)
            .map(|(&distance, &weight)| weight * (frequency * distance).cos())
            .sum()
    }

    fn evaluate(&self, frequency: f64) -> f64 {
        let x = frequency.abs();
        if x > self.limit {
            return self.quadrature(x);
        }
        // Clenshaw's recurrence for the Chebyshev series at t in [-1, 1].
        let t = 2.0 * x / self.limit - 1.0;
        let (mut next, mut after) = (0.0, 0.0);
        for &coefficient in self.coefficients[1..].iter().rev() {
            (next, after) = (coefficient + 2.0 * t * next - after, next);
        }
        self.coefficients[0] + t * next - after
    }
}
