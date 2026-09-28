//! Host-memory type-3 plans.

use std::f64::consts::PI;

use super::grid::FineGrid;
use super::points::{Slabs, SortedPoints};
use super::real::Real;
use super::spread::Points;
use super::workers::{for_each_chunk_group, map_ranges, ranges};
use super::{
    checked_product, join_df64, slab_thickness, split_df64, spread_points, widen, zeroed,
    CpuNufftPlan, Engine, TypedEngine,
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
        let precision = self.config().precision();
        let threads = self.threads();
        let (sources, targets) = match precision {
            FftPrecision::Df64 => {
                self.validate_word_lengths(sources.len(), targets.len())?;
                (join_df64(sources, threads)?, join_df64(targets, threads)?)
            }
            _ => (widen(sources, threads)?, widen(targets, threads)?),
        };
        match (&self.inner.engine, precision) {
            (Engine::F32(engine), _) => {
                self.validate(&sources, strengths.len(), &targets, output.len())?;
                self.run(
                    engine,
                    &sources,
                    bytemuck::cast_slice(strengths),
                    &targets,
                    bytemuck::cast_slice_mut(output),
                )
            }
            (Engine::F64(engine), FftPrecision::Df64) => {
                self.validate(&sources, strengths.len(), &targets, output.len())?;
                let strengths = join_df64(strengths, threads)?;
                let mut result = zeroed::<f64>("CPU Df64 type-3 output values", output.len() / 2)?;
                self.run(
                    engine,
                    &sources,
                    bytemuck::cast_slice(&strengths),
                    &targets,
                    bytemuck::cast_slice_mut(&mut result),
                )?;
                split_df64(&result, output, self.threads());
                Ok(())
            }
            (Engine::F64(_), requested) => Err(NufftError::PrecisionUnsupported {
                requested,
                stage: "CPU type-3 NUFFT execution",
                reason: "f64-plans-execute-with-execute_f64",
            }),
        }
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
        match (&self.inner.engine, self.config().precision()) {
            (Engine::F64(engine), FftPrecision::F64) => {
                self.validate(sources, strengths.len(), targets, output.len())?;
                self.run(
                    engine,
                    sources,
                    bytemuck::cast_slice(strengths),
                    targets,
                    bytemuck::cast_slice_mut(output),
                )
            }
            (_, requested) => Err(NufftError::PrecisionUnsupported {
                requested,
                stage: "CPU type-3 NUFFT execution",
                reason: "execute_f64-requires-an-f64-plan",
            }),
        }
    }

    fn scalar_words(&self) -> usize {
        match self.config().precision() {
            FftPrecision::Df64 => 2,
            FftPrecision::F32 | FftPrecision::F64 => 1,
        }
    }

    fn coordinate_words(&self, context: &'static str, count: usize) -> Result<usize> {
        checked_product(
            context,
            &[count, self.config().dimensions(), self.scalar_words()],
        )
    }

    fn complex_words(&self, context: &'static str, count: usize) -> Result<usize> {
        checked_product(
            context,
            &[count, self.config().batch(), 2 * self.scalar_words()],
        )
    }

    /// Checks that `Df64` coordinate slices hold whole word pairs.
    fn validate_word_lengths(&self, sources: usize, targets: usize) -> Result<()> {
        let dimensions = self.config().dimensions();
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

    /// Checks the coordinates against the plan's intervals and the slice
    /// lengths, given in storage words.
    fn validate(
        &self,
        sources: &[f64],
        strengths: usize,
        targets: &[f64],
        output: usize,
    ) -> Result<()> {
        let config = self.config();
        let threads = self.threads();
        let source_count = check_points("source", sources, config.source_bounds(), threads)?;
        let target_count = check_points("target", targets, config.target_bounds(), threads)?;
        for (input, expected, actual) in [
            (
                "CPU type-3 strengths",
                self.required_strengths_len(source_count)?,
                strengths,
            ),
            (
                "CPU type-3 output values",
                self.required_output_len(target_count)?,
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
        for (set, count) in [("source", source_count), ("target", target_count)] {
            if u32::try_from(count).is_err() {
                return Err(NufftError::LengthOverflow {
                    context: if set == "source" {
                        "CPU type-3 source count"
                    } else {
                        "CPU type-3 target count"
                    },
                });
            }
        }
        Ok(())
    }

    fn run<T: Real>(
        &self,
        engine: &TypedEngine<T>,
        sources: &[f64],
        strengths: &[[T; 2]],
        targets: &[f64],
        output: &mut [[T; 2]],
    ) -> Result<()> {
        match self.config().dimensions() {
            1 => self.run_rank::<T, 1>(engine, sources, strengths, targets, output),
            2 => self.run_rank::<T, 2>(engine, sources, strengths, targets, output),
            3 => self.run_rank::<T, 3>(engine, sources, strengths, targets, output),
            4 => self.run_rank::<T, 4>(engine, sources, strengths, targets, output),
            5 => self.run_rank::<T, 5>(engine, sources, strengths, targets, output),
            6 => self.run_rank::<T, 6>(engine, sources, strengths, targets, output),
            7 => self.run_rank::<T, 7>(engine, sources, strengths, targets, output),
            8 => self.run_rank::<T, 8>(engine, sources, strengths, targets, output),
            dimensions => unreachable!("validated dimension count {dimensions}"),
        }
    }

    fn run_rank<T: Real, const D: usize>(
        &self,
        engine: &TypedEngine<T>,
        sources: &[f64],
        strengths: &[[T; 2]],
        targets: &[f64],
        output: &mut [[T; 2]],
    ) -> Result<()> {
        match self.kernel().width() {
            0..=4 => self.run_with::<T, D, 4>(engine, sources, strengths, targets, output),
            5..=8 => self.run_with::<T, D, 8>(engine, sources, strengths, targets, output),
            _ => self.run_with::<T, D, 16>(engine, sources, strengths, targets, output),
        }
    }

    fn run_with<T: Real, const D: usize, const RUN: usize>(
        &self,
        engine: &TypedEngine<T>,
        sources: &[f64],
        strengths: &[[T; 2]],
        targets: &[f64],
        output: &mut [[T; 2]],
    ) -> Result<()> {
        let config = self.config();
        let batch = config.batch();
        let threads = self.threads();
        let sign = config.sign().multiplier();
        let axes = &self.axes;
        let source_count = sources.len() / D;
        let target_count = targets.len() / D;
        if target_count == 0 {
            return Ok(());
        }

        // Pre-phase the strengths by exp(isign i D.x) and spread them at the
        // rescaled sources (x - C) / gamma onto the outer grid.
        let grid = FineGrid::<D>::new(&self.outer_grid_shape(), self.kernel);
        let mut outer = zeroed::<[T; 2]>("CPU type-3 outer grid", grid.cells() * batch)?;
        if source_count > 0 {
            let mut rescaled = zeroed::<f64>("CPU type-3 rescaled sources", sources.len())?;
            for_each_chunk_group(
                threads,
                &mut rescaled,
                D,
                MIN_VALUES_PER_THREAD,
                |first, chunk| {
                    let sources = &sources[first * D..first * D + chunk.len()];
                    for (point, source) in chunk.chunks_exact_mut(D).zip(sources.chunks_exact(D)) {
                        for ((value, &coordinate), axis) in point.iter_mut().zip(source).zip(axes) {
                            *value = axis.rescale_source(coordinate);
                        }
                    }
                },
            );
            let mut phased = zeroed::<[T; 2]>("CPU type-3 phased strengths", strengths.len())?;
            for_each_chunk_group(
                threads,
                &mut phased,
                1,
                MIN_VALUES_PER_THREAD,
                |first, chunk| {
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        let flat = first + offset;
                        let source =
                            &sources[(flat % source_count) * D..(flat % source_count + 1) * D];
                        let angle = source
                            .iter()
                            .zip(axes)
                            .map(|(&x, axis)| axis.target_center() * x)
                            .sum::<f64>();
                        *value = rotate(strengths[flat], sign * angle, 1.0);
                    }
                },
            );
            let slabs = Slabs::new(
                grid.shape[D - 1],
                self.kernel().width(),
                slab_thickness::<D>(grid.shape[D - 1]),
            )?;
            let sorted = SortedPoints::new(&grid, &slabs, &rescaled, threads)?;
            let points = Points {
                grid: &grid,
                weights: &engine.weights,
                coordinates: &rescaled,
                count: source_count,
                batch,
            };
            spread_points::<T, f64, D, RUN>(
                &points, &phased, &slabs, &sorted, &mut outer, threads,
            )?;
        }

        // Evaluate the outer grid at the rescaled targets h*gamma*(s - D).
        let mut rescaled = zeroed::<f64>("CPU type-3 rescaled targets", targets.len())?;
        for_each_chunk_group(
            threads,
            &mut rescaled,
            D,
            MIN_VALUES_PER_THREAD,
            |first, chunk| {
                let targets = &targets[first * D..first * D + chunk.len()];
                for (point, target) in chunk.chunks_exact_mut(D).zip(targets.chunks_exact(D)) {
                    for ((value, &frequency), axis) in point.iter_mut().zip(target).zip(axes) {
                        *value = axis.rescale_target(frequency);
                    }
                }
            },
        );
        self.inner.run(
            engine,
            &rescaled,
            bytemuck::cast_slice(&outer),
            bytemuck::cast_slice_mut(output),
            target_count,
        )?;

        // Post-phase by exp(isign i C.(s - D)) and divide by the kernel's
        // Fourier transform at the rescaled target.
        let mut factors = zeroed::<[f64; 3]>("CPU type-3 target corrections", target_count)?;
        for_each_chunk_group(
            threads,
            &mut factors,
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
        for_each_chunk_group(threads, output, 1, MIN_VALUES_PER_THREAD, |first, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                let [cos, sin, scale] = factors[(first + offset) % target_count];
                *value = rotate_by(*value, cos, sin, scale);
            }
        });
        Ok(())
    }
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

/// `value * exp(i * angle) * scale`, computed in `f64`.
fn rotate<T: Real>(value: [T; 2], angle: f64, scale: f64) -> [T; 2] {
    let (sin, cos) = angle.sin_cos();
    rotate_by(value, cos, sin, scale)
}

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
