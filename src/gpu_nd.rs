//! Rank-generic (runtime-dimensional) type-2 GPU execution, and helpers
//! shared by the rank-generic paths.
//!
//! Dimension appears only as generated WGSL, so one implementation serves
//! every rank the plan accepts, in all three precisions. The modes are
//! scattered, deconvolved, onto the cleared fine grid, transformed by an
//! oversampled C2C FFT, and interpolated at points binned once per point set
//! ([`NdPointBins`]): lanes of one point walk contiguous rows along axis
//! zero, consecutive points share cached rows, and the lane sums are added in
//! lane order, so every output is a fixed-order sum.

use std::fmt::Write as _;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
use wgpu_fft::math::DoubleFloat;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{ModeOrder, NufftConfig, NufftSign};
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_nd_bins::{NdBinOrder, NdPointBins};
use crate::gpu_nd_wgsl::{position_wgsl, weight_wgsl, NdWgsl, WRAP_INDEX_WGSL};
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::{GpuProfileQueryWriter, NufftGpuProfileLayout};
use crate::gpu_recorder::GpuRecorder;
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
/// Invocations of an interpolation workgroup: `LANES` per point.
const INTERPOLATION_WORKGROUP_SIZE: u32 = 256;
/// Cells per bin along every axis for interpolation order.
const TYPE2_BIN_SIDE: usize = 4;
/// Bins above which interpolation bins grow, keeping the scan small.
const TYPE2_MAX_BINS: usize = 1 << 24;

pub(crate) struct Type2GpuPlanNd {
    fft: FftPlan,
    amplitudes: wgpu::Buffer,
    fine_input: wgpu::Buffer,
    fine_output: wgpu::Buffer,
    bins: NdPointBins,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    interpolation_pipeline: wgpu::ComputePipeline,
    interpolation_layout: wgpu::BindGroupLayout,
    points_per_workgroup: usize,
    max_workgroups_per_dimension: u32,
    batch_capacity: usize,
    mode_count: usize,
    dimensions: usize,
    precision: FftPrecision,
    max_storage_binding_bytes: u64,
}

impl Type2GpuPlanNd {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: &[usize],
        kernel_fourier_coefficients: &[Vec<f64>],
    ) -> Result<Self> {
        let dimensions = config.dimensions();
        debug_assert_eq!(fine_shape.len(), dimensions);
        debug_assert_eq!(kernel_fourier_coefficients.len(), dimensions);
        let precision = config.precision();
        // F32 and Df64 fold integer support indices through an f32 high word;
        // native f64 keeps adjacent cells distinct across the signed i32 range.
        let maximum_signed_length = match precision {
            FftPrecision::F64 => i32::MAX as usize - kernel.width(),
            FftPrecision::F32 | FftPrecision::Df64 => (1usize << f32::MANTISSA_DIGITS)
                .saturating_sub(kernel.width())
                .min(i32::MAX as usize - kernel.width()),
        };
        for (&length, &modes) in fine_shape.iter().zip(config.n_modes()) {
            if length % 2 != 0 {
                return Err(NufftError::FineGridMustBeEven { length });
            }
            debug_assert!(
                length >= modes,
                "select_fine_grid_size returned a fine axis below the mode count"
            );
            if length > maximum_signed_length {
                return Err(NufftError::GpuFineGridIndexUnsupported {
                    length,
                    maximum: maximum_signed_length,
                });
            }
            u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 fine-grid axis length",
            })?;
        }
        let limits = device.limits();
        let maximum_workgroup_size = max_supported_workgroup_size(&limits);
        if maximum_workgroup_size < INTERPOLATION_WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: INTERPOLATION_WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let batch_capacity = config.batch();
        let mode_count = config.mode_count()?;
        let fine_product = fine_shape.iter().try_fold(1usize, |product, &length| {
            product
                .checked_mul(length)
                .ok_or(NufftError::LengthOverflow {
                    context: "rank-generic type-2 fine-grid element count",
                })
        })?;
        let fine_element_count = checked_product(
            "rank-generic type-2 batched fine-grid element count",
            fine_product,
            batch_capacity,
        )?;
        u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-2 batched fine-grid shader index space",
        })?;
        let fine_bytes = checked_buffer_size(
            "rank-generic type-2 fine grid",
            fine_element_count,
            precision.complex_size_bytes(),
        )?;
        let amplitude_bytes = checked_buffer_size(
            "rank-generic type-2 deconvolution amplitudes",
            mode_count,
            precision.scalar_size_bytes(),
        )?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        validate_binding_limit(
            "rank-generic type-2 fine grid",
            fine_bytes,
            max_storage_binding_bytes,
        )?;
        validate_binding_limit(
            "rank-generic type-2 deconvolution amplitudes",
            amplitude_bytes,
            max_storage_binding_bytes,
        )?;
        if fine_bytes > limits.max_buffer_size {
            return Err(NufftError::GpuBufferBindingTooLarge {
                buffer: "rank-generic type-2 fine grid",
                required_bytes: fine_bytes,
                limit_bytes: limits.max_buffer_size,
            });
        }

        let amplitudes = mode_amplitude_bytes_nd(config, kernel_fourier_coefficients, precision)?;
        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type2_nd.deconvolution_amplitudes"),
            contents: &amplitudes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_nd.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_nd.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new_nd(fine_shape.to_vec())
            .with_direction(fft_direction)
            .with_normalization(Normalization::None)
            .with_batch(batch_capacity)
            .with_precision(precision);
        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "rank-generic type-2 oversampled-grid C2C plan",
                source,
            }
        })?;

        let bin_shape = type2_bin_shape(fine_shape);
        let bins = NdPointBins::new(
            device,
            kernel,
            fine_shape,
            &bin_shape,
            NdBinOrder::Grouped,
            precision,
        )?;
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_nd.mode_scatter",
            &generate_mode_scatter_wgsl(config, fine_shape, precision),
        );
        let lanes = interpolation_lanes(kernel.width());
        let interpolation_pipeline = crate::gpu_type1_3d::create_compute_pipeline(
            device,
            "wgpu_nufft.type2_nd.interpolation",
            &generate_interpolation_wgsl(kernel, fine_shape, precision),
        );

        Ok(Self {
            fft,
            amplitudes: amplitude_buffer,
            fine_input,
            fine_output,
            bins,
            scatter_layout: scatter_pipeline.get_bind_group_layout(0),
            scatter_pipeline,
            interpolation_layout: interpolation_pipeline.get_bind_group_layout(0),
            interpolation_pipeline,
            points_per_workgroup: INTERPOLATION_WORKGROUP_SIZE as usize / lanes,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            batch_capacity,
            mode_count,
            dimensions,
            precision,
            max_storage_binding_bytes,
        })
    }

    /// Validates `point_count > 0` points and returns their byte size.
    fn validate_points(&self, point_count: usize, points: &wgpu::Buffer) -> Result<u64> {
        u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-2 GPU point count",
        })?;
        let coordinate_count = checked_product(
            "rank-generic type-2 coordinate scalar count",
            point_count,
            self.dimensions,
        )?;
        u32::try_from(coordinate_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-2 coordinate shader index space",
        })?;
        let point_bytes = checked_buffer_size(
            "rank-generic type-2 point buffer",
            coordinate_count,
            self.precision.scalar_size_bytes(),
        )?;
        validate_external_storage_buffer(
            "rank-generic type-2 point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        Ok(point_bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batch(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_impl(
            device,
            recorder,
            active_batch,
            point_count,
            points,
            coefficients,
            output,
            true,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    /// Records the point-dependent preparation, the bin order and the
    /// prepared supports, for later
    /// [`Self::encode_batch_with_recorded_points`] calls with the same
    /// `points` contents.
    pub(crate) fn set_points(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        point_count: usize,
        points: &wgpu::Buffer,
    ) -> Result<()> {
        if point_count == 0 {
            return Ok(());
        }
        let point_bytes = self.validate_points(point_count, points)?;
        self.bins
            .encode(device, recorder, point_count, points, point_bytes)
    }

    /// Like [`Self::encode_batch`], but reuses the preparation recorded by the
    /// most recent [`Self::set_points`] for these `points`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batch_with_recorded_points(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_impl(
            device,
            recorder,
            active_batch,
            point_count,
            points,
            coefficients,
            output,
            false,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
        query_set: &wgpu::QuerySet,
        first_query: u32,
    ) -> Result<NufftGpuProfileLayout> {
        if point_count == 0 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "rank-generic type-2 stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type2(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 stage-profile query range",
            })?;
        self.encode_impl(
            device,
            recorder,
            self.batch_capacity,
            point_count,
            points,
            coefficients,
            output,
            true,
            GpuProfileQueryWriter::enabled(query_set, &layout),
        )?;
        Ok(layout)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_impl(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
        record_points: bool,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        if point_count == 0 {
            return Ok(());
        }
        let point_bytes = self.validate_points(point_count, points)?;
        let coefficient_elements = checked_product(
            "rank-generic type-2 batched Fourier coefficient count",
            active_batch,
            self.mode_count,
        )?;
        let output_elements = checked_product(
            "rank-generic type-2 batched output element count",
            active_batch,
            point_count,
        )?;
        let coefficient_bytes = checked_buffer_size(
            "rank-generic type-2 Fourier coefficient buffer",
            coefficient_elements,
            self.precision.complex_size_bytes(),
        )?;
        let output_bytes = checked_buffer_size(
            "rank-generic type-2 output buffer",
            output_elements,
            self.precision.complex_size_bytes(),
        )?;
        validate_external_storage_buffer(
            "rank-generic type-2 Fourier coefficient",
            coefficients,
            coefficient_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "rank-generic type-2 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;
        if points == output {
            return Err(NufftError::GpuBufferAliasUnsupported {
                first: "rank-generic type-2 point",
                second: "rank-generic type-2 output",
            });
        }
        let coefficient_elements_u32 =
            u32::try_from(coefficient_elements).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 batched coefficient shader index space",
            })?;
        let scatter_dispatch = split_workgroups(
            coefficient_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;
        let interpolation_workgroups = u32::try_from(
            output_elements.div_ceil(self.points_per_workgroup),
        )
        .map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-2 batched interpolation workgroup count",
        })?;
        let interpolation_dispatch =
            split_workgroups(interpolation_workgroups, self.max_workgroups_per_dimension)?;

        if record_points {
            self.bins
                .encode(device, recorder, point_count, points, point_bytes)?;
        }
        let prepared = self.bins.prepared(point_count)?;

        // Deconvolution scatters one invocation per active mode onto the
        // cleared grid.
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_nd.mode_scatter.bind_group"),
            layout: &self.scatter_layout,
            entries: &[
                binding_entry(0, coefficients, coefficient_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.amplitudes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fine_input.as_entire_binding(),
                },
            ],
        });
        #[cfg(feature = "gpu-profiling")]
        profile.encode_start_marker(recorder);
        recorder.clear_buffer(&self.fine_input, 0, None);
        recorder.dispatch_profiled(
            "wgpu_nufft.type2_nd.mode_scatter.pass",
            &self.scatter_pipeline,
            &scatter_bind_group,
            scatter_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(1)),
        );

        self.fft
            .record_views(
                device,
                recorder.fft(),
                BufferView::whole(&self.fine_input),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "rank-generic type-2 oversampled-grid C2C transform",
                source,
            })?;

        let interpolation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_nd.interpolation.bind_group"),
            layout: &self.interpolation_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, &prepared.offsets, prepared.offset_bytes),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fine_output.as_entire_binding(),
                },
                binding_entry(3, output, output_bytes),
            ],
        });
        recorder.dispatch_profiled(
            "wgpu_nufft.type2_nd.interpolation.pass",
            &self.interpolation_pipeline,
            &interpolation_bind_group,
            interpolation_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(Some(2), Some(3)),
        );
        Ok(())
    }

    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        self.fft.diagnostics()
    }
}

/// Interpolation bins: small enough that consecutive points share cached
/// grid rows, and few enough that binning stays cheap.
fn type2_bin_shape(fine_shape: &[usize]) -> Vec<usize> {
    let mut sides = vec![TYPE2_BIN_SIDE; fine_shape.len()];
    let bin_count = |sides: &[usize]| {
        fine_shape
            .iter()
            .zip(sides)
            .map(|(&length, &side)| length.div_ceil(side))
            .product::<usize>()
    };
    let mut axis = fine_shape.len();
    while bin_count(&sides) > TYPE2_MAX_BINS {
        axis = if axis == 0 { fine_shape.len() - 1 } else { axis - 1 };
        sides[axis] *= 2;
    }
    sides
}

/// Lanes per point: one per axis-zero support offset, rounded up to a power
/// of two.
fn interpolation_lanes(width: usize) -> usize {
    width.next_power_of_two()
}

/// Deconvolves and scatters every active mode onto the cleared fine grid:
/// `grid[fine(k)] = f_k / prod_a phi_hat_a(k_a)`.
fn generate_mode_scatter_wgsl(
    config: &NufftConfig,
    fine_shape: &[usize],
    precision: FftPrecision,
) -> String {
    let types = NdWgsl::new(precision);
    let dimensions = config.dimensions();
    let n_modes = config.n_modes();
    let mode_count: usize = n_modes.iter().product();
    let fine_product: usize = fine_shape.iter().product();
    let centered = matches!(config.mode_order(), ModeOrder::Centered);
    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const MODE_COUNT: u32 = {mode_count}u;");
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    let _ = writeln!(source, "const CENTERED_ORDER: bool = {centered};");
    for axis in 0..dimensions {
        let modes = n_modes[axis];
        let _ = writeln!(source, "const FINE{axis}: u32 = {}u;", fine_shape[axis]);
        let _ = writeln!(source, "const MODE{axis}: u32 = {modes}u;");
        let _ = writeln!(source, "const HALF{axis}: u32 = {}u;", modes / 2);
        let _ = writeln!(source, "const NONNEG{axis}: u32 = {}u;", modes.div_ceil(2));
    }
    let amplitude_type = match precision {
        FftPrecision::F32 => "f32",
        FftPrecision::F64 => "f64",
        FftPrecision::Df64 => "Df64",
    };
    let _ = write!(
        source,
        r#"
@group(0) @binding(0) var<storage, read> coefficients: array<{complex}>;
@group(0) @binding(1) var<storage, read> amplitudes: array<{amplitude_type}>;
@group(0) @binding(2) var<storage, read_write> fine_grid: array<{complex}>;

fn fine_index_for_axis(mode_index: u32, mode_count: u32, fine_length: u32, half: u32, nonnegative: u32) -> u32 {{
    if (CENTERED_ORDER) {{
        if (mode_index < half) {{ return fine_length - (half - mode_index); }}
        return mode_index - half;
    }}
    if (mode_index < nonnegative) {{ return mode_index; }}
    return fine_length - (mode_count - mode_index);
}}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total = arrayLength(&coefficients);
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let linear_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (linear_index >= total) {{ return; }}
    let vector_index = linear_index / MODE_COUNT;
    let mode_index = linear_index - vector_index * MODE_COUNT;
    var remainder = mode_index;
"#,
        complex = types.complex_type(),
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let mode_index{axis} = remainder % MODE{axis}; remainder = remainder / MODE{axis};\n    \
             let fine_index{axis} = fine_index_for_axis(mode_index{axis}, MODE{axis}, FINE{axis}, HALF{axis}, NONNEG{axis});"
        );
    }
    let mut fold = format!("fine_index{}", dimensions - 1);
    for axis in (0..dimensions.saturating_sub(1)).rev() {
        fold = format!("fine_index{axis} + FINE{axis} * ({fold})");
    }
    let _ = writeln!(source, "    let fine_flat = {fold};");
    let _ = writeln!(
        source,
        "    fine_grid[vector_index * FINE_PRODUCT + fine_flat] = {};\n}}",
        types.complex_scale("coefficients[linear_index]", "amplitudes[mode_index]")
    );
    types.with_library(&source)
}

/// Interpolation over prepared points in bin order.
///
/// `LANES` invocations evaluate one point for one vector. Lane `j` owns
/// axis-zero support offset `j`, so the lanes of a point read each fine-grid
/// row as one contiguous run. Each lane sums its column over the other axes,
/// nested from the last axis inward, with the axis-one sum unrolled; loops
/// over the outer axes use a bound the compiler cannot unroll, which keeps
/// shader compilation fast for high ranks. The lane sums are then added in
/// lane order through workgroup memory.
fn generate_interpolation_wgsl(
    kernel: EsKernel,
    fine_shape: &[usize],
    precision: FftPrecision,
) -> String {
    let types = NdWgsl::new(precision);
    let dimensions = fine_shape.len();
    let width = kernel.width();
    let lanes = interpolation_lanes(width);
    let zero = types.complex_zero();

    // The column sum of one lane: nested loops over axes d-1..2, and the
    // axis-1 sum unrolled at the innermost level.
    let mut column = String::new();
    let inner_indent = "            ".to_owned() + &"    ".repeat(dimensions.saturating_sub(2));
    let row_terms = (2..dimensions)
        .map(|axis| format!("index_{axis}"))
        .collect::<Vec<_>>();
    let row = std::iter::once("grid_base".to_owned())
        .chain(row_terms)
        .collect::<Vec<_>>()
        .join(" + ");
    if dimensions >= 2 {
        let _ = writeln!(
            column,
            "            let start_1 = prepared_starts[base + 1u];\n            let offset_1 = prepared_offsets[slot * DIMS + 1u];"
        );
        for support in 0..width {
            let _ = writeln!(
                column,
                "            let weight_1_{support} = support_weight(offset_1, {support}u);\n            let index_1_{support} = wrap_index(start_1 + {support}, FINE_1_I32) * FINE_STRIDE_1;"
            );
        }
        for axis in (2..dimensions).rev() {
            let indent = "            ".to_owned() + &"    ".repeat(dimensions - 1 - axis);
            let _ = writeln!(
                column,
                "{indent}let start_{axis} = prepared_starts[base + {axis}u];
{indent}let offset_{axis} = prepared_offsets[slot * DIMS + {axis}u];
{indent}var sum_{axis} = {zero};
{indent}for (var support_{axis} = 0u; support_{axis} < loop_width; support_{axis} = support_{axis} + 1u) {{
{indent}    let index_{axis} = wrap_index(start_{axis} + i32(support_{axis}), FINE_{axis}_I32) * FINE_STRIDE_{axis};"
            );
        }
        let _ = writeln!(column, "{inner_indent}let row = {row};");
        let mut terms = Vec::new();
        for support in 0..width {
            terms.push(types.complex_scale(
                &format!("fine_grid[row + index_1_{support}]"),
                &format!("weight_1_{support}"),
            ));
        }
        let sum_1 = terms
            .into_iter()
            .reduce(|left, right| types.complex_add(&left, &right))
            .unwrap_or_else(|| zero.to_owned());
        let _ = writeln!(column, "{inner_indent}let sum_1 = {sum_1};");
        for axis in 2..dimensions {
            let indent = "            ".to_owned() + &"    ".repeat(dimensions - 1 - axis);
            let inner = if axis == 2 {
                "sum_1".to_owned()
            } else {
                format!("sum_{}", axis - 1)
            };
            let _ = writeln!(
                column,
                "{indent}    sum_{axis} = {};\n{indent}}}",
                types.complex_add(
                    &format!("sum_{axis}"),
                    &types.complex_scale(
                        &inner,
                        &format!("support_weight(offset_{axis}, support_{axis})")
                    ),
                )
            );
        }
        let _ = writeln!(
            column,
            "            lane_sum = {};",
            types.complex_scale(
                &format!("sum_{}", dimensions - 1),
                "support_weight(prepared_offsets[slot * DIMS], lane)"
            )
        );
    } else {
        let _ = writeln!(
            column,
            "            lane_sum = {};",
            types.complex_scale(
                "fine_grid[grid_base]",
                "support_weight(prepared_offsets[slot * DIMS], lane)"
            )
        );
    }
    let lane_total = types.complex_add("sum", "lane_sums[invocation + support]");
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {INTERPOLATION_WORKGROUP_SIZE}u;
const LANES: u32 = {lanes}u;
const POINTS_PER_WORKGROUP: u32 = {points_per_workgroup}u;
const DIMS: u32 = {dimensions}u;
const STRIDE: u32 = {stride}u;
{WRAP_INDEX_WGSL}
@group(0) @binding(0) var<storage, read> prepared_starts: array<i32>;
@group(0) @binding(1) var<storage, read> prepared_offsets: array<{offset_type}>;
@group(0) @binding(2) var<storage, read> fine_grid: array<{complex}>;
@group(0) @binding(3) var<storage, read_write> output_values: array<{complex}>;

var<workgroup> lane_sums: array<{complex}, {INTERPOLATION_WORKGROUP_SIZE}>;

@compute @workgroup_size({INTERPOLATION_WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) invocation: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let point_count = arrayLength(&prepared_starts) / STRIDE;
    let total = arrayLength(&output_values);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / POINTS_PER_WORKGROUP) {{ return; }}
    let lane = invocation % LANES;
    let work_index = wg_flat * POINTS_PER_WORKGROUP + invocation / LANES;
    let has_work = work_index < total;
    // WIDTH, but not a constant the compiler can unroll loops by: point
    // counts stay below 2^31.
    let loop_width = WIDTH + (point_count >> 31u);

    var lane_sum = {zero};
    var batch_index = 0u;
    var point_index = 0u;
    if (has_work) {{
        batch_index = work_index / point_count;
        let slot = work_index - batch_index * point_count;
        let base = slot * STRIDE;
        point_index = bitcast<u32>(prepared_starts[base + DIMS]);
        if (lane < WIDTH) {{
            let grid_base = batch_index * FINE_COUNT
                + wrap_index(prepared_starts[base] + i32(lane), FINE_0_I32);
{column}        }}
    }}
    lane_sums[invocation] = lane_sum;
    workgroupBarrier();
    if (has_work && lane == 0u) {{
        var sum = lane_sums[invocation];
        for (var support = 1u; support < WIDTH; support = support + 1u) {{
            sum = {lane_total};
        }}
        output_values[batch_index * point_count + point_index] = sum;
    }}
}}
"#,
        points_per_workgroup = INTERPOLATION_WORKGROUP_SIZE as usize / lanes,
        stride = dimensions + 1,
        offset_type = types.offset_type(),
        complex = types.complex_type(),
    );
    types.with_library(&format!(
        "{}\n{}\n{entry}",
        position_wgsl(fine_shape, kernel, precision),
        weight_wgsl(kernel, precision),
    ))
}

/// Product amplitude table over the full mode box, axis-0-fast linearization:
/// `amplitude[flat] = prod_i 1 / phi_hat_i(|k_i|)`.
pub(crate) fn mode_amplitudes_nd(
    config: &NufftConfig,
    coefficients: &[Vec<f64>],
) -> Result<Vec<f32>> {
    let n_modes = config.n_modes();
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "rank-generic type-2 deconvolution amplitudes",
            elements: mode_count,
        })?;
    for flat in 0..mode_count {
        let mut remainder = flat;
        let mut amplitude = 1.0f64;
        for (axis, &length) in n_modes.iter().enumerate() {
            let index = remainder % length;
            remainder /= length;
            let mode = mode_for_storage_index(config.mode_order(), index, length);
            let coefficient = coefficients[axis][mode.unsigned_abs() as usize];
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
            amplitude /= coefficient;
        }
        let amplitude = amplitude as f32;
        if !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: 0,
                value: f64::from(amplitude),
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

/// Full-accuracy variant of [`mode_amplitudes_nd`] for the F64/Df64 paths.
pub(crate) fn mode_amplitudes_nd_f64(
    config: &NufftConfig,
    coefficients: &[Vec<f64>],
) -> Result<Vec<f64>> {
    let n_modes = config.n_modes();
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "rank-generic type-2 deconvolution amplitudes",
            elements: mode_count,
        })?;
    for flat in 0..mode_count {
        let mut remainder = flat;
        let mut amplitude = 1.0f64;
        for (axis, &length) in n_modes.iter().enumerate() {
            let index = remainder % length;
            remainder /= length;
            let mode = mode_for_storage_index(config.mode_order(), index, length);
            let coefficient = coefficients[axis][mode.unsigned_abs() as usize];
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
            amplitude /= coefficient;
        }
        if !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: 0,
                value: amplitude,
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

/// Uploadable amplitude bytes in each precision's storage layout: `f32`,
/// `f64`, or df64 hi/lo pairs.
pub(crate) fn mode_amplitude_bytes_nd(
    config: &NufftConfig,
    coefficients: &[Vec<f64>],
    precision: FftPrecision,
) -> Result<Vec<u8>> {
    match precision {
        FftPrecision::F64 => {
            Ok(bytemuck::cast_slice(&mode_amplitudes_nd_f64(config, coefficients)?).to_vec())
        }
        FftPrecision::Df64 => {
            let values = mode_amplitudes_nd_f64(config, coefficients)?
                .into_iter()
                .map(DoubleFloat::from_f64)
                .collect::<Vec<_>>();
            Ok(bytemuck::cast_slice(&values).to_vec())
        }
        FftPrecision::F32 => {
            Ok(bytemuck::cast_slice(&mode_amplitudes_nd(config, coefficients)?).to_vec())
        }
    }
}

pub(crate) fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

pub(crate) fn format_wgsl_f32(value: f32) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}

pub(crate) fn format_wgsl_f64(value: f64) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted.push_str("lf");
    formatted
}

pub(crate) fn checked_buffer_size(
    context: &'static str,
    elements: usize,
    bytes_per_element: u64,
) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(bytes_per_element))
        .ok_or(NufftError::LengthOverflow { context })
}

pub(crate) fn checked_product(context: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(NufftError::LengthOverflow { context })
}

pub(crate) fn validate_binding_limit(buffer: &'static str, bytes: u64, limit: u64) -> Result<()> {
    if bytes > limit {
        Err(NufftError::GpuBufferBindingTooLarge {
            buffer,
            required_bytes: bytes,
            limit_bytes: limit,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_external_storage_buffer(
    label: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
    max_storage_binding_bytes: u64,
) -> Result<()> {
    if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(NufftError::GpuBufferMissingUsage {
            buffer: label,
            required_usage: "STORAGE",
        });
    }
    if buffer.size() < required_bytes {
        return Err(NufftError::GpuBufferTooSmall {
            buffer: label,
            required_bytes,
            actual_bytes: buffer.size(),
        });
    }
    validate_binding_limit(label, required_bytes, max_storage_binding_bytes)
}

pub(crate) fn binding_entry<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    size: u64,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(size),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_nd_wgsl::assert_valid_nd_wgsl;

    #[test]
    fn type2_shaders_validate_in_every_precision_and_rank() {
        for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
            for (n_modes, eps) in [
                (vec![5usize, 4, 3, 3], 1.0e-6),
                (vec![4usize; 5], 1.0e-6),
                (vec![3usize; 6], 1.0e-3),
                (vec![2usize; 8], 1.0e-3),
            ] {
                let config = NufftConfig::new(n_modes.clone(), eps).with_precision(precision);
                let kernel = EsKernel::for_tolerance(eps, 2.0).unwrap();
                let fine_shape = n_modes
                    .iter()
                    .map(|&modes| crate::kernel::select_fine_grid_size(modes, 2.0, kernel.width()).unwrap())
                    .collect::<Vec<_>>();
                assert_valid_nd_wgsl(
                    precision,
                    &generate_mode_scatter_wgsl(&config, &fine_shape, precision),
                );
                assert_valid_nd_wgsl(
                    precision,
                    &generate_interpolation_wgsl(kernel, &fine_shape, precision),
                );
            }
        }
    }

    #[test]
    fn interpolation_unrolls_only_the_axis_one_sum() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, &[16, 14, 14, 14], FftPrecision::F32);
        assert!(source.contains("let loop_width = WIDTH + (point_count >> 31u);"));
        assert!(source.contains("for (var support_3 = 0u; support_3 < loop_width;"));
        assert!(source.contains("for (var support_2 = 0u; support_2 < loop_width;"));
        assert!(!source.contains("support_1 < loop_width"));
        assert!(source.contains("let weight_1_6 = support_weight(offset_1, 6u);"));
        assert!(source.contains("let row = grid_base + index_2 + index_3;"));
    }

    #[test]
    fn type2_bins_stay_bounded() {
        assert_eq!(type2_bin_shape(&[16, 16, 16, 16]), vec![4; 4]);
        let wide = type2_bin_shape(&[40; 8]);
        let bins: usize = wide.iter().map(|&side| 40usize.div_ceil(side)).product();
        assert!(bins <= TYPE2_MAX_BINS);
    }

    #[test]
    fn nd_amplitudes_f64_match_f32_products() {
        let config = NufftConfig::new([3, 2], 1.0e-3);
        let coefficients = vec![vec![2.0, 4.0], vec![5.0, 8.0]];
        let single = mode_amplitudes_nd(&config, &coefficients).unwrap();
        let double = mode_amplitudes_nd_f64(&config, &coefficients).unwrap();
        assert_eq!(single.len(), double.len());
        for (&narrow, &wide) in single.iter().zip(&double) {
            assert!((f64::from(narrow) - wide).abs() <= f64::from(f32::EPSILON) * 4.0);
        }
    }

    #[test]
    fn nd_amplitudes_are_per_axis_products() {
        let config = NufftConfig::new([3, 2], 1.0e-3);
        let coefficients = vec![vec![2.0, 4.0], vec![5.0, 8.0]];
        let amplitudes = mode_amplitudes_nd(&config, &coefficients).unwrap();
        assert_eq!(amplitudes.len(), 6);
        // centered order, axis-0-fast: k0 in {-1,0,1}, k1 in {-1,0}
        // amplitude(flat = i0 + 3*i1) = 1/(c0[|k0|] * c1[|k1|])
        let expected = [
            1.0 / (4.0 * 8.0), // k0=-1, k1=-1
            1.0 / (2.0 * 8.0), // k0=0,  k1=-1
            1.0 / (4.0 * 8.0), // k0=1,  k1=-1
            1.0 / (4.0 * 5.0), // k0=-1, k1=0
            1.0 / (2.0 * 5.0), // k0=0,  k1=0
            1.0 / (4.0 * 5.0), // k0=1,  k1=0
        ];
        for (actual, expected) in amplitudes.iter().zip(expected) {
            assert!((actual - expected as f32).abs() <= f32::EPSILON * 4.0);
        }
    }
}
