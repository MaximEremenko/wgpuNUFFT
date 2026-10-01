//! Rank-generic (runtime-dimensional) type-1 GPU execution.
//!
//! Points are binned once per point set into coarse bins in stable order
//! ([`NdPointBins`]), then spread by the deterministic output-stationary
//! block spreader ([`NdBlockSpread`]), transformed by an oversampled C2C
//! FFT, and deconvolved and truncated into the mode buffer. Dimension appears
//! only as generated WGSL, so one implementation serves every rank the plan
//! accepts, in all three precisions; `F64` and `Df64` evaluate the ES kernel
//! through the host-fitted Horner table because GLSL.std.450 transcendentals
//! are 32-bit only.

use std::fmt::Write as _;

use wgpu::util::DeviceExt;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{ModeOrder, NufftConfig, NufftSign};
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_nd::{
    binding_entry, checked_buffer_size, checked_product, create_compute_pipeline,
    mode_amplitude_bytes_nd, validate_binding_limit, validate_external_storage_buffer,
};
use crate::gpu_nd_bins::{NdBinOrder, NdPointBins};
use crate::gpu_nd_spread::{NdBlockSpread, NdSpreadLayout};
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::{GpuProfileQueryWriter, NufftGpuProfileLayout};
use crate::gpu_recorder::GpuRecorder;
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;

pub(crate) struct Type1GpuPlanNd {
    fft: FftPlan,
    amplitudes: wgpu::Buffer,
    fine_input: wgpu::Buffer,
    fine_output: wgpu::Buffer,
    bins: NdPointBins,
    spread: NdBlockSpread,
    deconvolution_pipeline: wgpu::ComputePipeline,
    deconvolution_layout: wgpu::BindGroupLayout,
    max_workgroups_per_dimension: u32,
    mode_count: usize,
    fine_product: usize,
    dimensions: usize,
    batch_capacity: usize,
    precision: FftPrecision,
    max_storage_binding_bytes: u64,
}

impl Type1GpuPlanNd {
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
        // Support indices and bin keys must remain exact in position
        // arithmetic; F32 and Df64 fold them through an f32 high word, while
        // native f64 keeps cells distinct across the signed i32 range.
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
            debug_assert!(length >= modes);
            if length > maximum_signed_length {
                return Err(NufftError::GpuFineGridIndexUnsupported {
                    length,
                    maximum: maximum_signed_length,
                });
            }
            u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 fine-grid axis length",
            })?;
        }
        let limits = device.limits();
        let maximum_workgroup_size = max_supported_workgroup_size(&limits);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let batch_capacity = config.batch();
        let mode_count = config.mode_count()?;
        let fine_product = fine_shape.iter().try_fold(1usize, |product, &length| {
            product
                .checked_mul(length)
                .ok_or(NufftError::LengthOverflow {
                    context: "rank-generic type-1 fine-grid element count",
                })
        })?;
        u32::try_from(fine_product).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-1 fine-grid index space",
        })?;
        let fine_element_count = checked_product(
            "rank-generic type-1 batched fine-grid element count",
            fine_product,
            batch_capacity,
        )?;
        u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-1 batched fine-grid shader index space",
        })?;
        let fine_bytes = checked_buffer_size(
            "rank-generic type-1 fine grid",
            fine_element_count,
            precision.complex_size_bytes(),
        )?;
        let amplitude_bytes = checked_buffer_size(
            "rank-generic type-1 deconvolution amplitudes",
            mode_count,
            precision.scalar_size_bytes(),
        )?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        for (label, bytes) in [
            ("rank-generic type-1 fine grid", fine_bytes),
            (
                "rank-generic type-1 deconvolution amplitudes",
                amplitude_bytes,
            ),
        ] {
            validate_binding_limit(label, bytes, max_storage_binding_bytes)?;
        }
        if fine_bytes > limits.max_buffer_size {
            return Err(NufftError::GpuBufferBindingTooLarge {
                buffer: "rank-generic type-1 fine grid",
                required_bytes: fine_bytes,
                limit_bytes: limits.max_buffer_size,
            });
        }

        let layout = NdSpreadLayout::for_grid(kernel, fine_shape, precision, &limits)?;
        let bins = NdPointBins::new(
            device,
            kernel,
            fine_shape,
            &layout.bin_shapes(),
            NdBinOrder::Stable,
            precision,
        )?;
        let spread = NdBlockSpread::new(device, kernel, fine_shape, precision, layout)?;

        let amplitudes = mode_amplitude_bytes_nd(config, kernel_fourier_coefficients, precision)?;
        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type1_nd.deconvolution_amplitudes"),
            contents: &amplitudes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_nd.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_nd.fine_output"),
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
                stage: "rank-generic type-1 oversampled-grid C2C plan",
                source,
            }
        })?;

        let deconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.deconvolution",
            &generate_deconvolution_wgsl_for_precision(config, fine_shape, precision),
        );
        let deconvolution_layout = deconvolution_pipeline.get_bind_group_layout(0);

        Ok(Self {
            fft,
            amplitudes: amplitude_buffer,
            fine_input,
            fine_output,
            bins,
            spread,
            deconvolution_pipeline,
            deconvolution_layout,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            mode_count,
            fine_product,
            dimensions,
            batch_capacity,
            precision,
            max_storage_binding_bytes,
        })
    }

    fn validate_active_batch(&self, active_batch: usize) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        Ok(())
    }

    /// Validates `point_count > 0` points and returns their byte size.
    fn validate_points(&self, point_count: usize, points: &wgpu::Buffer) -> Result<u64> {
        u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-1 GPU point count",
        })?;
        let coordinate_count = checked_product(
            "rank-generic type-1 coordinate scalar count",
            point_count,
            self.dimensions,
        )?;
        u32::try_from(coordinate_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-1 coordinate shader index space",
        })?;
        let point_bytes = checked_buffer_size(
            "rank-generic type-1 point buffer",
            coordinate_count,
            self.precision.scalar_size_bytes(),
        )?;
        validate_external_storage_buffer(
            "rank-generic type-1 point",
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
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_impl(
            device,
            recorder,
            active_batch,
            point_count,
            points,
            strengths,
            output,
            true,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    /// Records the point-dependent preparation, the stable bin order and the
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
        self.bins.encode(
            device,
            recorder,
            point_count,
            points,
            point_bytes,
            self.spread.choose_geometry(point_count),
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
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
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_impl(
            device,
            recorder,
            active_batch,
            point_count,
            points,
            strengths,
            output,
            false,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_impl(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
        record_points: bool,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        self.validate_active_batch(active_batch)?;
        let output_elements = checked_product(
            "rank-generic type-1 batched output element count",
            active_batch,
            self.mode_count,
        )?;
        let output_bytes = checked_buffer_size(
            "rank-generic type-1 output buffer",
            output_elements,
            self.precision.complex_size_bytes(),
        )?;
        validate_external_storage_buffer(
            "rank-generic type-1 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;

        // Type-1 with no sources has a nonempty, all-zero mode output: the
        // spread clears the fine grid, then FFT + deconvolution still run so
        // the external output is fully overwritten.
        self.encode_spread_impl(
            device,
            recorder,
            active_batch,
            point_count,
            points,
            strengths,
            record_points,
            #[cfg(feature = "gpu-profiling")]
            profile,
        )?;

        self.fft
            .record_views(
                device,
                recorder.fft(),
                BufferView::whole(&self.fine_input),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "rank-generic type-1 oversampled-grid C2C transform",
                source,
            })?;
        #[cfg(feature = "gpu-profiling")]
        profile.encode_marker(recorder, None, Some(6));

        let deconvolution_elements_u32 =
            u32::try_from(output_elements).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 deconvolution shader index space",
            })?;
        let deconvolution_dispatch = split_workgroups(
            deconvolution_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;
        let deconvolution_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.deconvolution.bind_group"),
            layout: &self.deconvolution_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.fine_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.amplitudes.as_entire_binding(),
                },
                binding_entry(2, output, output_bytes),
            ],
        });
        recorder.dispatch_profiled(
            "wgpu_nufft.type1_nd.deconvolution.pass",
            &self.deconvolution_pipeline,
            &deconvolution_bind_group,
            deconvolution_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(7)),
        );
        Ok(())
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
        query_set: &wgpu::QuerySet,
        first_query: u32,
    ) -> Result<NufftGpuProfileLayout> {
        if point_count == 0 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "rank-generic type-1 stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type1(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 stage-profile query range",
            })?;
        self.encode_impl(
            device,
            recorder,
            self.batch_capacity,
            point_count,
            points,
            strengths,
            output,
            true,
            GpuProfileQueryWriter::enabled(query_set, &layout),
        )?;
        Ok(layout)
    }

    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        self.fft.diagnostics()
    }

    /// Encodes only the deterministic spread into the plan-owned fine grid:
    /// no FFT, no deconvolution. With zero points the fine grid is cleared
    /// instead, so downstream consumers (the batched FFT here, or type-3's
    /// inner type-2) always see defined data.
    pub(crate) fn encode_spread_batch(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        self.validate_active_batch(active_batch)?;
        self.encode_spread_impl(
            device,
            recorder,
            active_batch,
            point_count,
            points,
            strengths,
            true,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    /// Returns the plan-owned fine grid populated by
    /// [`Self::encode_spread_batch`].
    pub(crate) fn fine_grid_buffer(&self) -> &wgpu::Buffer {
        &self.fine_input
    }

    pub(crate) const fn fine_grid_element_count(&self) -> usize {
        self.fine_product
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_spread_impl(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        record_points: bool,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        if point_count == 0 {
            recorder.clear_buffer(&self.fine_input, 0, None);
            return Ok(());
        }
        let point_bytes = self.validate_points(point_count, points)?;
        let strength_elements = checked_product(
            "rank-generic type-1 batched strength count",
            active_batch,
            point_count,
        )?;
        let strength_bytes = checked_buffer_size(
            "rank-generic type-1 strength buffer",
            strength_elements,
            self.precision.complex_size_bytes(),
        )?;
        validate_external_storage_buffer(
            "rank-generic type-1 strength",
            strengths,
            strength_bytes,
            self.max_storage_binding_bytes,
        )?;
        let active_fine_bytes = checked_buffer_size(
            "rank-generic type-1 active fine grid",
            checked_product(
                "rank-generic type-1 active fine elements",
                self.fine_product,
                active_batch,
            )?,
            self.precision.complex_size_bytes(),
        )?;
        // The geometry depends on the point count alone, so recorded points
        // were binned in the one chosen here.
        let geometry = self.spread.choose_geometry(point_count);
        if record_points {
            self.bins.encode(
                device,
                recorder,
                point_count,
                points,
                point_bytes,
                geometry,
                #[cfg(feature = "gpu-profiling")]
                profile,
            )?;
        }
        let prepared = self.bins.prepared(point_count)?;
        self.spread.encode_spread(
            device,
            recorder,
            active_batch,
            point_count,
            &prepared,
            self.bins.bin_offsets(),
            geometry,
            strengths,
            strength_bytes,
            &self.fine_input,
            active_fine_bytes,
            #[cfg(feature = "gpu-profiling")]
            profile,
        )?;
        // The spread writes the active vectors only; the batched FFT reads
        // the whole grid, so clear the tail.
        if active_batch < self.batch_capacity {
            recorder.clear_buffer(&self.fine_input, active_fine_bytes, None);
        }
        Ok(())
    }
}

fn generate_deconvolution_wgsl_for_precision(
    config: &NufftConfig,
    fine_shape: &[usize],
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_deconvolution_wgsl(config, fine_shape)
            .replace("vec2<f32>", "vec2<f64>")
            .replace("array<f32>", "array<f64>"),
        FftPrecision::Df64 => {
            let source = generate_deconvolution_wgsl(config, fine_shape)
                .replace("array<vec2<f32>>", "array<vec4<f32>>")
                .replace("array<f32>", "array<Df64>")
                .replace(
                    "output_values[linear_index] = fine_grid[vector_index * FINE_PRODUCT + fine_flat] *\n        amplitudes[output_index];",
                    "output_values[linear_index] = df64_complex_scale(\n        \
                     fine_grid[vector_index * FINE_PRODUCT + fine_flat],\n        \
                     amplitudes[output_index],\n    );",
                );
            format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
        }
        FftPrecision::F32 => generate_deconvolution_wgsl(config, fine_shape),
    }
}

fn generate_deconvolution_wgsl(config: &NufftConfig, fine_shape: &[usize]) -> String {
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
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f32>>;\n\
         @group(0) @binding(1) var<storage, read> amplitudes: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;\n\n",
    );
    source.push_str(
        "fn fine_index_for_axis(output_index: u32, mode_count: u32, fine_length: u32, half: u32, nonnegative: u32) -> u32 {\n\
         \x20   if (CENTERED_ORDER) {\n\
         \x20       if (output_index < half) {\n\
         \x20           return fine_length - (half - output_index);\n\
         \x20       }\n\
         \x20       return output_index - half;\n\
         \x20   }\n\
         \x20   if (output_index < nonnegative) {\n\
         \x20       return output_index;\n\
         \x20   }\n\
         \x20   return fine_length - (mode_count - output_index);\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   let total = arrayLength(&output_values);\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let linear_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (linear_index >= total) { return; }\n\
         \x20   let vector_index = linear_index / MODE_COUNT;\n\
         \x20   let output_index = linear_index - vector_index * MODE_COUNT;\n\n\
         \x20   var remainder = output_index;\n",
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
    source.push_str(
        "    output_values[linear_index] = fine_grid[vector_index * FINE_PRODUCT + fine_flat] *\n\
         \x20       amplitudes[output_index];\n}\n",
    );
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd_deconvolution_lowers_to_f64_and_df64() {
        let config = NufftConfig::new([5, 4, 3, 3], 1.0e-3);
        let f64_source =
            generate_deconvolution_wgsl_for_precision(&config, &[12, 10, 8, 8], FftPrecision::F64);
        assert!(f64_source.contains("array<vec2<f64>>"));
        assert!(f64_source.contains("var<storage, read> amplitudes: array<f64>;"));
        assert!(!f64_source.contains("vec2<f32>"));
        let df64_source =
            generate_deconvolution_wgsl_for_precision(&config, &[12, 10, 8, 8], FftPrecision::Df64);
        assert!(df64_source.contains("array<vec4<f32>>"));
        assert!(df64_source.contains("var<storage, read> amplitudes: array<Df64>;"));
        assert!(df64_source.contains("output_values[linear_index] = df64_complex_scale("));
        assert!(df64_source.contains("struct Df64"));
    }

    #[test]
    fn nd_deconvolution_shader_maps_modes_per_axis() {
        let config = NufftConfig::new([5, 4, 3, 3], 1.0e-3);
        let source = generate_deconvolution_wgsl(&config, &[12, 10, 8, 8]);
        assert!(source.contains("fn fine_index_for_axis("));
        assert!(source.contains("let mode_index3 = remainder % MODE3;"));
        assert!(source.contains(
            "let fine_flat = fine_index0 + FINE0 * (fine_index1 + FINE1 * (fine_index2 + FINE2 * (fine_index3)));"
        ));
        crate::wgsl_validation::assert_valid_wgsl(&source);
    }
}
