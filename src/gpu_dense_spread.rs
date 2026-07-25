//! Deterministic spreading of many points onto a small F32 fine grid.
//!
//! Binned output-stationary spreaders parallelize over cells, which starves
//! a small grid holding many points per cell (the outer grids of 1D and 2D
//! type-3 plans hold tens of thousands of sources on under a hundred cells
//! per axis). [`DenseSpread`] instead splits the points, in their original
//! order, into at most [`MAX_GROUPS`] equal groups. One workgroup per group
//! and grid tile folds its group's points into workgroup memory chunk by
//! chunk; every invocation owns a segment of cells along axis 0 and sums the
//! group's contributions in point order into a per-group partial grid. A
//! reduction then adds the partial grids of every cell in group order. The
//! groups depend only on the point count, so the result is deterministic and
//! needs neither atomics nor point sorting.

use std::sync::Mutex;

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_type1_3d::{
    binding_entry, checked_buffer_size, create_compute_pipeline, encode_pass, format_wgsl_f32,
    validate_binding_limit, validate_buffer_limit,
};
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
/// Consecutive axis-0 cells owned by one spreading invocation.
const SEGMENT_CELLS: usize = 8;
/// Points folded into workgroup memory at a time.
const CHUNK: usize = 128;
/// Points per group before the group count reaches [`MAX_GROUPS`].
const TARGET_GROUP_POINTS: usize = 256;
/// Upper bound on groups, which bounds the partial grids.
const MAX_GROUPS: usize = 1024;
/// Largest fine grid the dense spreader serves: every invocation checks every
/// point of its group, so the work grows with the cell count.
const MAX_CELLS: usize = 1 << 16;
const COMPLEX_F32_BYTES: u64 = 8;
const F32_BYTES: u64 = 4;

/// Point groups for `point_count > 0` points; the shaders repeat this.
fn group_count(point_count: usize) -> usize {
    point_count
        .div_ceil(TARGET_GROUP_POINTS)
        .clamp(1, MAX_GROUPS)
}

pub(crate) struct DenseSpread {
    device: wgpu::Device,
    dimensions: usize,
    fine_count: usize,
    batch_capacity: usize,
    fine_grid: wgpu::Buffer,
    spread_pipeline: wgpu::ComputePipeline,
    spread_layout: wgpu::BindGroupLayout,
    reduce_pipeline: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    partials: Mutex<Option<wgpu::Buffer>>,
    tiles: usize,
    max_workgroups_per_dimension: u32,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
}

impl DenseSpread {
    /// Whether the dense spreader serves this F32 grid: one or two axes, at
    /// most [`MAX_CELLS`] cells, and room for a full support on every axis.
    pub(crate) fn supports(kernel: EsKernel, fine_shape: &[usize]) -> bool {
        (1..=2).contains(&fine_shape.len())
            && fine_shape
                .iter()
                .all(|&length| length >= 2 * kernel.width())
            && fine_shape
                .iter()
                .try_fold(1usize, |count, &length| count.checked_mul(length))
                .is_some_and(|count| count <= MAX_CELLS)
    }

    pub(crate) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        fine_shape: &[usize],
        batch_capacity: usize,
    ) -> Result<Self> {
        assert!(Self::supports(kernel, fine_shape));
        let dimensions = fine_shape.len();
        let limits = device.limits();
        let fine_count = fine_shape.iter().product::<usize>();
        let fine_bytes = checked_buffer_size(
            "dense-spread fine grid",
            fine_count
                .checked_mul(batch_capacity)
                .ok_or(NufftError::LengthOverflow {
                    context: "dense-spread batched fine-grid element count",
                })?,
            COMPLEX_F32_BYTES,
        )?;
        validate_binding_limit(
            "dense-spread fine grid",
            fine_bytes,
            limits.max_storage_buffer_binding_size,
        )?;
        validate_buffer_limit("dense-spread fine grid", fine_bytes, limits.max_buffer_size)?;
        let fine_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.dense_spread.fine_grid"),
            size: fine_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let segments = fine_shape[0].div_ceil(SEGMENT_CELLS) * fine_count / fine_shape[0];
        let tiles = segments.div_ceil(WORKGROUP_SIZE as usize);
        let spread_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.dense_spread.spread",
            &generate_spread_wgsl(kernel, fine_shape, tiles),
        );
        let reduce_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.dense_spread.reduce",
            &generate_reduce_wgsl(fine_count),
        );
        Ok(Self {
            device: device.clone(),
            dimensions,
            fine_count,
            batch_capacity,
            fine_grid,
            spread_layout: spread_pipeline.get_bind_group_layout(0),
            spread_pipeline,
            reduce_layout: reduce_pipeline.get_bind_group_layout(0),
            reduce_pipeline,
            partials: Mutex::new(None),
            tiles,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    pub(crate) fn fine_grid_buffer(&self) -> &wgpu::Buffer {
        &self.fine_grid
    }

    pub(crate) const fn fine_grid_element_count(&self) -> usize {
        self.fine_count
    }

    fn partials(&self, bytes: u64) -> wgpu::Buffer {
        let mut cached = self
            .partials
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match cached.as_ref() {
            Some(buffer) if buffer.size() >= bytes => buffer.clone(),
            _ => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wgpu_nufft.dense_spread.partials"),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                *cached = Some(buffer.clone());
                buffer
            }
        }
    }

    /// Spreads `active_batch` transform-major strength vectors of
    /// `point_count` point-major F32 points into the plan-owned fine grid,
    /// overwriting its active vectors.
    pub(crate) fn encode_spread_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        let active_cells =
            self.fine_count
                .checked_mul(active_batch)
                .ok_or(NufftError::LengthOverflow {
                    context: "dense-spread active fine-grid element count",
                })?;
        let active_fine_bytes =
            checked_buffer_size("dense-spread fine grid", active_cells, COMPLEX_F32_BYTES)?;
        if point_count == 0 {
            encoder.clear_buffer(&self.fine_grid, 0, Some(active_fine_bytes));
            return Ok(());
        }
        let coordinates =
            point_count
                .checked_mul(self.dimensions)
                .ok_or(NufftError::LengthOverflow {
                    context: "dense-spread point coordinate count",
                })?;
        u32::try_from(coordinates).map_err(|_| NufftError::LengthOverflow {
            context: "dense-spread point coordinate shader index space",
        })?;
        let point_bytes = checked_buffer_size("dense-spread points", coordinates, F32_BYTES)?;
        let strength_bytes = checked_buffer_size(
            "dense-spread strengths",
            point_count
                .checked_mul(active_batch)
                .ok_or(NufftError::LengthOverflow {
                    context: "dense-spread strength count",
                })?,
            COMPLEX_F32_BYTES,
        )?;
        let groups = group_count(point_count);
        let partial_bytes = checked_buffer_size(
            "dense-spread partial grids",
            active_cells
                .checked_mul(groups)
                .ok_or(NufftError::LengthOverflow {
                    context: "dense-spread partial-grid element count",
                })?,
            COMPLEX_F32_BYTES,
        )?;
        validate_binding_limit(
            "dense-spread partial grids",
            partial_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_buffer_limit(
            "dense-spread partial grids",
            partial_bytes,
            self.max_buffer_bytes,
        )?;
        let partials = self.partials(partial_bytes);

        let spread_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.dense_spread.spread.bind_group"),
            layout: &self.spread_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, strengths, strength_bytes),
                binding_entry(2, &partials, partial_bytes),
            ],
        });
        let reduce_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.dense_spread.reduce.bind_group"),
            layout: &self.reduce_layout,
            entries: &[
                binding_entry(0, &partials, partial_bytes),
                binding_entry(1, &self.fine_grid, active_fine_bytes),
            ],
        });
        let spread_workgroups = self
            .tiles
            .checked_mul(groups)
            .and_then(|count| count.checked_mul(active_batch))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "dense-spread workgroup count",
            })?;
        let reduce_workgroups = u32::try_from(active_cells.div_ceil(WORKGROUP_SIZE as usize))
            .map_err(|_| NufftError::LengthOverflow {
                context: "dense-spread reduction workgroup count",
            })?;
        encode_pass(
            encoder,
            "wgpu_nufft.dense_spread.spread.pass",
            &self.spread_pipeline,
            &spread_bind_group,
            split_workgroups(spread_workgroups, self.max_workgroups_per_dimension)?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );
        encode_pass(
            encoder,
            "wgpu_nufft.dense_spread.reduce.pass",
            &self.reduce_pipeline,
            &reduce_bind_group,
            split_workgroups(reduce_workgroups, self.max_workgroups_per_dimension)?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );
        Ok(())
    }
}

fn split_f64(value: f64) -> (f32, f32) {
    let high = value as f32;
    (high, (value - f64::from(high)) as f32)
}

fn generate_spread_wgsl(kernel: EsKernel, fine_shape: &[usize], tiles: usize) -> String {
    let dimensions = fine_shape.len();
    let width = kernel.width();
    let axis_constants: String = fine_shape
        .iter()
        .enumerate()
        .map(|(axis, &length)| {
            let (scale_hi, scale_lo) = split_f64(length as f64 / std::f64::consts::TAU);
            format!(
                "const FINE_{axis}: u32 = {length}u;
const FINE_{axis}_I32: i32 = {length}i;
const FINE_{axis}_F32: f32 = {length_f32};
const SCALE_{axis}_HI: f32 = {scale_hi};
const SCALE_{axis}_LO: f32 = {scale_lo};
const ORIGIN_{axis}: f32 = {origin};
",
                length_f32 = format_wgsl_f32(length as f32),
                scale_hi = format_wgsl_f32(scale_hi),
                scale_lo = format_wgsl_f32(scale_lo),
                origin = format_wgsl_f32((length / 2) as f32),
            )
        })
        .collect();
    let load_axes: String = (0..dimensions)
        .map(|axis| {
            format!(
                "            let position_{axis} = fold_position(points[point_base + {axis}u], FINE_{axis}_F32, SCALE_{axis}_HI, SCALE_{axis}_LO, ORIGIN_{axis});
            let start_{axis} = ceil_df64_to_i32(df64_sub(position_{axis}, Df64(HALF_WIDTH, 0.0)));
            for (var support = 0u; support < WIDTH; support = support + 1u) {{
                let distance = df64_sub(Df64(f32(start_{axis} + i32(support)), 0.0), position_{axis});
                point_weights[weight_base + {axis}u * WIDTH + support] = es_weight(distance.hi + distance.lo);
            }}
"
            )
        })
        .collect();
    let starts = if dimensions == 1 {
        "vec2<i32>(start_0, 0)".to_string()
    } else {
        "vec2<i32>(start_0, start_1)".to_string()
    };
    let initialize: String = (0..SEGMENT_CELLS)
        .map(|cell| format!("    var sum_{cell} = vec2<f32>(0.0, 0.0);\n"))
        .collect();
    let accumulate: String = (0..SEGMENT_CELLS)
        .map(|cell| {
            format!(
                "                {{ let dx = wrap_offset(cell_0 + {cell} - start.x, FINE_0_I32); if (dx < WIDTH_I32) {{ sum_{cell} = sum_{cell} + value * point_weights[weight_base + u32(dx)]; }} }}\n"
            )
        })
        .collect();
    let store: String = (0..SEGMENT_CELLS)
        .map(|cell| {
            format!(
                "        if (cell_0 + {cell} < FINE_0_I32) {{ partials[row_index + {cell}u] = sum_{cell}; }}\n"
            )
        })
        .collect();
    let (row_setup, row_check) = if dimensions == 1 {
        ("    let row_1 = 0;\n".to_string(), "true".to_string())
    } else {
        (
            "    let row_1 = i32(segment / SEGMENTS_PER_ROW);\n".to_string(),
            "wrap_offset(row_1 - start.y, FINE_1_I32) < WIDTH_I32".to_string(),
        )
    };
    let axis1_weight = if dimensions == 1 {
        "1.0".to_string()
    } else {
        "point_weights[weight_base + WIDTH + u32(wrap_offset(row_1 - start.y, FINE_1_I32))]"
            .to_string()
    };
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const POINT_DIMENSIONS: u32 = {dimensions}u;
{axis_constants}const FINE_COUNT: u32 = {fine_count}u;
const SEGMENT_CELLS: i32 = {SEGMENT_CELLS}i;
const SEGMENTS_PER_ROW: u32 = {segments_per_row}u;
const SEGMENTS: u32 = {segments}u;
const TILES: u32 = {tiles}u;
const CHUNK: u32 = {CHUNK}u;
const TARGET_GROUP_POINTS: u32 = {TARGET_GROUP_POINTS}u;
const MAX_GROUPS: u32 = {MAX_GROUPS}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const WEIGHTS_PER_POINT: u32 = {weights_per_point}u;

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec2<f32>>;

var<workgroup> point_starts: array<vec2<i32>, {CHUNK}>;
var<workgroup> point_values: array<vec2<f32>, {CHUNK}>;
var<workgroup> point_weights: array<f32, {chunk_weights}>;

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn position_is_negative(value: Df64) -> bool {{
    return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);
}}

fn position_at_least_grid(value: Df64, fine_length: f32) -> bool {{
    return value.hi > fine_length ||
        (value.hi == fine_length && value.lo >= 0.0);
}}

fn fold_position(point: f32, fine_length: f32, scale_hi: f32, scale_lo: f32, origin: f32) -> Df64 {{
    let scaled = df64_mul(Df64(point, 0.0), Df64(scale_hi, scale_lo));
    var position = df64_add(scaled, Df64(origin, 0.0));
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(fine_length, 0.0));
    }}
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(fine_length, 0.0));
    }}
    if (position_at_least_grid(position, fine_length)) {{
        position = df64_sub(position, Df64(fine_length, 0.0));
    }}
    if (position_at_least_grid(position, fine_length)) {{
        position = df64_sub(position, Df64(fine_length, 0.0));
    }}
    return position;
}}

// Periodic offset of a cell from a support start, in [0, fine_length).
fn wrap_offset(offset: i32, fine_length: i32) -> i32 {{
    var wrapped = offset;
    if (wrapped < 0) {{ wrapped = wrapped + fine_length; }}
    if (wrapped >= fine_length) {{ wrapped = wrapped - fine_length; }}
    return wrapped;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let point_count = arrayLength(&points) / POINT_DIMENSIONS;
    let groups = clamp((point_count + TARGET_GROUP_POINTS - 1u) / TARGET_GROUP_POINTS, 1u, MAX_GROUPS);
    let group_points = (point_count + groups - 1u) / groups;
    let total_vectors = arrayLength(&partials) / (FINE_COUNT * groups);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat >= TILES * groups * total_vectors) {{ return; }}
    let tile = wg_flat % TILES;
    let group = (wg_flat / TILES) % groups;
    let vector_index = wg_flat / (TILES * groups);
    let group_start = group * group_points;
    let group_end = min(group_start + group_points, point_count);

    let segment = tile * WORKGROUP_SIZE + lid;
    let cell_0 = i32(segment % SEGMENTS_PER_ROW) * SEGMENT_CELLS;
{row_setup}{initialize}
    var chunk_start = group_start;
    loop {{
        if (chunk_start >= group_end) {{ break; }}
        let chunk_length = min(CHUNK, group_end - chunk_start);
        for (var slot = lid; slot < chunk_length; slot = slot + WORKGROUP_SIZE) {{
            let point_index = chunk_start + slot;
            let point_base = point_index * POINT_DIMENSIONS;
            let weight_base = slot * WEIGHTS_PER_POINT;
{load_axes}            point_starts[slot] = {starts};
            point_values[slot] = strengths[vector_index * point_count + point_index];
        }}
        workgroupBarrier();
        if (segment < SEGMENTS) {{
            for (var slot = 0u; slot < chunk_length; slot = slot + 1u) {{
                let start = point_starts[slot];
                if ({row_check}) {{
                    let weight_base = slot * WEIGHTS_PER_POINT;
                    let value = point_values[slot] * {axis1_weight};
{accumulate}                }}
            }}
        }}
        workgroupBarrier();
        chunk_start = chunk_start + chunk_length;
    }}

    if (segment < SEGMENTS) {{
        let row_index = (vector_index * groups + group) * FINE_COUNT + u32(cell_0) +
            FINE_0 * u32(row_1);
{store}    }}
}}
"#,
        fine_count = fine_shape.iter().product::<usize>(),
        segments_per_row = fine_shape[0].div_ceil(SEGMENT_CELLS),
        segments =
            fine_shape[0].div_ceil(SEGMENT_CELLS) * fine_shape.iter().skip(1).product::<usize>(),
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
        weights_per_point = dimensions * width,
        chunk_weights = CHUNK * dimensions * width,
    );
    format!("{}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_reduce_wgsl(fine_count: usize) -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE_COUNT: u32 = {fine_count}u;

@group(0) @binding(0) var<storage, read> partials: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fine_grid: array<vec2<f32>>;

// Adds the partial grids of every active cell in group order.
@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&fine_grid);
    let groups = arrayLength(&partials) / total;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= total) {{ return; }}
    let vector_index = index / FINE_COUNT;
    let cell = index - vector_index * FINE_COUNT;
    let base = vector_index * groups * FINE_COUNT + cell;
    var sum = vec2<f32>(0.0, 0.0);
    for (var group = 0u; group < groups; group = group + 1u) {{
        sum = sum + partials[base + group * FINE_COUNT];
    }}
    fine_grid[index] = sum;
}}
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_wgsl(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}"));
    }

    #[test]
    fn dense_spread_shaders_validate_for_one_and_two_axes() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        for shape in [vec![72], vec![72, 64]] {
            assert!(DenseSpread::supports(kernel, &shape));
            let source = generate_spread_wgsl(kernel, &shape, 3);
            assert_valid_wgsl(&source);
            assert!(!source.contains("atomic"));
            assert_eq!(
                source.matches("partials[row_index + ").count(),
                SEGMENT_CELLS
            );
        }
        assert_valid_wgsl(&generate_reduce_wgsl(72 * 64));
        assert!(!DenseSpread::supports(kernel, &[512, 512]));
        assert!(!DenseSpread::supports(kernel, &[8]));
        assert!(!DenseSpread::supports(kernel, &[72, 72, 72]));
    }

    #[test]
    fn group_count_bounds_partials_and_matches_the_shader_formula() {
        assert_eq!(group_count(1), 1);
        assert_eq!(group_count(256), 1);
        assert_eq!(group_count(257), 2);
        assert_eq!(group_count(65_536), 256);
        assert_eq!(group_count(10_000_000), MAX_GROUPS);
    }
}
