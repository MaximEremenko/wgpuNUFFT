//! Deterministic output-stationary 3D type-1 spreading over coarse point bins.
//!
//! [`PointBins`] records a stable coarse-bin order of the points and their
//! prepared support starts and df64 offsets. [`BlockSpread3d::encode_spread`]
//! then assigns each workgroup a `16x16x8` block of fine-grid cells. Every
//! invocation owns one 16-cell x-row and keeps its sums in registers while the
//! workgroup streams the sorted points of the bins that can reach the block, in
//! a fixed order, through workgroup memory. Each cell therefore sums its
//! contributions in a fixed order, and every fine-grid cell is written exactly
//! once: no float atomics, no read-modify-write, and no clearing pass.

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::GpuProfileQueryWriter;
use crate::kernel::EsKernel;

use super::{
    binding_entry, create_compute_pipeline, encode_pass, format_wgsl_f32, generate_position_wgsl,
    DIMENSIONS,
};
use crate::gpu_point_bins::{PointBinOrder, PointBins};

/// Cells per coarse point bin along every axis.
const BIN_SIDE: usize = 4;
/// Fine-grid cells owned by one spreading workgroup, axis zero fastest.
/// `16x16x8` and `16x8x8` blocks spread 64^3 and
/// 128^3 type-1 plans equally fast; `16x16x16` was about 40% slower.
const BLOCK: [usize; DIMENSIONS] = [16, 16, 8];
/// One invocation per x-row of the block.
const SPREAD_WORKGROUP_SIZE: u32 = (BLOCK[1] * BLOCK[2]) as u32;
/// Extra cells gathered on each side of a block's reach. None are needed:
/// binning and spreading derive a point's cell `floor(p)` and support start
/// `ceil(p - w/2)` from the same folded df64 position `p`, and a point whose
/// support starts in `[o - w + 1, o + B - 1]` has `floor(p)` in
/// `[o - ceil(w/2), o + B - 1 + floor(w/2)]` even when rounding in the
/// subtraction moves the start by one cell.
const REACH_MARGIN: usize = 0;
const CHUNK_CANDIDATES: [usize; 3] = [128, 64, 32];

/// Static geometry of the block spreader for one plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlockLayout {
    width: usize,
    bins: [usize; DIMENSIONS],
    blocks: [usize; DIMENSIONS],
    max_rows: usize,
    chunk: usize,
}

impl BlockLayout {
    /// Returns the layout when this device and grid can run the block
    /// spreader, or `None` so the caller keeps the per-cell gather.
    pub(super) fn for_grid(
        kernel: EsKernel,
        fine_shape: [usize; DIMENSIONS],
        limits: &wgpu::Limits,
    ) -> Option<Self> {
        let width = kernel.width();
        let reach = width + 2 * REACH_MARGIN;
        let workgroup_size = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x);
        if workgroup_size < SPREAD_WORKGROUP_SIZE {
            return None;
        }
        let bins = fine_shape.map(|length| length.div_ceil(BIN_SIDE));
        let mut blocks = [0usize; DIMENSIONS];
        let mut max_bins = [0usize; DIMENSIONS];
        for axis in 0..DIMENSIONS {
            let length = fine_shape[axis];
            let block = BLOCK[axis];
            let region = block + reach;
            // A block's reach must not wrap onto itself, both wrapped parts of
            // a reach must fall into distinct bins, and every relative support
            // start must have a unique unwrapped representative.
            let minimum = (2 * block)
                .max(block + 2 * width)
                .max(region + BIN_SIDE + 1);
            if length < minimum {
                return None;
            }
            blocks[axis] = length.div_ceil(block);
            max_bins[axis] = region.div_ceil(BIN_SIDE) + 3;
        }
        let bin_count = bins
            .iter()
            .try_fold(1usize, |count, &n| count.checked_mul(n))?;
        let block_count = blocks
            .iter()
            .try_fold(1usize, |count, &n| count.checked_mul(n))?;
        u32::try_from(bin_count.checked_add(1)?).ok()?;
        u32::try_from(block_count).ok()?;
        let max_rows = max_bins[1] * max_bins[2];
        let chunk = CHUNK_CANDIDATES.into_iter().find(|&chunk| {
            storage_bytes(width, max_rows, chunk)
                .is_some_and(|bytes| bytes <= limits.max_compute_workgroup_storage_size)
        })?;
        Some(Self {
            width,
            bins,
            blocks,
            max_rows,
            chunk,
        })
    }

    fn block_count(&self) -> usize {
        self.blocks.iter().product()
    }
}

fn storage_bytes(width: usize, max_rows: usize, chunk: usize) -> Option<u32> {
    let segments = 2 * max_rows;
    // vec3<i32> has a 16-byte array stride; vec2<f32> an 8-byte stride.
    let per_point = 16 + 8 + 3 * width * 4;
    let bytes = chunk
        .checked_mul(per_point)?
        .checked_add(segments * 8)?
        .checked_add((segments + 1) * 4)?;
    u32::try_from(bytes).ok()
}

pub(super) struct BlockSpread3d {
    layout: BlockLayout,
    bins: PointBins,
    spread_pipeline: wgpu::ComputePipeline,
    spread_layout: wgpu::BindGroupLayout,
    max_workgroups_per_dimension: u32,
}

impl BlockSpread3d {
    pub(super) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        fine_shape: [usize; DIMENSIONS],
        layout: BlockLayout,
    ) -> Result<Self> {
        let bins = PointBins::new(
            device,
            kernel,
            &fine_shape,
            &[BIN_SIDE; DIMENSIONS],
            PointBinOrder::Stable,
            &generate_position_wgsl(fine_shape),
        )?;
        let spread_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_3d.block_spread",
            &generate_block_spread_wgsl(kernel, fine_shape, layout),
        );
        let spread_layout = spread_pipeline.get_bind_group_layout(0);
        Ok(Self {
            layout,
            bins,
            spread_pipeline,
            spread_layout,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        })
    }

    /// Records the stable coarse-bin order of `point_count > 0` points.
    ///
    /// The order stays valid for later [`Self::encode_spread`] calls until the
    /// next call to this method; executions must keep queue order.
    pub(super) fn encode_bins(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        point_bytes: u64,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        self.bins.encode(
            device,
            encoder,
            point_count,
            points,
            point_bytes,
            #[cfg(feature = "gpu-profiling")]
            profile,
        )
    }

    /// Spreads `active_batch` transform-major strength vectors of the points
    /// last recorded by [`Self::encode_bins`] into `fine_grid`, overwriting
    /// every active cell.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_spread(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        strengths: &wgpu::Buffer,
        strength_bytes: u64,
        fine_grid: &wgpu::Buffer,
        active_fine_bytes: u64,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        debug_assert!(point_count > 0);
        let prepared = self.bins.prepared(point_count)?;
        let bin_offsets = self.bins.bin_offsets();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.block_spread.bind_group"),
            layout: &self.spread_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, &prepared.offsets, prepared.offset_bytes),
                binding_entry(2, strengths, strength_bytes),
                binding_entry(3, bin_offsets, bin_offsets.size()),
                binding_entry(4, fine_grid, active_fine_bytes),
            ],
        });
        let workgroups = self
            .layout
            .block_count()
            .checked_mul(active_batch)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 3D block-spread workgroup count",
            })?;
        let dispatch = split_workgroups(workgroups, self.max_workgroups_per_dimension)?;
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.block_spread.pass",
            &self.spread_pipeline,
            &bind_group,
            dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(5)),
        );
        Ok(())
    }
}

fn generate_block_spread_wgsl(
    kernel: EsKernel,
    fine_shape: [usize; DIMENSIONS],
    layout: BlockLayout,
) -> String {
    let position = generate_position_wgsl(fine_shape);
    let width = layout.width;
    let segments = 2 * layout.max_rows;
    let rows_per_block_x = BLOCK[0];
    let initialize: String = (0..rows_per_block_x)
        .map(|cell| format!("    var sum_{cell} = vec2<f32>(0.0, 0.0);\n"))
        .collect();
    let accumulate: String = (0..rows_per_block_x)
        .map(|cell| {
            format!(
                "                {{ let dx = {cell} - start.x; if (dx >= 0 && dx < WIDTH_I32) {{ sum_{cell} = sum_{cell} + value * point_weights[weight_base + u32(dx)]; }} }}\n"
            )
        })
        .collect();
    let store: String = (0..rows_per_block_x)
        .map(|cell| {
            format!(
                "        if (origin.x + {cell} < FINE_0_I32) {{ fine_grid[row_index + {cell}u] = sum_{cell}; }}\n"
            )
        })
        .collect();
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {SPREAD_WORKGROUP_SIZE}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const BIN_SIDE: i32 = {BIN_SIDE}i;
const BINS_0: i32 = {bins_0}i;
const BINS_1: i32 = {bins_1}i;
const BLOCK_0: i32 = {block_0}i;
const BLOCK_1: i32 = {block_1}i;
const BLOCK_2: i32 = {block_2}i;
const BLOCKS_0: u32 = {blocks_0}u;
const BLOCKS_1: u32 = {blocks_1}u;
const BLOCK_COUNT: u32 = {block_count}u;
const REACH_BELOW: i32 = {reach_below}i;
const REACH_ABOVE: i32 = {reach_above}i;
const MAX_ROWS: u32 = {max_rows}u;
const SEGMENTS: u32 = {segments}u;
const CHUNK: u32 = {chunk}u;
const WEIGHTS_PER_POINT: u32 = {weights_per_point}u;

@group(0) @binding(0) var<storage, read> prepared_starts: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read> prepared_offsets: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<vec2<f32>>;

var<workgroup> segment_start: array<u32, {segments}>;
var<workgroup> segment_length: array<u32, {segments}>;
var<workgroup> segment_prefix: array<u32, {segments_plus_one}>;
var<workgroup> point_starts: array<vec3<i32>, {chunk}>;
var<workgroup> point_values: array<vec2<f32>, {chunk}>;
var<workgroup> point_weights: array<f32, {chunk_weights}>;

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

// Bins covering cells [low, high] of a periodic axis as up to two ranges.
struct BinRanges {{
    first_0: i32,
    last_0: i32,
    first_1: i32,
    last_1: i32,
}}

fn bin_ranges(low: i32, high: i32, fine_length: i32) -> BinRanges {{
    if (low < 0) {{
        return BinRanges((low + fine_length) / BIN_SIDE, (fine_length - 1) / BIN_SIDE,
            0, high / BIN_SIDE);
    }}
    if (high >= fine_length) {{
        return BinRanges(low / BIN_SIDE, (fine_length - 1) / BIN_SIDE,
            0, (high - fine_length) / BIN_SIDE);
    }}
    return BinRanges(low / BIN_SIDE, high / BIN_SIDE, 0, -1);
}}

fn bin_range_length(ranges: BinRanges) -> i32 {{
    return (ranges.last_0 - ranges.first_0 + 1) + max(ranges.last_1 - ranges.first_1 + 1, 0);
}}

fn bin_range_item(ranges: BinRanges, item: i32) -> i32 {{
    let first_length = ranges.last_0 - ranges.first_0 + 1;
    if (item < first_length) {{ return ranges.first_0 + item; }}
    return ranges.first_1 + (item - first_length);
}}

// Support start relative to the block origin, unwrapped into the block frame.
fn relative_start(start: i32, origin: i32, fine_length: i32) -> i32 {{
    var relative = start - origin;
    if (relative < -2 * WIDTH_I32) {{ relative = relative + fine_length; }}
    if (relative > fine_length / 2) {{ relative = relative - fine_length; }}
    return relative;
}}

@compute @workgroup_size({SPREAD_WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    if (wg_flat >= BLOCK_COUNT * total_vectors) {{ return; }}
    let vector_index = wg_flat / BLOCK_COUNT;
    let block = wg_flat - vector_index * BLOCK_COUNT;
    let origin = vec3<i32>(
        i32(block % BLOCKS_0) * BLOCK_0,
        i32((block / BLOCKS_0) % BLOCKS_1) * BLOCK_1,
        i32(block / (BLOCKS_0 * BLOCKS_1)) * BLOCK_2);
    let point_count = arrayLength(&prepared_starts);

    let ranges_0 = bin_ranges(origin.x - REACH_BELOW, origin.x + BLOCK_0 - 1 + REACH_ABOVE,
        FINE_0_I32);
    let ranges_1 = bin_ranges(origin.y - REACH_BELOW, origin.y + BLOCK_1 - 1 + REACH_ABOVE,
        FINE_1_I32);
    let ranges_2 = bin_ranges(origin.z - REACH_BELOW, origin.z + BLOCK_2 - 1 + REACH_ABOVE,
        FINE_2_I32);
    let rows_1 = bin_range_length(ranges_1);
    let rows = u32(rows_1 * bin_range_length(ranges_2));
    // Each row of bins along x is one contiguous sorted range (two if it wraps).
    for (var row = lid; row < MAX_ROWS; row = row + WORKGROUP_SIZE) {{
        var start_0 = 0u;
        var length_0 = 0u;
        var start_1 = 0u;
        var length_1 = 0u;
        if (row < rows) {{
            let bin_1 = bin_range_item(ranges_1, i32(row) % rows_1);
            let bin_2 = bin_range_item(ranges_2, i32(row) / rows_1);
            let row_bin = BINS_0 * (bin_1 + BINS_1 * bin_2);
            start_0 = bin_offsets[u32(row_bin + ranges_0.first_0)];
            length_0 = bin_offsets[u32(row_bin + ranges_0.last_0) + 1u] - start_0;
            if (ranges_0.last_1 >= ranges_0.first_1) {{
                start_1 = bin_offsets[u32(row_bin + ranges_0.first_1)];
                length_1 = bin_offsets[u32(row_bin + ranges_0.last_1) + 1u] - start_1;
            }}
        }}
        segment_start[2u * row] = start_0;
        segment_length[2u * row] = length_0;
        segment_start[2u * row + 1u] = start_1;
        segment_length[2u * row + 1u] = length_1;
    }}
    workgroupBarrier();
    if (lid == 0u) {{
        var running = 0u;
        for (var segment = 0u; segment < SEGMENTS; segment = segment + 1u) {{
            segment_prefix[segment] = running;
            running = running + segment_length[segment];
        }}
        segment_prefix[SEGMENTS] = running;
    }}
    let total = workgroupUniformLoad(&segment_prefix[SEGMENTS]);

    let row_1 = i32(lid) % BLOCK_1;
    let row_2 = i32(lid) / BLOCK_1;
{initialize}
    var batch_start = 0u;
    loop {{
        if (batch_start >= total) {{ break; }}
        let batch_length = min(CHUNK, total - batch_start);
        // Every chunk slot is filled, whether or not CHUNK exceeds WORKGROUP_SIZE.
        for (var slot = lid; slot < batch_length; slot = slot + WORKGROUP_SIZE) {{
            let record = batch_start + slot;
            // Last segment whose prefix does not exceed `record`.
            var low = 0u;
            var high = SEGMENTS;
            loop {{
                if (high - low <= 1u) {{ break; }}
                let middle = (low + high) / 2u;
                if (segment_prefix[middle] <= record) {{ low = middle; }} else {{ high = middle; }}
            }}
            loop {{
                if (low + 1u >= SEGMENTS || segment_prefix[low + 1u] > record) {{ break; }}
                low = low + 1u;
            }}
            let sorted_slot = segment_start[low] + (record - segment_prefix[low]);
            let prepared = prepared_starts[sorted_slot];
            let point_index = bitcast<u32>(prepared.w);
            point_starts[slot] = vec3<i32>(
                relative_start(prepared.x, origin.x, FINE_0_I32),
                relative_start(prepared.y, origin.y, FINE_1_I32),
                relative_start(prepared.z, origin.z, FINE_2_I32));
            point_values[slot] = strengths[vector_index * point_count + point_index];
            // Distance from support cell `start + j` is `j + (start - position)`,
            // with the df64 offset split into high and low f32 words.
            let offset_0 = prepared_offsets[3u * sorted_slot];
            let offset_1 = prepared_offsets[3u * sorted_slot + 1u];
            let offset_2 = prepared_offsets[3u * sorted_slot + 2u];
            let weight_base = slot * WEIGHTS_PER_POINT;
            for (var support = 0u; support < WIDTH; support = support + 1u) {{
                let cell = f32(support);
                point_weights[weight_base + support] =
                    es_weight((cell + offset_0.x) + offset_0.y);
                point_weights[weight_base + WIDTH + support] =
                    es_weight((cell + offset_1.x) + offset_1.y);
                point_weights[weight_base + 2u * WIDTH + support] =
                    es_weight((cell + offset_2.x) + offset_2.y);
            }}
        }}
        workgroupBarrier();
        for (var slot = 0u; slot < batch_length; slot = slot + 1u) {{
            let start = point_starts[slot];
            let dy = row_1 - start.y;
            let dz = row_2 - start.z;
            if (dy >= 0 && dy < WIDTH_I32 && dz >= 0 && dz < WIDTH_I32) {{
                let weight_base = slot * WEIGHTS_PER_POINT;
                let value = point_values[slot] *
                    (point_weights[weight_base + WIDTH + u32(dy)] *
                        point_weights[weight_base + 2u * WIDTH + u32(dz)]);
{accumulate}
            }}
        }}
        workgroupBarrier();
        batch_start = batch_start + batch_length;
    }}

    let cell_1 = origin.y + row_1;
    let cell_2 = origin.z + row_2;
    if (cell_1 < FINE_1_I32 && cell_2 < FINE_2_I32) {{
        let row_index = vector_index * FINE_COUNT + u32(origin.x) +
            FINE_0 * (u32(cell_1) + FINE_1 * u32(cell_2));
{store}
    }}
}}
"#,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
        bins_0 = layout.bins[0],
        bins_1 = layout.bins[1],
        block_0 = BLOCK[0],
        block_1 = BLOCK[1],
        block_2 = BLOCK[2],
        blocks_0 = layout.blocks[0],
        blocks_1 = layout.blocks[1],
        block_count = layout.block_count(),
        reach_below = width.div_ceil(2) + REACH_MARGIN,
        reach_above = width / 2 + REACH_MARGIN,
        max_rows = layout.max_rows,
        segments_plus_one = segments + 1,
        chunk = layout.chunk,
        weights_per_point = 3 * width,
        chunk_weights = layout.chunk * 3 * width,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses and validates generated WGSL without a GPU, so reserved words
    /// and type errors fail here rather than at pipeline creation.
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

    fn limits_with_storage(bytes: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_compute_workgroup_storage_size: bytes,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn layout_requires_room_for_an_unambiguous_block_reach() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let limits = limits_with_storage(16 * 1024);
        assert!(BlockLayout::for_grid(kernel, [256, 256, 256], &limits).is_some());
        assert!(BlockLayout::for_grid(kernel, [32, 32, 32], &limits).is_some());
        assert!(BlockLayout::for_grid(kernel, [30, 256, 256], &limits).is_none());
        assert!(BlockLayout::for_grid(kernel, [256, 256, 14], &limits).is_none());
    }

    #[test]
    fn layout_fits_webgpu_default_workgroup_storage() {
        for eps in [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-7] {
            let kernel = EsKernel::for_tolerance(eps, 2.0).unwrap();
            let layout =
                BlockLayout::for_grid(kernel, [256, 256, 256], &wgpu::Limits::default()).unwrap();
            let bytes = storage_bytes(layout.width, layout.max_rows, layout.chunk).unwrap();
            assert!(bytes <= wgpu::Limits::default().max_compute_workgroup_storage_size);
            assert!(
                layout.max_rows * 2 >= 2 * 4,
                "at least a 2x2 bin neighbourhood"
            );
        }
    }

    #[test]
    fn spread_shader_writes_each_cell_once_without_atomics() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let layout =
            BlockLayout::for_grid(kernel, [256, 256, 256], &wgpu::Limits::default()).unwrap();
        let source = generate_block_spread_wgsl(kernel, [256, 256, 256], layout);
        assert_valid_wgsl(&source);
        assert!(!source.contains("atomic"));
        assert_eq!(source.matches("fine_grid[row_index + ").count(), BLOCK[0]);
        assert!(source.contains("var sum_15 = vec2<f32>(0.0, 0.0);"));
        assert!(source.contains("workgroupUniformLoad(&segment_prefix[SEGMENTS])"));
        // Chunks may be larger than the workgroup; every slot must be loaded.
        assert!(source.contains("slot < batch_length; slot = slot + WORKGROUP_SIZE"));
    }
}
