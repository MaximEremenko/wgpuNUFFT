//! Deterministic output-stationary 2D type-1 spreading over coarse point bins.
//!
//! [`PointBins`] records a stable coarse-bin order of the points and their
//! prepared support starts and df64 offsets. [`BlockSpread2d::encode_spread`]
//! then lets every invocation own a `4x4` tile of fine-grid cells: it walks
//! the few `4x4`-cell bins that can reach the tile, in a fixed order, and
//! adds every point of those bins, in bin order, into register sums. Each
//! cell therefore sums its contributions in a fixed order, and every
//! fine-grid cell is written exactly once: no float atomics, no
//! read-modify-write, and no clearing pass. Prepared offsets turn every
//! kernel weight into `es_weight((j + hi) + lo)`, without df64 folding, and
//! a permutation pass first copies the strengths into bin order, so the
//! gather reads them as contiguously as the prepared points.

use std::sync::Mutex;

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_point_bins::{PointBinOrder, PointBins};
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::GpuProfileQueryWriter;
use crate::gpu_recorder::GpuRecorder;
use crate::kernel::EsKernel;

use super::{
    binding_entry, create_compute_pipeline, encode_pass, format_wgsl_f32, generate_position_wgsl,
    DIMENSIONS,
};

const WORKGROUP_SIZE: u32 = 64;
/// Cells per coarse point bin along both axes.
const BIN_SIDE: usize = 4;
/// Fine-grid cells owned by one invocation, axis zero first.
const TILE: [usize; DIMENSIONS] = [4, 4];

/// Static geometry of the tile gather for one plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlockLayout2d {
    width: usize,
    bins: [usize; DIMENSIONS],
    tiles: [usize; DIMENSIONS],
}

impl BlockLayout2d {
    /// Returns the layout when the grid can run the tile gather, or `None`
    /// so the caller keeps the per-cell gather.
    pub(super) fn for_grid(
        kernel: EsKernel,
        fine_shape: [usize; DIMENSIONS],
        limits: &wgpu::Limits,
    ) -> Option<Self> {
        let width = kernel.width();
        let workgroup_size = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x);
        if workgroup_size < WORKGROUP_SIZE {
            return None;
        }
        let mut bins = [0usize; DIMENSIONS];
        let mut tiles = [0usize; DIMENSIONS];
        for axis in 0..DIMENSIONS {
            let length = fine_shape[axis];
            // A tile's reach must not wrap onto itself, both wrapped parts of
            // a reach must fall into distinct bins, and every relative support
            // start must have a unique unwrapped representative. As in 3D, a
            // point whose support starts in `[o - w + 1, o + T - 1]` has its
            // folded cell in `[o - ceil(w/2), o + T - 1 + floor(w/2)]`.
            let tile = TILE[axis];
            let minimum = (2 * tile)
                .max(tile + 2 * width)
                .max(tile + width + BIN_SIDE + 1);
            if length < minimum {
                return None;
            }
            bins[axis] = length.div_ceil(BIN_SIDE);
            tiles[axis] = length.div_ceil(TILE[axis]);
        }
        u32::try_from(bins[0].checked_mul(bins[1])?.checked_add(1)?).ok()?;
        u32::try_from(tiles[0].checked_mul(tiles[1])?).ok()?;
        Some(Self { width, bins, tiles })
    }

    fn tile_count(&self) -> usize {
        self.tiles.iter().product()
    }
}

pub(super) struct BlockSpread2d {
    layout: BlockLayout2d,
    bins: PointBins,
    permute_pipeline: wgpu::ComputePipeline,
    permute_layout: wgpu::BindGroupLayout,
    spread_pipeline: wgpu::ComputePipeline,
    spread_layout: wgpu::BindGroupLayout,
    /// Strengths in bin order, grown on demand.
    binned_strengths: Mutex<Option<wgpu::Buffer>>,
    device: wgpu::Device,
    max_workgroups_per_dimension: u32,
}

impl BlockSpread2d {
    pub(super) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        fine_shape: [usize; DIMENSIONS],
        layout: BlockLayout2d,
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
            "wgpu_nufft.type1_2d.tile_gather",
            &generate_tile_gather_wgsl(kernel, fine_shape, layout),
        );
        let spread_layout = spread_pipeline.get_bind_group_layout(0);
        let permute_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_2d.permute_strengths",
            &generate_permute_wgsl(),
        );
        Ok(Self {
            layout,
            bins,
            permute_layout: permute_pipeline.get_bind_group_layout(0),
            permute_pipeline,
            spread_pipeline,
            spread_layout,
            binned_strengths: Mutex::new(None),
            device: device.clone(),
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
        recorder: &mut GpuRecorder<'_>,
        point_count: usize,
        points: &wgpu::Buffer,
        point_bytes: u64,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        self.bins.encode(
            device,
            recorder,
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
        recorder: &mut GpuRecorder<'_>,
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
        let binned_strengths = {
            let mut cached = self
                .binned_strengths
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match cached.as_ref() {
                Some(buffer) if buffer.size() >= strength_bytes => buffer.clone(),
                _ => {
                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("wgpu_nufft.type1_2d.binned_strengths"),
                        size: strength_bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    *cached = Some(buffer.clone());
                    buffer
                }
            }
        };
        let permute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_2d.permute_strengths.bind_group"),
            layout: &self.permute_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, strengths, strength_bytes),
                binding_entry(2, &binned_strengths, strength_bytes),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_2d.tile_gather.bind_group"),
            layout: &self.spread_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, &prepared.offsets, prepared.offset_bytes),
                binding_entry(2, &binned_strengths, strength_bytes),
                binding_entry(3, bin_offsets, bin_offsets.size()),
                binding_entry(4, fine_grid, active_fine_bytes),
            ],
        });
        let permute_workgroups = point_count
            .checked_mul(active_batch)
            .map(|count| count.div_ceil(WORKGROUP_SIZE as usize))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 2D strength permutation workgroup count",
            })?;
        encode_pass(
            recorder,
            "wgpu_nufft.type1_2d.permute_strengths.pass",
            &self.permute_pipeline,
            &permute_bind_group,
            split_workgroups(permute_workgroups, self.max_workgroups_per_dimension)?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );
        let workgroups = self
            .layout
            .tile_count()
            .checked_mul(active_batch)
            .map(|count| count.div_ceil(WORKGROUP_SIZE as usize))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 2D tile-gather workgroup count",
            })?;
        let dispatch = split_workgroups(workgroups, self.max_workgroups_per_dimension)?;
        encode_pass(
            recorder,
            "wgpu_nufft.type1_2d.tile_gather.pass",
            &self.spread_pipeline,
            &bind_group,
            dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(5)),
        );
        Ok(())
    }
}

/// Copies transform-major strengths into bin order: slot `s` of every
/// vector receives the strength of the point prepared into slot `s`.
fn generate_permute_wgsl() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> prepared_starts: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> binned_strengths: array<vec2<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&binned_strengths);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= total) {{ return; }}
    let point_count = arrayLength(&prepared_starts);
    let vector_base = (index / point_count) * point_count;
    let slot = index - vector_base;
    binned_strengths[index] = strengths[vector_base + bitcast<u32>(prepared_starts[slot].w)];
}}
"#,
    )
}

fn generate_tile_gather_wgsl(
    kernel: EsKernel,
    fine_shape: [usize; DIMENSIONS],
    layout: BlockLayout2d,
) -> String {
    let position = generate_position_wgsl(fine_shape);
    let width = layout.width;
    let initialize: String = (0..TILE[1])
        .flat_map(|row| {
            (0..TILE[0])
                .map(move |column| format!("    var sum_{row}_{column} = vec2<f32>(0.0, 0.0);\n"))
        })
        .collect();
    let axis_weights = |axis: usize, name: &str, length: &str| -> String {
        (0..TILE[axis])
            .map(|cell| {
                format!(
                    "        var {name}{cell} = 0.0;
        {{
            var offset = base_{axis} + {cell};
            if (offset >= {length}) {{ offset = offset - {length}; }}
            if (offset < WIDTH_I32) {{
                {name}{cell} = es_weight((f32(offset) + offset_{axis}.x) + offset_{axis}.y);
            }}
        }}
"
                )
            })
            .collect()
    };
    let weights_0 = axis_weights(0, "wx", "FINE_0_I32");
    let weights_1 = axis_weights(1, "wy", "FINE_1_I32");
    let accumulate: String = (0..TILE[1])
        .map(|row| {
            let columns: String = (0..TILE[0])
                .map(|column| {
                    format!(
                        "                    sum_{row}_{column} = sum_{row}_{column} + value_{row} * wx{column};\n"
                    )
                })
                .collect();
            format!(
                "                {{\n                    let value_{row} = value * wy{row};\n{columns}                }}\n"
            )
        })
        .collect();
    let store: String = (0..TILE[1])
        .flat_map(|row| {
            (0..TILE[0]).map(move |column| {
                format!(
                    "    if (x0 + {column} < FINE_0_I32 && y0 + {row} < FINE_1_I32) {{ fine_grid[row_base + u32(x0 + {column}) + FINE_0 * u32(y0 + {row})] = sum_{row}_{column}; }}\n"
                )
            })
        })
        .collect();
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const BETA: f32 = {beta};
const BIN_SIDE: i32 = {BIN_SIDE}i;
const BINS_0: i32 = {bins_0}i;
const TILE_0: i32 = {tile_0}i;
const TILE_1: i32 = {tile_1}i;
const TILES_0: u32 = {tiles_0}u;
const TILE_COUNT: u32 = {tile_count}u;
const REACH_BELOW: i32 = {reach_below}i;
const REACH_ABOVE: i32 = {reach_above}i;

@group(0) @binding(0) var<storage, read> prepared_starts: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read> prepared_offsets: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<vec2<f32>>;

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

// Periodic offset of a cell from a support start, in [0, fine_length).
fn wrap_offset(offset: i32, fine_length: i32) -> i32 {{
    var wrapped = offset;
    if (wrapped < 0) {{ wrapped = wrapped + fine_length; }}
    if (wrapped >= fine_length) {{ wrapped = wrapped - fine_length; }}
    return wrapped;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= TILE_COUNT * total_vectors) {{ return; }}
    let vector_index = index / TILE_COUNT;
    let tile = index - vector_index * TILE_COUNT;
    let x0 = i32(tile % TILES_0) * TILE_0;
    let y0 = i32(tile / TILES_0) * TILE_1;
    let point_count = arrayLength(&prepared_starts);
    let strength_base = vector_index * point_count;

    let ranges_0 = bin_ranges(x0 - REACH_BELOW, x0 + TILE_0 - 1 + REACH_ABOVE, FINE_0_I32);
    let ranges_1 = bin_ranges(y0 - REACH_BELOW, y0 + TILE_1 - 1 + REACH_ABOVE, FINE_1_I32);
    let bin_rows = bin_range_length(ranges_1);
    // The x bins of one bin row hold consecutive records, split in two only
    // where the tile's reach wraps around the grid.
    let row_pieces = select(1, 2, ranges_0.last_1 >= ranges_0.first_1);
{initialize}
    // Rows and pieces in a fixed order, then records in bin order and stable
    // in-bin order. One flat loop moves to the next range inline, so lanes
    // of a warp diverge by their total record counts rather than reconverging
    // after every bin.
    var item_1 = -1;
    var piece = row_pieces - 1;
    var record = 0u;
    var end = 0u;
    loop {{
        if (record >= end) {{
            piece = piece + 1;
            if (piece == row_pieces) {{
                piece = 0;
                item_1 = item_1 + 1;
                if (item_1 == bin_rows) {{ break; }}
            }}
            let row_bin = BINS_0 * bin_range_item(ranges_1, item_1);
            var first = ranges_0.first_0;
            var last = ranges_0.last_0;
            if (piece == 1) {{
                first = ranges_0.first_1;
                last = ranges_0.last_1;
            }}
            record = bin_offsets[u32(row_bin + first)];
            end = bin_offsets[u32(row_bin + last) + 1u];
            continue;
        }}
        let start = prepared_starts[record];
        let base_0 = wrap_offset(x0 - start.x, FINE_0_I32);
        let base_1 = wrap_offset(y0 - start.y, FINE_1_I32);
        record = record + 1u;
        // The support [start, start + width) meets the tile when some tile
        // cell's periodic offset from the start lies below the width.
        if ((base_0 >= WIDTH_I32 && base_0 <= FINE_0_I32 - TILE_0)
            || (base_1 >= WIDTH_I32 && base_1 <= FINE_1_I32 - TILE_1)) {{
            continue;
        }}
        let slot = record - 1u;
        let offset_0 = prepared_offsets[2u * slot];
        let offset_1 = prepared_offsets[2u * slot + 1u];
{weights_0}{weights_1}        let value = strengths[strength_base + slot];
{accumulate}    }}

    let row_base = vector_index * FINE_COUNT;
{store}}}
"#,
        beta = format_wgsl_f32(kernel.beta() as f32),
        bins_0 = layout.bins[0],
        tile_0 = TILE[0],
        tile_1 = TILE[1],
        tiles_0 = layout.tiles[0],
        tile_count = layout.tile_count(),
        reach_below = width.div_ceil(2),
        reach_above = width / 2,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl_validation::assert_valid_wgsl;

    #[test]
    fn layout_requires_room_for_an_unambiguous_tile_reach() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let limits = wgpu::Limits::default();
        assert!(BlockLayout2d::for_grid(kernel, [1024, 1024], &limits).is_some());
        // Width 7 needs at least TILE + 2 * width = 18 cells per axis.
        assert!(BlockLayout2d::for_grid(kernel, [18, 18], &limits).is_some());
        assert!(BlockLayout2d::for_grid(kernel, [17, 1024], &limits).is_none());
        assert!(BlockLayout2d::for_grid(kernel, [1024, 16], &limits).is_none());
    }

    #[test]
    fn tile_gather_writes_each_cell_once_without_atomics() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let layout =
            BlockLayout2d::for_grid(kernel, [1024, 768], &wgpu::Limits::default()).unwrap();
        let source = generate_tile_gather_wgsl(kernel, [1024, 768], layout);
        assert_valid_wgsl(&source);
        assert!(!source.contains("atomic"));
        assert!(!source.contains("var<workgroup>"));
        assert_eq!(
            source.matches("fine_grid[row_base + ").count(),
            TILE[0] * TILE[1]
        );
        assert!(source.contains("let value = strengths[strength_base + slot];"));
        assert_valid_wgsl(&generate_permute_wgsl());
    }
}
