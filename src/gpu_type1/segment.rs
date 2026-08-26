//! Deterministic output-stationary 1D type-1 spreading over coarse point bins.
//!
//! [`PointBins`] records a stable coarse-bin order of the points and their
//! prepared support starts and df64 offsets. [`SegmentGather1d::encode_spread`]
//! then lets every invocation own a segment of consecutive fine-grid cells:
//! it walks the few bins that can reach the segment, in a fixed order, and
//! adds every point of those bins, in bin order, into register sums. Each cell
//! therefore sums its contributions in a fixed order, and every fine-grid cell
//! is written exactly once: no float atomics, no read-modify-write, and no
//! clearing pass. Prepared offsets turn every kernel weight into
//! `es_weight((j + hi) + lo)`, and a permutation pass first copies the
//! strengths into bin order, so the gather reads them contiguously.
//!
//! A dense cluster would make one invocation walk a huge reach serially, so
//! a segment whose reach holds more than [`HEAVY_SEGMENT_RECORDS`] records is
//! left to a second pass, in which a whole workgroup shares the segment:
//! every invocation takes every [`HEAVY_WORKGROUP_SIZE`]-th record in reach
//! order, and the partial sums meet in a fixed-order tree, so those cells
//! stay deterministic too.

use std::sync::Mutex;

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_point_bins::{PointBinOrder, PointBins};
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::GpuProfileQueryWriter;
use crate::gpu_recorder::GpuRecorder;
use crate::kernel::EsKernel;

use super::{
    binding_entry, create_compute_pipeline, encode_pass, format_wgsl_f32,
    generate_binned_position_wgsl, WORKGROUP_SIZE,
};

/// Cells per coarse point bin.
const BIN_SIDE: usize = 4;
/// Consecutive fine-grid cells owned by one invocation. 4-cell segments
/// over 4-cell bins gathered 21% faster than 8-cell segments at 1M points;
/// 16-cell segments were slower still.
const SEGMENT: usize = 4;
/// Records in reach above which a segment goes to the heavy pass. One
/// invocation spends about 0.12 us per record.
const HEAVY_SEGMENT_RECORDS: u32 = 4096;
/// Invocations sharing one heavy segment.
const HEAVY_WORKGROUP_SIZE: u32 = 256;
/// Workgroups of the heavy pass, which stride over the heavy segments.
const HEAVY_WORKGROUPS: u32 = 128;

/// Static geometry of the segment gather for one plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SegmentLayout1d {
    width: usize,
    fine_length: usize,
    segments: usize,
}

impl SegmentLayout1d {
    /// Returns the layout when the grid can run the segment gather, or `None`
    /// so the caller keeps the per-cell gather.
    pub(super) fn for_grid(kernel: EsKernel, fine_length: usize) -> Option<Self> {
        let width = kernel.width();
        // A segment's reach must not wrap onto itself, both wrapped parts of a
        // reach must fall into distinct bins, and every relative support
        // start must have a unique unwrapped representative. As in 3D, a point
        // whose support starts in `[o - w + 1, o + S - 1]` has its folded cell
        // in `[o - ceil(w/2), o + S - 1 + floor(w/2)]`.
        let minimum = (2 * SEGMENT)
            .max(SEGMENT + 2 * width)
            .max(SEGMENT + width + BIN_SIDE + 1);
        if fine_length < minimum {
            return None;
        }
        u32::try_from(fine_length.div_ceil(BIN_SIDE).checked_add(1)?).ok()?;
        Some(Self {
            width,
            fine_length,
            segments: fine_length.div_ceil(SEGMENT),
        })
    }
}

pub(super) struct SegmentGather1d {
    layout: SegmentLayout1d,
    bins: PointBins,
    permute_pipeline: wgpu::ComputePipeline,
    permute_layout: wgpu::BindGroupLayout,
    gather_pipeline: wgpu::ComputePipeline,
    gather_layout: wgpu::BindGroupLayout,
    heavy_pipeline: wgpu::ComputePipeline,
    heavy_layout: wgpu::BindGroupLayout,
    /// A count and the segments left to the heavy pass; the permutation pass
    /// resets the count.
    heavy_segments: wgpu::Buffer,
    /// Strengths in bin order, grown on demand.
    binned_strengths: Mutex<Option<wgpu::Buffer>>,
    device: wgpu::Device,
    max_workgroups_per_dimension: u32,
}

impl SegmentGather1d {
    pub(super) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        layout: SegmentLayout1d,
    ) -> Result<Self> {
        let bins = PointBins::new(
            device,
            kernel,
            &[layout.fine_length],
            &[BIN_SIDE],
            PointBinOrder::Stable,
            &generate_binned_position_wgsl(layout.fine_length),
        )?;
        let permute_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.permute_strengths",
            &generate_permute_wgsl(),
        );
        let gather_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.segment_gather",
            &generate_segment_gather_wgsl(kernel, layout),
        );
        let heavy_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.heavy_segment_gather",
            &generate_heavy_segment_gather_wgsl(kernel, layout),
        );
        let heavy_segments = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.heavy_segments"),
            size: (layout.segments as u64 + 1) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(Self {
            layout,
            bins,
            permute_layout: permute_pipeline.get_bind_group_layout(0),
            permute_pipeline,
            gather_layout: gather_pipeline.get_bind_group_layout(0),
            gather_pipeline,
            heavy_layout: heavy_pipeline.get_bind_group_layout(0),
            heavy_pipeline,
            heavy_segments,
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
                        label: Some("wgpu_nufft.type1.binned_strengths"),
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
            label: Some("wgpu_nufft.type1.permute_strengths.bind_group"),
            layout: &self.permute_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, strengths, strength_bytes),
                binding_entry(2, &binned_strengths, strength_bytes),
                binding_entry(3, &self.heavy_segments, self.heavy_segments.size()),
            ],
        });
        let gather_entries = [
            binding_entry(0, &prepared.starts, prepared.start_bytes),
            binding_entry(1, &prepared.offsets, prepared.offset_bytes),
            binding_entry(2, &binned_strengths, strength_bytes),
            binding_entry(3, bin_offsets, bin_offsets.size()),
            binding_entry(4, fine_grid, active_fine_bytes),
            binding_entry(5, &self.heavy_segments, self.heavy_segments.size()),
        ];
        let gather_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.segment_gather.bind_group"),
            layout: &self.gather_layout,
            entries: &gather_entries,
        });
        let heavy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.heavy_segment_gather.bind_group"),
            layout: &self.heavy_layout,
            entries: &gather_entries,
        });
        let permute_workgroups = point_count
            .checked_mul(active_batch)
            .map(|count| count.div_ceil(WORKGROUP_SIZE as usize))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 strength permutation workgroup count",
            })?;
        let gather_workgroups = self
            .layout
            .segments
            .checked_mul(active_batch)
            .map(|count| count.div_ceil(WORKGROUP_SIZE as usize))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 segment-gather workgroup count",
            })?;
        encode_pass(
            recorder,
            "wgpu_nufft.type1.permute_strengths.pass",
            &self.permute_pipeline,
            &permute_bind_group,
            split_workgroups(permute_workgroups, self.max_workgroups_per_dimension)?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );
        encode_pass(
            recorder,
            "wgpu_nufft.type1.segment_gather.pass",
            &self.gather_pipeline,
            &gather_bind_group,
            split_workgroups(gather_workgroups, self.max_workgroups_per_dimension)?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );
        encode_pass(
            recorder,
            "wgpu_nufft.type1.heavy_segment_gather.pass",
            &self.heavy_pipeline,
            &heavy_bind_group,
            (HEAVY_WORKGROUPS, 1, 1),
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(5)),
        );
        Ok(())
    }
}

/// Copies transform-major strengths into bin order: slot `s` of every
/// vector receives the strength of the point prepared into slot `s`. It also
/// resets the heavy-segment count for the gather that follows.
fn generate_permute_wgsl() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> prepared_starts: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> binned_strengths: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> heavy_segments: array<u32>;

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
    if (index == 0u) {{ heavy_segments[0] = 0u; }}
    let point_count = arrayLength(&prepared_starts);
    let vector_base = (index / point_count) * point_count;
    let slot = index - vector_base;
    binned_strengths[index] = strengths[vector_base + bitcast<u32>(prepared_starts[slot].w)];
}}
"#,
    )
}

/// Constants, bindings 0-4 and helper functions shared by the segment
/// gathers.
fn segment_gather_common(kernel: EsKernel, layout: SegmentLayout1d) -> String {
    let width = layout.width;
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const HEAVY_WORKGROUP_SIZE: u32 = {HEAVY_WORKGROUP_SIZE}u;
const HEAVY_SEGMENT_RECORDS: u32 = {HEAVY_SEGMENT_RECORDS}u;
const FINE: u32 = {fine}u;
const FINE_I32: i32 = {fine}i;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const BETA: f32 = {beta};
const BIN_SIDE: i32 = {BIN_SIDE}i;
const SEGMENT: i32 = {SEGMENT}i;
const SEGMENTS: u32 = {segments}u;
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

// Bins covering cells [low, high] of the periodic axis as up to two ranges.
struct BinRanges {{
    first_0: i32,
    last_0: i32,
    first_1: i32,
    last_1: i32,
}}

fn bin_ranges(low: i32, high: i32) -> BinRanges {{
    if (low < 0) {{
        return BinRanges((low + FINE_I32) / BIN_SIDE, (FINE_I32 - 1) / BIN_SIDE,
            0, high / BIN_SIDE);
    }}
    if (high >= FINE_I32) {{
        return BinRanges(low / BIN_SIDE, (FINE_I32 - 1) / BIN_SIDE,
            0, (high - FINE_I32) / BIN_SIDE);
    }}
    return BinRanges(low / BIN_SIDE, high / BIN_SIDE, 0, -1);
}}

// Records [start, end) of piece `piece` of the reach.
fn piece_records(ranges: BinRanges, piece: i32) -> vec2<u32> {{
    var first = ranges.first_0;
    var last = ranges.last_0;
    if (piece == 1) {{
        first = ranges.first_1;
        last = ranges.last_1;
    }}
    return vec2<u32>(bin_offsets[u32(first)], bin_offsets[u32(last) + 1u]);
}}
"#,
        fine = layout.fine_length,
        beta = format_wgsl_f32(kernel.beta() as f32),
        segments = layout.segments,
        reach_below = width.div_ceil(2),
        reach_above = width / 2,
    )
}

/// Statements locating segment `segment` of vector `vector_index` and its
/// reach.
const SEGMENT_REACH: &str = r#"let x0 = i32(segment) * SEGMENT;
let strength_base = vector_index * arrayLength(&prepared_starts);
let ranges = bin_ranges(x0 - REACH_BELOW, x0 + SEGMENT - 1 + REACH_ABOVE);
// The bins in reach hold consecutive records, split in two only where the
// segment's reach wraps around the grid.
let pieces = select(1, 2, ranges.last_1 >= ranges.first_1);
"#;

/// Indents every nonempty line of `block` by `spaces`.
fn indent(block: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    block
        .lines()
        .map(|line| {
            if line.is_empty() {
                "\n".to_string()
            } else {
                format!("{pad}{line}\n")
            }
        })
        .collect()
}

fn segment_sums() -> String {
    (0..SEGMENT)
        .map(|cell| format!("var sum_{cell} = vec2<f32>(0.0, 0.0);\n"))
        .collect()
}

/// Adds the point prepared into `record` to the segment's register sums.
fn segment_contribution() -> String {
    let accumulate: String = (0..SEGMENT)
        .map(|cell| {
            format!(
                "{{
    var offset = base + {cell};
    if (offset >= FINE_I32) {{ offset = offset - FINE_I32; }}
    if (offset < WIDTH_I32) {{
        sum_{cell} = sum_{cell} + value * es_weight((f32(offset) + support.x) + support.y);
    }}
}}
"
            )
        })
        .collect();
    format!(
        "var base = x0 - prepared_starts[record].x;
if (base < 0) {{ base = base + FINE_I32; }}
if (base >= FINE_I32) {{ base = base - FINE_I32; }}
let support = prepared_offsets[record];
let value = strengths[strength_base + record];
{accumulate}"
    )
}

fn generate_segment_gather_wgsl(kernel: EsKernel, layout: SegmentLayout1d) -> String {
    let common = segment_gather_common(kernel, layout);
    let reach = indent(SEGMENT_REACH, 4);
    let sums = indent(&segment_sums(), 4);
    let contribution = indent(&segment_contribution(), 12);
    let store: String = (0..SEGMENT)
        .map(|cell| {
            format!(
                "    if (x0 + {cell} < FINE_I32) {{ fine_grid[vector_base + u32(x0 + {cell})] = sum_{cell}; }}\n"
            )
        })
        .collect();
    format!(
        r#"{common}
@group(0) @binding(5) var<storage, read_write> heavy_segments: array<atomic<u32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total_vectors = arrayLength(&fine_grid) / FINE;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= SEGMENTS * total_vectors) {{ return; }}
    let vector_index = index / SEGMENTS;
    let segment = index - vector_index * SEGMENTS;
{reach}
{sums}
    // Pieces in a fixed order, then records in bin order and stable in-bin
    // order. Looping over whole pieces instead of single bins keeps the lanes
    // of a warp from reconverging after every bin.
    var reach_records = 0u;
    for (var piece = 0; piece < pieces; piece = piece + 1) {{
        let records = piece_records(ranges, piece);
        // A dense cluster would serialize this invocation: once the reach
        // passes the limit, the segment goes to the heavy pass, which writes
        // its cells for every vector. The running total costs no reads.
        reach_records = reach_records + (records.y - records.x);
        if (reach_records > HEAVY_SEGMENT_RECORDS) {{
            if (vector_index == 0u) {{
                atomicStore(&heavy_segments[atomicAdd(&heavy_segments[0], 1u) + 1u], segment);
            }}
            return;
        }}
        for (var record = records.x; record < records.y; record = record + 1u) {{
{contribution}        }}
    }}

    let vector_base = vector_index * FINE;
{store}}}
"#,
    )
}

/// The heavy pass: every workgroup strides over the (heavy segment, vector)
/// pairs the segment gather left. Each invocation sums every
/// [`HEAVY_WORKGROUP_SIZE`]-th record of the segment's reach, in reach
/// order, and a fixed-order tree in workgroup memory adds the partial sums.
fn generate_heavy_segment_gather_wgsl(kernel: EsKernel, layout: SegmentLayout1d) -> String {
    let common = segment_gather_common(kernel, layout);
    let reach = indent(SEGMENT_REACH, 8);
    let sums = indent(&segment_sums(), 8);
    let contribution = indent(&segment_contribution(), 16);
    let publish: String = (0..SEGMENT)
        .map(|cell| {
            format!("        partials[{cell}u * HEAVY_WORKGROUP_SIZE + lid.x] = sum_{cell};\n")
        })
        .collect();
    let add: String = (0..SEGMENT)
        .map(|cell| {
            format!(
                "                partials[{cell}u * HEAVY_WORKGROUP_SIZE + lid.x] = partials[{cell}u * HEAVY_WORKGROUP_SIZE + lid.x]
                    + partials[{cell}u * HEAVY_WORKGROUP_SIZE + lid.x + half];\n"
            )
        })
        .collect();
    let store: String = (0..SEGMENT)
        .map(|cell| {
            format!(
                "            if (x0 + {cell} < FINE_I32) {{ fine_grid[vector_base + u32(x0 + {cell})] = partials[{cell}u * HEAVY_WORKGROUP_SIZE]; }}\n"
            )
        })
        .collect();
    format!(
        r#"{common}
@group(0) @binding(5) var<storage, read> heavy_segments: array<u32>;

var<workgroup> heavy_count: u32;
var<workgroup> partials: array<vec2<f32>, {partial_count}>;

@compute @workgroup_size({HEAVY_WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    if (lid.x == 0u) {{ heavy_count = heavy_segments[0]; }}
    let segment_total = workgroupUniformLoad(&heavy_count);
    let items = segment_total * (arrayLength(&fine_grid) / FINE);
    for (var item = wid.x; item < items; item = item + nwg.x) {{
        let vector_index = item / segment_total;
        let segment = heavy_segments[1u + item - vector_index * segment_total];
{reach}
{sums}
        var ordinal = 0u;
        for (var piece = 0; piece < pieces; piece = piece + 1) {{
            let records = piece_records(ranges, piece);
            // This invocation's records are those whose ordinal in the reach
            // is congruent to its index.
            let skipped = (lid.x + HEAVY_WORKGROUP_SIZE - ordinal % HEAVY_WORKGROUP_SIZE)
                % HEAVY_WORKGROUP_SIZE;
            for (var record = records.x + skipped; record < records.y;
                record = record + HEAVY_WORKGROUP_SIZE) {{
{contribution}            }}
            ordinal = ordinal + (records.y - records.x);
        }}

        let vector_base = vector_index * FINE;
{publish}        workgroupBarrier();
        for (var half = HEAVY_WORKGROUP_SIZE / 2u; half > 0u; half = half / 2u) {{
            if (lid.x < half) {{
{add}            }}
            workgroupBarrier();
        }}
        if (lid.x == 0u) {{
{store}        }}
        workgroupBarrier();
    }}
}}
"#,
        partial_count = HEAVY_WORKGROUP_SIZE as usize * SEGMENT,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl_validation::assert_valid_wgsl;

    #[test]
    fn layout_requires_room_for_an_unambiguous_segment_reach() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        assert!(SegmentLayout1d::for_grid(kernel, 1 << 20).is_some());
        // Width 7 needs at least SEGMENT + 2 * width = 18 cells.
        assert!(SegmentLayout1d::for_grid(kernel, 18).is_some());
        assert!(SegmentLayout1d::for_grid(kernel, 17).is_none());
    }

    #[test]
    fn segment_gather_writes_each_cell_once_without_atomics() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let layout = SegmentLayout1d::for_grid(kernel, 1000).unwrap();
        let source = generate_segment_gather_wgsl(kernel, layout);
        assert_valid_wgsl(&source);
        assert_valid_wgsl(&generate_permute_wgsl());
        // The only atomics count and list heavy segments.
        assert_eq!(source.matches("atomic").count(), 3);
        assert!(!source.contains("var<workgroup>"));
        assert_eq!(source.matches("fine_grid[vector_base + ").count(), SEGMENT);
        let heavy = generate_heavy_segment_gather_wgsl(kernel, layout);
        assert_valid_wgsl(&heavy);
        assert!(!heavy.contains("atomic"));
        assert_eq!(heavy.matches("fine_grid[vector_base + ").count(), SEGMENT);
    }
}
