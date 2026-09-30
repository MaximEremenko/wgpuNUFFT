//! Coarse-bin ordering and per-point preparation for the rank-generic paths,
//! in any rank and precision.
//!
//! [`NdPointBins::encode`] folds every coordinate, counts the points of
//! every coarse bin with integer atomics (keeping each point's bin and its
//! rank among the points of that bin), scans the counts into bin offsets,
//! scatters point indices into their bins, and prepares every slot: the
//! support start per axis and the original point index (`i32` words, `D + 1`
//! per slot) and per axis the offset `start - position` (a df64 high/low
//! `f32` pair, or an `f64`). Consumers evaluate the distance from support cell
//! `start + j` as `j + offset` and never fold again.
//!
//! Bins hold support starts, wrapped onto the grid, rather than the cells of
//! the positions: the records whose support reaches a block of cells are then
//! exactly those of the bins that hold the kernel width of starts below the
//! block, and the type-1 spreader stages no others.
//!
//! [`NdBinOrder::Stable`] restores original point order inside every bin, as
//! the type-1 spreader needs for deterministic sums: [`LargeBinSort`] sorts
//! bins above [`SMALL_BIN`] points, and preparation ranks the points of the
//! smaller bins. Its preparation also keeps every start inside its point's
//! bin, which the count pass derived in a separate shader. [`NdBinOrder::Grouped`]
//! keeps the order the scatter left, which suits type-2 interpolation, whose
//! per-point results do not depend on it.
//!
//! A coordinate the fold cannot bring onto the grid, NaN included, lands in
//! the first bin with support start zero and offsets past the kernel support,
//! so every consumer reads inside the grid and weighs it by zero.

use crate::error::{NufftError, Result};
use crate::gpu_bin_sort::{LargeBinSort, SMALL_BIN};
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_nd_wgsl::{linear_index, position_wgsl, NdWgsl};
use crate::gpu_point_bins::GrowOnlyBuffer;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::GpuProfileQueryWriter;
use crate::gpu_recorder::GpuRecorder;
use crate::gpu_scan::GpuExclusiveScanU32;
use crate::gpu_type1_3d::{
    binding_entry, checked_buffer_size, create_compute_pipeline, encode_pass,
    validate_binding_limit, validate_buffer_limit,
};
use crate::kernel::EsKernel;
use wgpu_fft::FftPrecision;

const WORKGROUP_SIZE: u32 = 64;
const U32_BYTES: u64 = 4;
/// `vec2<u32>`: a point's bin and its rank among the points of that bin.
const POINT_SLOT_BYTES: u64 = 8;

/// How [`NdPointBins::encode`] orders the points inside each bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NdBinOrder {
    /// Scheduling-dependent order inside every bin.
    Grouped,
    /// Original point order inside every bin.
    Stable,
}

/// Prepared point data in bin order.
pub(crate) struct NdPreparedPoints {
    /// `D + 1` `i32` words per slot: the support start per axis, then the
    /// original point index.
    pub(crate) starts: wgpu::Buffer,
    pub(crate) start_bytes: u64,
    /// `D` offsets `start - position` per slot.
    pub(crate) offsets: wgpu::Buffer,
    pub(crate) offset_bytes: u64,
}

pub(crate) struct NdPointBins {
    dimensions: usize,
    types: NdWgsl,
    /// One count per bin plus a trailing zero, so the scan also yields the
    /// total, and behind them the flag stable counting sets when a bin holds
    /// more than [`SMALL_BIN`] points.
    bin_counts: wgpu::Buffer,
    bin_offsets: wgpu::Buffer,
    point_slots: GrowOnlyBuffer,
    count_pipeline: wgpu::ComputePipeline,
    count_layout: wgpu::BindGroupLayout,
    prefix_scan: GpuExclusiveScanU32,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    sort: Option<LargeBinSort>,
    prepare_pipeline: wgpu::ComputePipeline,
    prepare_layout: wgpu::BindGroupLayout,
    sorted_indices: GrowOnlyBuffer,
    prepared_starts: GrowOnlyBuffer,
    prepared_offsets: GrowOnlyBuffer,
    max_workgroups_per_dimension: u32,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
}

impl NdPointBins {
    /// Bins the points of a `fine_shape` grid into `bin_shape`-cell bins,
    /// axis zero fastest.
    pub(crate) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        fine_shape: &[usize],
        bin_shape: &[usize],
        order: NdBinOrder,
        precision: FftPrecision,
    ) -> Result<Self> {
        let dimensions = fine_shape.len();
        assert!(
            dimensions >= 1 && bin_shape.len() == dimensions,
            "one bin side per axis"
        );
        let limits = device.limits();
        let bins = fine_shape
            .iter()
            .zip(bin_shape)
            .map(|(&length, &side)| length.div_ceil(side))
            .collect::<Vec<_>>();
        let bin_count = bins
            .iter()
            .try_fold(1usize, |count, &n| count.checked_mul(n))
            .ok_or(NufftError::LengthOverflow {
                context: "rank-generic point-bin count",
            })?;
        let entry_count = bin_count.checked_add(1).ok_or(NufftError::LengthOverflow {
            context: "rank-generic point-bin offset count",
        })?;
        u32::try_from(entry_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic point-bin shader index space",
        })?;
        let entry_bytes = checked_buffer_size(
            "rank-generic point-bin counts and offsets",
            entry_count,
            U32_BYTES,
        )?;
        // The counts also carry the large-bin flag behind their trailing zero.
        let count_bytes = entry_bytes + U32_BYTES;
        validate_binding_limit(
            "rank-generic point-bin counts and offsets",
            count_bytes,
            limits.max_storage_buffer_binding_size,
        )?;
        validate_buffer_limit(
            "rank-generic point-bin counts and offsets",
            count_bytes,
            limits.max_buffer_size,
        )?;
        let bin_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.nd_bins.bin_counts"),
            size: count_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bin_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.nd_bins.bin_offsets"),
            size: entry_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let stable = order == NdBinOrder::Stable;
        let types = NdWgsl::new(precision);
        let position = position_wgsl(fine_shape, kernel, precision);
        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.nd_bins.count",
            &generate_count_wgsl(types, bin_shape, &bins, stable, &position),
        );
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.nd_bins.scatter",
            &generate_scatter_wgsl(),
        );
        let prepare_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.nd_bins.prepare",
            &generate_prepare_wgsl(types, bin_shape, &bins, stable, &position),
        );
        Ok(Self {
            dimensions,
            types,
            bin_counts,
            bin_offsets,
            point_slots: GrowOnlyBuffer::new(device, "wgpu_nufft.nd_bins.point_slots"),
            count_layout: count_pipeline.get_bind_group_layout(0),
            count_pipeline,
            prefix_scan: GpuExclusiveScanU32::new(device, entry_count)?,
            scatter_layout: scatter_pipeline.get_bind_group_layout(0),
            scatter_pipeline,
            sort: stable.then(|| LargeBinSort::new(device)),
            prepare_layout: prepare_pipeline.get_bind_group_layout(0),
            prepare_pipeline,
            sorted_indices: GrowOnlyBuffer::new(device, "wgpu_nufft.nd_bins.sorted_indices"),
            prepared_starts: GrowOnlyBuffer::new(device, "wgpu_nufft.nd_bins.prepared_starts"),
            prepared_offsets: GrowOnlyBuffer::new(device, "wgpu_nufft.nd_bins.prepared_offsets"),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    /// Bin offsets of the last [`Self::encode`], one entry per bin plus a
    /// terminal total, in axis-zero-fastest bin order.
    pub(crate) fn bin_offsets(&self) -> &wgpu::Buffer {
        &self.bin_offsets
    }

    fn scratch(
        &self,
        buffer: &GrowOnlyBuffer,
        label: &'static str,
        elements: usize,
        bytes_per_element: u64,
    ) -> Result<(wgpu::Buffer, u64)> {
        let bytes = checked_buffer_size(label, elements, bytes_per_element)?;
        validate_binding_limit(label, bytes, self.max_storage_binding_bytes)?;
        validate_buffer_limit(label, bytes, self.max_buffer_bytes)?;
        Ok((buffer.get(bytes), bytes))
    }

    /// Prepared data for the `point_count` points of the last
    /// [`Self::encode`], in bin order.
    pub(crate) fn prepared(&self, point_count: usize) -> Result<NdPreparedPoints> {
        let start_words =
            point_count
                .checked_mul(self.dimensions + 1)
                .ok_or(NufftError::LengthOverflow {
                    context: "rank-generic prepared point starts",
                })?;
        let offset_count =
            point_count
                .checked_mul(self.dimensions)
                .ok_or(NufftError::LengthOverflow {
                    context: "rank-generic prepared point offsets",
                })?;
        u32::try_from(start_words).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic prepared point shader index space",
        })?;
        let (starts, start_bytes) = self.scratch(
            &self.prepared_starts,
            "rank-generic prepared point starts",
            start_words,
            U32_BYTES,
        )?;
        let (offsets, offset_bytes) = self.scratch(
            &self.prepared_offsets,
            "rank-generic prepared point offsets",
            offset_count,
            self.types.offset_bytes(),
        )?;
        Ok(NdPreparedPoints {
            starts,
            start_bytes,
            offsets,
            offset_bytes,
        })
    }

    /// Records the bin order and prepared data of `point_count > 0` points.
    /// Both stay valid for consumers until the next call; executions must
    /// keep queue order.
    ///
    /// With profiling, stable mode writes the start marker and the ends of
    /// the count, scan, scatter, and sort-and-prepare stages at offsets 0-4
    /// of the type-1 layout. Grouped mode writes the start marker at offset 0
    /// and the end of the last pass at offset 1.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        point_count: usize,
        points: &wgpu::Buffer,
        point_bytes: u64,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        #[cfg(feature = "gpu-profiling")]
        let stable = self.sort.is_some();
        debug_assert!(point_count > 0);
        let point_count_u32 =
            u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic binned point count",
            })?;
        let point_dispatch = split_workgroups(
            point_count_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;
        let (point_slots, slot_bytes) = self.scratch(
            &self.point_slots,
            "rank-generic point-bin slots",
            point_count,
            POINT_SLOT_BYTES,
        )?;
        let (sorted_indices, index_bytes) = self.scratch(
            &self.sorted_indices,
            "rank-generic sorted point indices",
            point_count,
            U32_BYTES,
        )?;
        let prepared = self.prepared(point_count)?;
        let offset_bytes = self.bin_offsets.size();

        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_bins.count.bind_group"),
            layout: &self.count_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, &self.bin_counts, self.bin_counts.size()),
                binding_entry(2, &point_slots, slot_bytes),
            ],
        });
        #[cfg(feature = "gpu-profiling")]
        profile.encode_start_marker(recorder);
        recorder.clear_buffer(&self.bin_counts, 0, None);
        encode_pass(
            recorder,
            "wgpu_nufft.nd_bins.count.pass",
            &self.count_pipeline,
            &count_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            stable
                .then(|| profile.timestamp_writes(None, Some(1)))
                .flatten(),
        );
        self.prefix_scan
            .encode(device, recorder, &self.bin_counts, &self.bin_offsets)?;
        #[cfg(feature = "gpu-profiling")]
        if stable {
            profile.encode_marker(recorder, None, Some(2));
        }

        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_bins.scatter.bind_group"),
            layout: &self.scatter_layout,
            entries: &[
                binding_entry(0, &point_slots, slot_bytes),
                binding_entry(1, &self.bin_offsets, offset_bytes),
                binding_entry(2, &sorted_indices, index_bytes),
            ],
        });
        encode_pass(
            recorder,
            "wgpu_nufft.nd_bins.scatter.pass",
            &self.scatter_pipeline,
            &scatter_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            stable
                .then(|| profile.timestamp_writes(None, Some(3)))
                .flatten(),
        );
        if let Some(sort) = &self.sort {
            // The count pass left the large-bin flag in the last count word.
            sort.encode(
                device,
                recorder,
                &self.bin_offsets,
                offset_bytes,
                &self.bin_counts,
                &sorted_indices,
                point_count,
            )?;
        }
        let mut prepare_entries = vec![
            binding_entry(0, points, point_bytes),
            binding_entry(1, &sorted_indices, index_bytes),
            binding_entry(2, &prepared.starts, prepared.start_bytes),
            binding_entry(3, &prepared.offsets, prepared.offset_bytes),
        ];
        if self.sort.is_some() {
            // Stable mode ranks the points of small bins while preparing.
            prepare_entries.push(binding_entry(4, &point_slots, slot_bytes));
            prepare_entries.push(binding_entry(5, &self.bin_offsets, offset_bytes));
        }
        let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_bins.prepare.bind_group"),
            layout: &self.prepare_layout,
            entries: &prepare_entries,
        });
        encode_pass(
            recorder,
            "wgpu_nufft.nd_bins.prepare.pass",
            &self.prepare_pipeline,
            &prepare_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(if stable { 4 } else { 1 })),
        );
        Ok(())
    }
}

fn generate_count_wgsl(
    types: NdWgsl,
    bin_shape: &[usize],
    bins: &[usize],
    stable: bool,
    position: &str,
) -> String {
    let dimensions = bin_shape.len();
    let mut constants = String::new();
    for (axis, side) in bin_shape.iter().enumerate() {
        constants.push_str(&format!("const BIN_SIDE_{axis}: u32 = {side}u;\n"));
    }
    for (axis, count) in bins.iter().enumerate() {
        constants.push_str(&format!("const BINS_{axis}: u32 = {count}u;\n"));
    }
    let loads: String = (0..dimensions)
        .map(|axis| format!("    let coordinate_{axis} = points[point_base + {axis}u];\n"))
        .collect();
    let in_reach = (0..dimensions)
        .map(|axis| types.in_reach(&format!("coordinate_{axis}")))
        .collect::<Vec<_>>()
        .join(" && ");
    // The same start the prepare pass derives, wrapped onto the grid.
    let starts: String = (0..dimensions)
        .map(|axis| {
            format!(
                "    let start_{axis} = wrapped_start(select(0, {start}, in_reach), FINE_{axis}_I32);\n",
                start = types.start(&types.fold(axis, &format!("coordinate_{axis}"))),
            )
        })
        .collect();
    let terms = (0..dimensions)
        .map(|axis| format!("start_{axis} / BIN_SIDE_{axis}"))
        .collect::<Vec<_>>();
    let lengths = (0..dimensions)
        .map(|axis| format!("BINS_{axis}"))
        .collect::<Vec<_>>();
    let bin = linear_index(&terms, &lengths);
    let record = if stable {
        // The point that overfills a small bin flags the large-bin sort,
        // behind the counts and their trailing zero; points at every later
        // power-of-two rank raise the flag, so it bounds the largest bin.
        "    let rank = atomicAdd(&bin_counts[bin], 1u);
    point_slots[point_index] = vec2<u32>(bin, rank);
    if (rank >= SMALL_BIN && (rank & (rank - 1u)) == 0u) {
        atomicMax(&bin_counts[arrayLength(&bin_counts) - 1u], rank);
    }
"
    } else {
        "    point_slots[point_index] = vec2<u32>(bin, atomicAdd(&bin_counts[bin], 1u));\n"
    };
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMS: u32 = {dimensions}u;
const SMALL_BIN: u32 = {SMALL_BIN}u;
{constants}
@group(0) @binding(0) var<storage, read> points: array<{coordinate}>;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> point_slots: array<vec2<u32>>;

// A support start wrapped onto the grid, clamped so that even a coordinate
// outside the contract lands in a bin.
fn wrapped_start(start: i32, fine_length: i32) -> u32 {{
    return u32(clamp(select(start, start + fine_length, start < 0), 0, fine_length - 1));
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&point_slots);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * DIMS;
{loads}    let in_reach = {in_reach};
{starts}    let bin = {bin};
{record}}}
"#,
        coordinate = types.coordinate_type(),
    );
    types.with_library(&format!("{position}\n{entry}"))
}

fn generate_scatter_wgsl() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> point_slots: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&point_slots);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let slot = point_slots[point_index];
    sorted_indices[bin_offsets[slot.x] + slot.y] = point_index;
}}
"#,
    )
}

/// Prepares binned points in slot order. In stable mode the scatter left
/// small bins in atomic order: each of their points then takes the slot of
/// its rank by original index within the bin, while [`LargeBinSort`] already
/// ordered the larger bins, whose points keep their slots.
///
/// Every coordinate is loaded once, before any store, and the fallback for a
/// coordinate out of the fold's reach is selected without branches.
fn generate_prepare_wgsl(
    types: NdWgsl,
    bin_shape: &[usize],
    bins: &[usize],
    stable: bool,
    position: &str,
) -> String {
    let dimensions = bin_shape.len();
    let last = dimensions - 1;
    let bin_axes: String = (0..dimensions)
        .map(|axis| {
            if axis == last {
                format!("    let bin_{axis} = i32(bin_rest);\n")
            } else {
                format!(
                    "    let bin_{axis} = i32(bin_rest % BINS_{axis});\n    bin_rest = bin_rest / BINS_{axis};\n"
                )
            }
        })
        .collect();
    let (rank_bindings, rank) = if stable {
        (
            "@group(0) @binding(4) var<storage, read> point_slots: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> bin_offsets: array<u32>;
",
            format!(
                "    var slot = record;
    let bin = point_slots[point_index].x;
    let bin_start = bin_offsets[bin];
    let bin_end = bin_offsets[bin + 1u];
    if (bin_end - bin_start <= {SMALL_BIN}u) {{
        var rank = 0u;
        for (var other = bin_start; other < bin_end; other = other + 1u) {{
            if (sorted_indices[other] < point_index) {{ rank = rank + 1u; }}
        }}
        slot = bin_start + rank;
    }}
    var bin_rest = bin;
{bin_axes}"
            ),
        )
    } else {
        ("", "    let slot = record;\n".to_owned())
    };
    let loads: String = (0..dimensions)
        .map(|axis| format!("    let coordinate_{axis} = points[point_base + {axis}u];\n"))
        .collect();
    let in_reach = (0..dimensions)
        .map(|axis| types.in_reach(&format!("coordinate_{axis}")))
        .collect::<Vec<_>>()
        .join(" && ");
    let axes: String = (0..dimensions)
        .map(|axis| {
            let position_value = types.fold(axis, &format!("coordinate_{axis}"));
            let start = format!(
                "select(0, {}, in_reach)",
                types.start(&format!("position_{axis}"))
            );
            let start = if stable {
                format!("start_in_bin({start}, bin_{axis} * BIN_SIDE_{axis}, BIN_SIDE_{axis}, FINE_{axis}_I32)")
            } else {
                start
            };
            format!(
                "    let position_{axis} = {position_value};
    let start_{axis} = {start};
    prepared_starts[STRIDE * slot + {axis}u] = start_{axis};
    prepared_offsets[DIMS * slot + {axis}u] = select({outside}, {offset}, in_reach);
",
                outside = types.outside_offset(),
                offset = types.offset(&format!("start_{axis}"), &format!("position_{axis}")),
            )
        })
        .collect();
    let bin_constants: String = if stable {
        bin_shape
            .iter()
            .zip(bins)
            .enumerate()
            .map(|(axis, (side, count))| {
                format!(
                    "const BIN_SIDE_{axis}: i32 = {side}i;\nconst BINS_{axis}: u32 = {count}u;\n"
                )
            })
            .collect()
    } else {
        String::new()
    };
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMS: u32 = {dimensions}u;
const STRIDE: u32 = {stride}u;
{bin_constants}
@group(0) @binding(0) var<storage, read> points: array<{coordinate}>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> prepared_starts: array<i32>;
@group(0) @binding(3) var<storage, read_write> prepared_offsets: array<{offset_type}>;
{rank_bindings}
// The count pass binned this start in another shader, where rounding may
// have carried it one cell across the edge of the bin. The nearest start of
// the bin keeps the point in reach of every block it touches; the cell it
// trades lies at the edge of the support, whose weight is about exp(-beta).
fn start_in_bin(start: i32, low: i32, side: i32, fine_length: i32) -> i32 {{
    let high = min(low + side, fine_length) - 1;
    var delta = select(start, start + fine_length, start < 0) - low;
    if (delta > fine_length / 2) {{ delta = delta - fine_length; }}
    if (delta < -(fine_length / 2)) {{ delta = delta + fine_length; }}
    return start + clamp(delta, 0, high - low) - delta;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&sorted_indices);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let record = wg_flat * WORKGROUP_SIZE + lid.x;
    if (record >= total) {{ return; }}

    let point_index = sorted_indices[record];
{rank}    let point_base = point_index * DIMS;
{loads}    let in_reach = {in_reach};
{axes}    prepared_starts[STRIDE * slot + DIMS] = bitcast<i32>(point_index);
}}
"#,
        stride = dimensions + 1,
        coordinate = types.coordinate_type(),
        offset_type = types.offset_type(),
    );
    types.with_library(&format!("{position}\n{entry}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_nd_wgsl::assert_valid_nd_wgsl;

    #[test]
    fn binning_shaders_validate_in_every_precision_and_rank() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
            let types = NdWgsl::new(precision);
            for dimensions in [1usize, 4, 6, 8] {
                let shape = vec![14usize; dimensions];
                let sides = vec![2usize; dimensions];
                let bins = vec![7usize; dimensions];
                let position = position_wgsl(&shape, kernel, precision);
                for stable in [false, true] {
                    let count = generate_count_wgsl(types, &sides, &bins, stable, &position);
                    assert_valid_nd_wgsl(precision, &count);
                    let prepare = generate_prepare_wgsl(types, &sides, &bins, stable, &position);
                    assert_valid_nd_wgsl(precision, &prepare);
                    assert_eq!(prepare.contains("rank = rank + 1u"), stable);
                    assert_eq!(prepare.contains("let start_0 = start_in_bin("), stable);
                }
            }
        }
        crate::wgsl_validation::assert_valid_wgsl(&generate_scatter_wgsl());
    }

    #[test]
    fn count_shader_bins_axis_zero_fastest() {
        let kernel = EsKernel::for_tolerance(1.0e-3, 2.0).unwrap();
        let position = position_wgsl(&[16, 12, 10, 8], kernel, FftPrecision::F32);
        let source = generate_count_wgsl(
            NdWgsl::new(FftPrecision::F32),
            &[4, 4, 2, 2],
            &[4, 3, 5, 4],
            true,
            &position,
        );
        assert!(source.contains(
            "let bin = start_0 / BIN_SIDE_0 + BINS_0 * (start_1 / BIN_SIDE_1 + BINS_1 * (start_2 / BIN_SIDE_2 + BINS_2 * (start_3 / BIN_SIDE_3)));"
        ));
        assert!(source.contains("atomicMax(&bin_counts[arrayLength(&bin_counts) - 1u], rank);"));
    }
}
