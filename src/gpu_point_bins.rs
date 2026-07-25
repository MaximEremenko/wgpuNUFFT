//! Coarse-bin ordering and per-point preparation for 1D-3D F32 points.
//!
//! [`PointBins::encode`] counts the points of every coarse bin with integer
//! atomics, keeping each point's bin and its rank among the points of that
//! bin. An exclusive prefix scan over the counts and one trailing zero gives
//! every bin's first slot and, in the last entry, the total. Every point then
//! owns slot `offset[bin] + rank`.
//!
//! The binned data is prepared per slot: the support start per axis and the
//! original index (`vec4<i32>`, index in `w`), and per axis the df64 offset
//! `start - position` split into an `f32` high/low pair. Consumers evaluate
//! the distance from support cell `start + j` as `(j + hi) + lo`, which
//! matches a per-use df64 subtraction without repeating df64 folding.
//!
//! [`PointBinOrder::Grouped`] suits consumers whose per-point results do not
//! depend on the order inside a bin, such as the binned type-2
//! interpolations, which walk points in bin order only for fine-grid cache
//! locality. Bins then come from a plain `f32` fold (a point near a bin edge
//! may land in the neighbouring bin), and the order inside a bin follows
//! atomic scheduling. While the prepared data of a point set is small enough
//! to stay cache-resident, one pass writes every point's prepared data
//! straight to its slot; larger sets scatter point indices first and then
//! prepare in slot order, which keeps the wide writes coalesced.
//! [`PointBinOrder::Stable`] bins with the exact df64
//! fold and restores original point order inside every bin with a per-bin
//! sort before preparing, so the order is deterministic; the 3D type-1 block
//! spreader sums points in this order, walking the bins through
//! [`PointBins::bin_offsets`].

use std::sync::Mutex;

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::GpuProfileQueryWriter;
use crate::gpu_scan::GpuExclusiveScanU32;
use crate::gpu_type1_3d::{
    binding_entry, checked_buffer_size, create_compute_pipeline, encode_pass, format_wgsl_f32,
    generate_sort_wgsl, validate_binding_limit, validate_buffer_limit,
};
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
const U32_BYTES: u64 = 4;
/// `vec2<u32>`: a point's bin and its rank among the points of that bin.
const POINT_SLOT_BYTES: u64 = 8;
/// `vec4<i32>`: support start per axis and the original point index.
const PREPARED_START_BYTES: u64 = 16;
/// One `vec2<f32>` high/low offset `start - position` per axis.
const PREPARED_OFFSET_BYTES_PER_AXIS: u64 = 8;
/// Largest prepared footprint written by the fused scatter-and-prepare pass.
/// Its writes land in random slots; the fused pass
/// beat the indexed passes by 13-17% up to 32 MiB (2D, 1M points) but was
/// 2.8x slower at 80 MiB (3D, 2M points), once the writes left the cache.
const FUSED_PREPARE_LIMIT_BYTES: u64 = 32 << 20;

/// How [`PointBins::encode`] orders the points inside each bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PointBinOrder {
    /// Scheduling-dependent order inside approximate bins.
    Grouped,
    /// Original point order inside exact bins.
    Stable,
}

/// Plan-owned storage that grows with the largest point set. A clone of an
/// older allocation stays alive in command buffers that captured it.
struct GrowOnlyBuffer {
    device: wgpu::Device,
    label: &'static str,
    buffer: Mutex<Option<wgpu::Buffer>>,
}

impl GrowOnlyBuffer {
    fn new(device: &wgpu::Device, label: &'static str) -> Self {
        Self {
            device: device.clone(),
            label,
            buffer: Mutex::new(None),
        }
    }

    fn get(&self, required_bytes: u64) -> wgpu::Buffer {
        let mut cached = self
            .buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match cached.as_ref() {
            Some(buffer) if buffer.size() >= required_bytes => buffer.clone(),
            _ => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(self.label),
                    size: required_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                *cached = Some(buffer.clone());
                buffer
            }
        }
    }
}

/// Prepared point data bound by consumers, in bin order.
pub(crate) struct PreparedPoints {
    pub(crate) starts: wgpu::Buffer,
    pub(crate) start_bytes: u64,
    pub(crate) offsets: wgpu::Buffer,
    pub(crate) offset_bytes: u64,
}

pub(crate) struct PointBins {
    dimensions: usize,
    /// One count per bin plus a trailing zero, so the scan also yields the
    /// total.
    bin_counts: wgpu::Buffer,
    bin_offsets: wgpu::Buffer,
    point_slots: GrowOnlyBuffer,
    count_pipeline: wgpu::ComputePipeline,
    count_layout: wgpu::BindGroupLayout,
    prefix_scan: GpuExclusiveScanU32,
    /// Grouped mode only: every point prepares itself into its slot.
    fused: Option<FusedPrepare>,
    indexed: IndexedPrepare,
    prepared_starts: GrowOnlyBuffer,
    prepared_offsets: GrowOnlyBuffer,
    max_workgroups_per_dimension: u32,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
}

struct FusedPrepare {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Index scatter, the per-bin sort of stable mode, then preparation in slot
/// order.
struct IndexedPrepare {
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    sort: Option<BinSort>,
    prepare_pipeline: wgpu::ComputePipeline,
    prepare_layout: wgpu::BindGroupLayout,
    sorted_indices: GrowOnlyBuffer,
}

struct BinSort {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    dispatch: (u32, u32, u32),
}

impl PointBins {
    /// Bins points of a `fine_shape` grid into `bin_shape`-cell bins
    /// (axis zero fastest). `position_wgsl` must define, on top of the df64
    /// library, `fold_position_{axis}` (an `f32` coordinate to its folded
    /// df64 grid position) and the `FINE_{axis}`, `FINE_{axis}_F32`,
    /// `POSITION_SCALE_{axis}_HI`, and `GRID_ORIGIN_{axis}` constants for
    /// every axis, and `floor_df64_to_i32`.
    pub(crate) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        fine_shape: &[usize],
        bin_shape: &[usize],
        order: PointBinOrder,
        position_wgsl: &str,
    ) -> Result<Self> {
        let dimensions = fine_shape.len();
        assert!(
            (1..=3).contains(&dimensions) && bin_shape.len() == dimensions,
            "point bins support one to three axes"
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
                context: "coarse point-bin count",
            })?;
        let entry_count = bin_count.checked_add(1).ok_or(NufftError::LengthOverflow {
            context: "coarse point-bin offset count",
        })?;
        u32::try_from(entry_count).map_err(|_| NufftError::LengthOverflow {
            context: "coarse point-bin shader index space",
        })?;
        let entry_bytes = checked_buffer_size(
            "coarse point-bin counts and offsets",
            entry_count,
            U32_BYTES,
        )?;
        validate_binding_limit(
            "coarse point-bin counts and offsets",
            entry_bytes,
            limits.max_storage_buffer_binding_size,
        )?;
        validate_buffer_limit(
            "coarse point-bin counts and offsets",
            entry_bytes,
            limits.max_buffer_size,
        )?;
        let bin_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.point_bins.bin_counts"),
            size: entry_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bin_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.point_bins.bin_offsets"),
            size: entry_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.point_bins.count",
            &generate_bin_count_wgsl(
                bin_shape,
                &bins,
                order == PointBinOrder::Stable,
                position_wgsl,
            ),
        );
        let count_layout = count_pipeline.get_bind_group_layout(0);
        let prefix_scan = GpuExclusiveScanU32::new(device, entry_count)?;
        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let fused = (order == PointBinOrder::Grouped).then(|| {
            let pipeline = create_compute_pipeline(
                device,
                "wgpu_nufft.point_bins.scatter_prepare",
                &generate_scatter_prepare_wgsl(kernel, dimensions, position_wgsl),
            );
            FusedPrepare {
                layout: pipeline.get_bind_group_layout(0),
                pipeline,
            }
        });
        let sort = match order {
            PointBinOrder::Grouped => None,
            PointBinOrder::Stable => {
                let pipeline = create_compute_pipeline(
                    device,
                    "wgpu_nufft.point_bins.sort",
                    &generate_sort_wgsl(),
                );
                let workgroups = u32::try_from(bin_count.div_ceil(WORKGROUP_SIZE as usize))
                    .map_err(|_| NufftError::LengthOverflow {
                        context: "coarse point-bin sort workgroup count",
                    })?;
                Some(BinSort {
                    layout: pipeline.get_bind_group_layout(0),
                    pipeline,
                    dispatch: split_workgroups(workgroups, max_workgroups_per_dimension)?,
                })
            }
        };
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.point_bins.scatter",
            &generate_bin_scatter_wgsl(),
        );
        let prepare_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.point_bins.prepare",
            &generate_prepare_wgsl(kernel, dimensions, position_wgsl),
        );
        let indexed = IndexedPrepare {
            scatter_layout: scatter_pipeline.get_bind_group_layout(0),
            scatter_pipeline,
            sort,
            prepare_layout: prepare_pipeline.get_bind_group_layout(0),
            prepare_pipeline,
            sorted_indices: GrowOnlyBuffer::new(device, "wgpu_nufft.point_bins.sorted_indices"),
        };
        Ok(Self {
            dimensions,
            bin_counts,
            bin_offsets,
            point_slots: GrowOnlyBuffer::new(device, "wgpu_nufft.point_bins.point_slots"),
            count_pipeline,
            count_layout,
            prefix_scan,
            fused,
            indexed,
            prepared_starts: GrowOnlyBuffer::new(device, "wgpu_nufft.point_bins.prepared_starts"),
            prepared_offsets: GrowOnlyBuffer::new(device, "wgpu_nufft.point_bins.prepared_offsets"),
            max_workgroups_per_dimension,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    /// Bin offsets of the last [`Self::encode`], one entry per coarse bin plus
    /// a terminal total, in axis-zero-fastest bin order.
    pub(crate) fn bin_offsets(&self) -> &wgpu::Buffer {
        &self.bin_offsets
    }

    fn scratch(
        &self,
        buffer: &GrowOnlyBuffer,
        label: &'static str,
        point_count: usize,
        bytes_per_point: u64,
    ) -> Result<(wgpu::Buffer, u64)> {
        let bytes = checked_buffer_size(label, point_count, bytes_per_point)?;
        validate_binding_limit(label, bytes, self.max_storage_binding_bytes)?;
        validate_buffer_limit(label, bytes, self.max_buffer_bytes)?;
        Ok((buffer.get(bytes), bytes))
    }

    /// Prepared data for the `point_count` points of the last
    /// [`Self::encode`], in bin order.
    pub(crate) fn prepared(&self, point_count: usize) -> Result<PreparedPoints> {
        let (starts, start_bytes) = self.scratch(
            &self.prepared_starts,
            "prepared point starts",
            point_count,
            PREPARED_START_BYTES,
        )?;
        let (offsets, offset_bytes) = self.scratch(
            &self.prepared_offsets,
            "prepared point offsets",
            point_count,
            PREPARED_OFFSET_BYTES_PER_AXIS * self.dimensions as u64,
        )?;
        Ok(PreparedPoints {
            starts,
            start_bytes,
            offsets,
            offset_bytes,
        })
    }

    /// Records the coarse-bin order and prepared data of `point_count > 0`
    /// points. Both stay valid for consumers until the next call; executions
    /// must keep queue order.
    ///
    /// With profiling, stable mode writes the start marker and the ends of the
    /// count, scan, scatter, and sort-and-prepare stages at offsets 0-4 of the
    /// type-1 layout (the scan ends where the scatter begins). Grouped mode
    /// writes the start marker at offset 0 and the end of the last pass at
    /// offset 1.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `point_count` is zero.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        point_bytes: u64,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        debug_assert!(point_count > 0);
        let point_count_u32 =
            u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
                context: "binned GPU point count",
            })?;
        let point_dispatch = split_workgroups(
            point_count_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;
        let (point_slots, slot_bytes) = self.scratch(
            &self.point_slots,
            "coarse-bin point slots",
            point_count,
            POINT_SLOT_BYTES,
        )?;
        let prepared = self.prepared(point_count)?;
        let bin_counts = &self.bin_counts;
        let bin_offsets = &self.bin_offsets;
        let offset_bytes = bin_offsets.size();
        #[cfg(feature = "gpu-profiling")]
        let stable = self.indexed.sort.is_some();
        let fused = self
            .fused
            .as_ref()
            .filter(|_| prepared.start_bytes + prepared.offset_bytes <= FUSED_PREPARE_LIMIT_BYTES);

        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.point_bins.count.bind_group"),
            layout: &self.count_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, bin_counts, bin_counts.size()),
                binding_entry(2, &point_slots, slot_bytes),
            ],
        });
        #[cfg(feature = "gpu-profiling")]
        profile.encode_start_marker(encoder);
        encoder.clear_buffer(bin_counts, 0, None);
        encode_pass(
            encoder,
            "wgpu_nufft.point_bins.count.pass",
            &self.count_pipeline,
            &count_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            stable
                .then(|| profile.timestamp_writes(None, Some(1)))
                .flatten(),
        );
        self.prefix_scan
            .encode(device, encoder, bin_counts, bin_offsets)?;

        if let Some(fused) = fused {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.point_bins.scatter_prepare.bind_group"),
                layout: &fused.layout,
                entries: &[
                    binding_entry(0, points, point_bytes),
                    binding_entry(1, &point_slots, slot_bytes),
                    binding_entry(2, bin_offsets, offset_bytes),
                    binding_entry(3, &prepared.starts, prepared.start_bytes),
                    binding_entry(4, &prepared.offsets, prepared.offset_bytes),
                ],
            });
            encode_pass(
                encoder,
                "wgpu_nufft.point_bins.scatter_prepare.pass",
                &fused.pipeline,
                &bind_group,
                point_dispatch,
                #[cfg(feature = "gpu-profiling")]
                profile.timestamp_writes(None, Some(1)),
            );
            return Ok(());
        }

        let indexed = &self.indexed;
        let (sorted_indices, index_bytes) = self.scratch(
            &indexed.sorted_indices,
            "coarse-bin sorted point indices",
            point_count,
            U32_BYTES,
        )?;
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.point_bins.scatter.bind_group"),
            layout: &indexed.scatter_layout,
            entries: &[
                binding_entry(0, &point_slots, slot_bytes),
                binding_entry(1, bin_offsets, offset_bytes),
                binding_entry(2, &sorted_indices, index_bytes),
            ],
        });
        let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.point_bins.prepare.bind_group"),
            layout: &indexed.prepare_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, &sorted_indices, index_bytes),
                binding_entry(2, &prepared.starts, prepared.start_bytes),
                binding_entry(3, &prepared.offsets, prepared.offset_bytes),
            ],
        });
        encode_pass(
            encoder,
            "wgpu_nufft.point_bins.scatter.pass",
            &indexed.scatter_pipeline,
            &scatter_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            stable
                .then(|| profile.timestamp_writes(Some(2), Some(3)))
                .flatten(),
        );
        if let Some(sort) = &indexed.sort {
            let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.point_bins.sort.bind_group"),
                layout: &sort.layout,
                entries: &[
                    binding_entry(0, bin_offsets, offset_bytes),
                    binding_entry(1, &sorted_indices, index_bytes),
                ],
            });
            encode_pass(
                encoder,
                "wgpu_nufft.point_bins.sort.pass",
                &sort.pipeline,
                &sort_bind_group,
                sort.dispatch,
                #[cfg(feature = "gpu-profiling")]
                None,
            );
        }
        encode_pass(
            encoder,
            "wgpu_nufft.point_bins.prepare.pass",
            &indexed.prepare_pipeline,
            &prepare_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(if stable { 4 } else { 1 })),
        );
        Ok(())
    }
}

fn generate_bin_count_wgsl(
    bin_shape: &[usize],
    bins: &[usize],
    exact: bool,
    position_wgsl: &str,
) -> String {
    let dimensions = bin_shape.len();
    let constants: String = bin_shape
        .iter()
        .enumerate()
        .map(|(axis, side)| format!("const BIN_SIDE_{axis}: u32 = {side}u;\n"))
        .chain(
            bins.iter()
                .enumerate()
                .take(dimensions - 1)
                .map(|(axis, count)| format!("const BINS_{axis}: u32 = {count}u;\n")),
        )
        .collect();
    let cells: String = (0..dimensions)
        .map(|axis| {
            if exact {
                format!(
                    "    let cell_{axis} = u32(floor_df64_to_i32(fold_position_{axis}(points[point_base + {axis}u])));\n"
                )
            } else {
                format!(
                    "    let cell_{axis} = approximate_cell(points[point_base + {axis}u], POSITION_SCALE_{axis}_HI, GRID_ORIGIN_{axis}, FINE_{axis}_F32, FINE_{axis});\n"
                )
            }
        })
        .collect();
    // Axis-zero-fastest bin index, nested from the last axis inward.
    let mut bin = format!("cell_{last} / BIN_SIDE_{last}", last = dimensions - 1);
    for axis in (0..dimensions - 1).rev() {
        bin = format!("cell_{axis} / BIN_SIDE_{axis} + BINS_{axis} * ({bin})");
    }
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const POINT_DIMENSIONS: u32 = {dimensions}u;
{constants}
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> point_slots: array<vec2<u32>>;

// Grid cell of a coordinate folded in plain f32, clamped onto the grid. It may
// differ from the exact df64 cell near a cell edge.
fn approximate_cell(point: f32, scale: f32, origin: f32, fine_length: f32, cells: u32) -> u32 {{
    let position = point * scale + origin;
    let wrapped = position - fine_length * floor(position / fine_length);
    return min(u32(max(wrapped, 0.0)), cells - 1u);
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

    let point_base = point_index * POINT_DIMENSIONS;
{cells}    let bin = {bin};
    point_slots[point_index] = vec2<u32>(bin, atomicAdd(&bin_counts[bin], 1u));
}}
"#,
    );
    format!("{}\n{position_wgsl}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_bin_scatter_wgsl() -> String {
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

/// Shared declarations of the preparing shaders: constants, the output
/// bindings at `binding` and `binding + 1`, and `ceil_df64_to_i32`.
fn prepare_header(kernel: EsKernel, dimensions: usize, binding: u32) -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const POINT_DIMENSIONS: u32 = {dimensions}u;
const HALF_WIDTH: f32 = {half_width};

@group(0) @binding({binding}) var<storage, read_write> prepared_starts: array<vec4<i32>>;
@group(0) @binding({offsets_binding}) var<storage, read_write> prepared_offsets: array<vec2<f32>>;

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}
"#,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        offsets_binding = binding + 1,
    )
}

/// WGSL statements preparing `point_index` (with coordinates at
/// `point_base`) into `slot`.
fn prepare_statements(dimensions: usize) -> String {
    let axes: String = (0..dimensions)
        .map(|axis| {
            format!(
                "    let position_{axis} = fold_position_{axis}(points[point_base + {axis}u]);
    let start_{axis} = ceil_df64_to_i32(df64_sub(position_{axis}, Df64(HALF_WIDTH, 0.0)));
    let offset_{axis} = df64_sub(Df64(f32(start_{axis}), 0.0), position_{axis});
    prepared_offsets[POINT_DIMENSIONS * slot + {axis}u] = vec2<f32>(offset_{axis}.hi, offset_{axis}.lo);
"
            )
        })
        .collect();
    let starts = (0..3)
        .map(|axis| {
            if axis < dimensions {
                format!("start_{axis}")
            } else {
                "0".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{axes}    prepared_starts[slot] = vec4<i32>({starts}, bitcast<i32>(point_index));\n")
}

fn generate_prepare_wgsl(kernel: EsKernel, dimensions: usize, position_wgsl: &str) -> String {
    let header = prepare_header(kernel, dimensions, 2);
    let statements = prepare_statements(dimensions);
    let entry = format!(
        r#"{header}
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&sorted_indices);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let slot = wg_flat * WORKGROUP_SIZE + lid.x;
    if (slot >= total) {{ return; }}

    let point_index = sorted_indices[slot];
    let point_base = point_index * POINT_DIMENSIONS;
{statements}}}
"#,
    );
    format!("{}\n{position_wgsl}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_scatter_prepare_wgsl(
    kernel: EsKernel,
    dimensions: usize,
    position_wgsl: &str,
) -> String {
    let header = prepare_header(kernel, dimensions, 3);
    let statements = prepare_statements(dimensions);
    let entry = format!(
        r#"{header}
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> point_slots: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;

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

    let bin_slot = point_slots[point_index];
    let slot = bin_offsets[bin_slot.x] + bin_slot.y;
    let point_base = point_index * POINT_DIMENSIONS;
{statements}}}
"#,
    );
    format!("{}\n{position_wgsl}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
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
    fn one_axis_binning_shaders_validate_with_the_1d_position_adapter() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let position = crate::gpu_type1::generate_binned_position_wgsl(128);
        for exact in [false, true] {
            let count = generate_bin_count_wgsl(&[8], &[16], exact, &position);
            assert!(count.contains("let bin = cell_0 / BIN_SIDE_0;"));
            assert_valid_wgsl(&count);
        }
        for source in [
            generate_prepare_wgsl(kernel, 1, &position),
            generate_scatter_prepare_wgsl(kernel, 1, &position),
        ] {
            assert!(source.contains("vec4<i32>(start_0, 0, 0, bitcast<i32>(point_index))"));
            assert_valid_wgsl(&source);
        }
    }

    #[test]
    fn two_and_three_axis_binning_shaders_validate() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let position_2d = crate::gpu_type1_2d::generate_position_wgsl([40, 24]);
        let position_3d = crate::gpu_type1_3d::generate_position_wgsl([40, 24, 32]);
        for exact in [false, true] {
            let count_2d = generate_bin_count_wgsl(&[8, 8], &[5, 3], exact, &position_2d);
            assert!(count_2d
                .contains("let bin = cell_0 / BIN_SIDE_0 + BINS_0 * (cell_1 / BIN_SIDE_1);"));
            assert_eq!(
                count_2d.contains("floor_df64_to_i32(fold_position_0"),
                exact
            );
            let count_3d = generate_bin_count_wgsl(&[4, 4, 4], &[10, 6, 8], exact, &position_3d);
            assert!(count_3d.contains(
                "let bin = cell_0 / BIN_SIDE_0 + BINS_0 * (cell_1 / BIN_SIDE_1 + BINS_1 * (cell_2 / BIN_SIDE_2));"
            ));
            assert_valid_wgsl(&count_2d);
            assert_valid_wgsl(&count_3d);
        }
        let prepare_2d = generate_prepare_wgsl(kernel, 2, &position_2d);
        assert!(prepare_2d.contains("vec4<i32>(start_0, start_1, 0, bitcast<i32>(point_index))"));
        let prepare_3d = generate_scatter_prepare_wgsl(kernel, 3, &position_3d);
        assert!(
            prepare_3d.contains("vec4<i32>(start_0, start_1, start_2, bitcast<i32>(point_index))")
        );
        assert!(prepare_3d.contains("let slot = bin_offsets[bin_slot.x] + bin_slot.y;"));
        for source in [
            prepare_2d,
            prepare_3d,
            generate_prepare_wgsl(kernel, 3, &position_3d),
            generate_scatter_prepare_wgsl(kernel, 2, &position_2d),
            generate_bin_scatter_wgsl(),
        ] {
            assert_valid_wgsl(&source);
        }
    }
}
