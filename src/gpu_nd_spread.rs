//! Deterministic output-stationary type-1 spreading in any rank and
//! precision.
//!
//! [`NdPointBins`](crate::gpu_nd_bins::NdPointBins) records a stable bin
//! order of the points and their prepared support starts and offsets.
//! [`NdBlockSpread::encode_spread`] then assigns each workgroup a block of
//! fine-grid cells: every invocation owns a run of cells along axis zero and
//! one cell along every other axis, and keeps its sums in registers. The
//! workgroup walks the bins that can reach the block, one batch of bin rows
//! along axis zero at a time, and streams their records in a fixed order
//! through workgroup memory. A record whose support misses the block is
//! skipped after its start is read; the others get their kernel weights
//! computed once for the whole workgroup. Each cell therefore sums its
//! contributions in a fixed order and is written exactly once: no float
//! atomics, no read-modify-write, and no clearing pass.
//!
//! A classify pass first counts the records in every block's reach. Empty
//! blocks write zeros at once. Blocks above a per-plan record limit, which
//! keeps every light workgroup short, are listed instead. A plan pass gives
//! every listed block as many parts as its records need, at a record count
//! per part that depends only on the plan, the point count and the batch;
//! the heavy part pass then spreads each part, in reach order, in its own
//! workgroup, and the heavy reduce pass adds the parts of every cell in part
//! order. A dense cluster is therefore split across many workgroups, and its
//! cells stay deterministic too.

use std::fmt::Write as _;
use std::sync::Mutex;

use wgpu_fft::FftPrecision;

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::{split_workgroups, RangedDispatch, DISPATCH_RANGE_WGSL};
use crate::gpu_nd_bins::NdPreparedPoints;
use crate::gpu_nd_wgsl::{position_wgsl, weight_wgsl, NdWgsl};
use crate::gpu_point_bins::GrowOnlyBuffer;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::GpuProfileQueryWriter;
use crate::gpu_recorder::GpuRecorder;
use crate::gpu_type1_3d::{binding_entry, create_compute_pipeline, encode_pass};
use crate::kernel::EsKernel;

/// Largest block edge along an axis other than zero.
const MAX_BLOCK_EDGE: usize = 16;
/// Bin rows gathered and scanned at a time.
const MAX_ROW_BATCH: usize = 128;
/// Records staged in workgroup memory at a time, at most.
const MAX_CHUNK: usize = 64;
/// Bin rows a block's reach may span.
const MAX_ROWS: usize = 1 << 20;
/// Bin rows a block's reach spans on average before the bins coarsen: every
/// row costs a lookup, while coarser bins stage records outside the reach.
const TARGET_REACH_ROWS: f64 = 1024.0;
/// Bins a grid holds before the bins coarsen: every bin costs a count and a
/// scan entry per point set.
const MAX_BINS: usize = 1 << 23;
/// Invocation steps a light workgroup may spend, which bounds its records.
const LIGHT_WORK_BUDGET: u64 = 1 << 29;
/// Bounds of the per-plan record limit of a light block.
const MIN_HEAVY_RECORDS: u64 = 1 << 10;
const MAX_HEAVY_RECORDS: u64 = 1 << 18;
/// Workgroups that keep a large GPU busy. A grid with fewer blocks than this
/// lowers its record limit so dense blocks are split into parts too.
const TARGET_PARALLEL_WORKGROUPS: u64 = 4096;
/// The smallest record limit and the smallest heavy part, so parts stay
/// worth their partial sums.
const MIN_PART_RECORDS: u64 = 256;
/// Partial-sum scratch for the heavy parts: more parts fit when the block
/// and the batch are small.
const HEAVY_SCRATCH_BUDGET_BYTES: u64 = 64 << 20;
/// Invocations of the heavy plan pass, a single workgroup.
const PLAN_WORKGROUP_SIZE: u32 = 256;
/// Invocation steps one spreading dispatch may take; larger workloads are
/// split into consecutive dispatches.
const DISPATCH_WORK_BUDGET: u64 = 1 << 36;
/// Workgroups below which a dispatch is never split further, so that every
/// dispatch still fills a large GPU.
const MIN_WORKGROUPS_PER_DISPATCH: u64 = 4096;
/// Invocations of the classify pass, one block each.
const CLASSIFY_WORKGROUP_SIZE: u32 = 64;
/// Invocations of the heavy reduce pass, one cell each.
const REDUCE_WORKGROUP_SIZE: u32 = 256;
/// Block state of a block left to the heavy passes.
const HEAVY_STATE: u32 = u32::MAX;

/// Static geometry of the spreader for one plan.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NdSpreadLayout {
    width: usize,
    /// Cells per invocation along axis zero.
    run: usize,
    /// Invocations along axis zero.
    groups_0: usize,
    /// Block edge per axis; axis zero spans `groups_0 * run` cells.
    block: Vec<usize>,
    /// Blocks per axis.
    blocks: Vec<usize>,
    /// Cells per bin along each axis.
    bin_side: Vec<usize>,
    /// Bins per axis.
    bins: Vec<usize>,
    /// Whether a block's reach covers the whole axis.
    full: Vec<bool>,
    /// Most bins a block's reach spans along each axis.
    reach_bins: Vec<usize>,
    workgroup_size: usize,
    row_batch: usize,
    chunk: usize,
    /// Records in reach above which a block goes to the heavy passes.
    heavy_records: u32,
    /// The record limit is at most `point_count / dense_divisor`: grids with
    /// few blocks split dense blocks into parts to use the whole GPU.
    dense_divisor: u32,
    /// The share of evenly spread points whose support reaches one block.
    block_fraction: f64,
    fine_shape: Vec<usize>,
    /// Invocation steps a workgroup spends on one staged record.
    record_steps: u64,
}

impl NdSpreadLayout {
    /// Chooses blocks, bins and staging sizes for this grid and device.
    pub(crate) fn for_grid(
        kernel: EsKernel,
        fine_shape: &[usize],
        precision: FftPrecision,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        let dimensions = fine_shape.len();
        let width = kernel.width();
        let types = NdWgsl::new(precision);
        let max_invocations = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x) as usize;
        let target = target_workgroup_size(dimensions, precision)
            .min(max_invocations)
            .max(1);
        // Support starts relative to a block stay unique while the block edge
        // plus the kernel width fits the axis.
        let edge_limit = |axis: usize| (fine_shape[axis] + 1).saturating_sub(width).max(1);
        let run = largest_power_of_two_at_most(max_run(dimensions, precision).min(edge_limit(0)));
        let mut block = vec![1usize; dimensions];
        // Double the thinnest axis first: a point reaches about
        // `(edge + width) / edge` blocks along each axis.
        let mut invocations = 1usize;
        loop {
            let candidate = (1..dimensions)
                .filter(|&axis| {
                    block[axis] * 2 <= edge_limit(axis).min(MAX_BLOCK_EDGE)
                        && invocations * 2 <= target
                })
                .min_by_key(|&axis| (block[axis], usize::MAX - fine_shape[axis]));
            let Some(axis) = candidate else {
                break;
            };
            block[axis] *= 2;
            invocations *= 2;
        }
        // Axes too short for more invocations leave room for several runs
        // along axis zero.
        let mut groups_0 = 1usize;
        while invocations * 2 <= target && run * groups_0 * 2 <= edge_limit(0) {
            groups_0 *= 2;
            invocations *= 2;
        }
        block[0] = run * groups_0;
        let workgroup_size = groups_0 * block[1..].iter().product::<usize>();

        // Bins hold support starts, so a block's records are those of the
        // bins over the block and the kernel width below it. Single-cell bins
        // stage no other record, but every bin row of a reach costs a lookup
        // and every bin a count and a scan entry: the axes above zero start
        // from single cells and coarsen along the widest reach while a reach
        // spans more than `TARGET_REACH_ROWS` rows on average or the grid
        // holds more than `MAX_BINS` bins. Along axis zero a row is one
        // contiguous range of records; two-cell bins measured best there,
        // except for the small two-dimensional F32 blocks.
        let reach_cells = |axis: usize| block[axis] + width - 1;
        let mut bin_side = vec![1usize; dimensions];
        if dimensions != 2 || precision != FftPrecision::F32 {
            bin_side[0] = 2;
        }
        let mean_reach_bins = |bin_side: &[usize], axis: usize| {
            ((reach_cells(axis) - 1) as f64 / bin_side[axis] as f64 + 1.0)
                .min(fine_shape[axis].div_ceil(bin_side[axis]) as f64)
        };
        loop {
            let rows = (1..dimensions)
                .map(|axis| mean_reach_bins(&bin_side, axis))
                .product::<f64>();
            let bin_count = (0..dimensions)
                .try_fold(1usize, |count, axis| {
                    count.checked_mul(fine_shape[axis].div_ceil(bin_side[axis]))
                })
                .unwrap_or(usize::MAX);
            if rows <= TARGET_REACH_ROWS && bin_count <= MAX_BINS {
                break;
            }
            let widest = (1..dimensions)
                .filter(|&axis| bin_side[axis] < fine_shape[axis])
                .max_by(|&a, &b| {
                    mean_reach_bins(&bin_side, a).total_cmp(&mean_reach_bins(&bin_side, b))
                });
            let axis = match widest {
                Some(axis) => axis,
                None if bin_count > MAX_BINS && bin_side[0] < fine_shape[0] => 0,
                None => break,
            };
            bin_side[axis] *= 2;
        }
        let reach_geometry = |bin_side: &[usize]| {
            let mut bins = Vec::with_capacity(dimensions);
            let mut full = Vec::with_capacity(dimensions);
            let mut reach_bins = Vec::with_capacity(dimensions);
            for axis in 0..dimensions {
                let count = fine_shape[axis].div_ceil(bin_side[axis]);
                let whole = reach_cells(axis) + bin_side[axis] > fine_shape[axis];
                bins.push(count);
                full.push(whole);
                reach_bins.push(if whole {
                    count
                } else {
                    count.min(reach_cells(axis).div_ceil(bin_side[axis]) + 2)
                });
            }
            (bins, full, reach_bins)
        };
        let (mut bins, mut full, mut reach_bins) = reach_geometry(&bin_side);
        // Coarser bins along the widest reaches keep the row count bounded.
        while reach_bins[1..]
            .iter()
            .try_fold(1usize, |rows, &n| rows.checked_mul(n))
            .is_none_or(|rows| rows > MAX_ROWS)
        {
            let Some(axis) = (1..dimensions)
                .filter(|&axis| reach_bins[axis] > 1)
                .max_by_key(|&axis| reach_bins[axis])
            else {
                break;
            };
            bin_side[axis] *= 2;
            (bins, full, reach_bins) = reach_geometry(&bin_side);
        }
        let rows = reach_bins[1..].iter().product::<usize>().max(1);
        let row_batch = MAX_ROW_BATCH
            .min(workgroup_size)
            .min(rows.next_power_of_two())
            .max(1);
        let storage_limit = limits.max_compute_workgroup_storage_size as usize;
        let chunk_bytes =
            4 * dimensions + types.complex_bytes() + dimensions * width * types.weight_bytes();
        let row_bytes = 4 * 4 * row_batch;
        let chunk = (0..=max_chunk(dimensions, precision).trailing_zeros())
            .rev()
            .map(|shift| 1usize << shift)
            .find(|&chunk| row_bytes + chunk * chunk_bytes <= storage_limit)
            .ok_or(NufftError::GpuWorkgroupStorageUnsupported {
                requested_bytes: u32::try_from(row_bytes + chunk_bytes).unwrap_or(u32::MAX),
                maximum_bytes: limits.max_compute_workgroup_storage_size,
            })?;
        let cost_factor = match precision {
            FftPrecision::F32 => 1,
            FftPrecision::F64 => 4,
            FftPrecision::Df64 => 8,
        };
        let record_steps = (workgroup_size * (dimensions + run) * cost_factor) as u64;
        let heavy_records = (LIGHT_WORK_BUDGET / record_steps.max(1))
            .clamp(MIN_HEAVY_RECORDS, MAX_HEAVY_RECORDS) as u32;
        let blocks = fine_shape
            .iter()
            .zip(&block)
            .map(|(&length, &edge)| length.div_ceil(edge))
            .collect::<Vec<_>>();
        // Record visits over all blocks per point, for evenly spread points.
        let block_fraction = (0..dimensions)
            .map(|axis| ((block[axis] + width - 1) as f64 / fine_shape[axis] as f64).min(1.0))
            .product::<f64>();
        let visits_per_point = block_fraction * blocks.iter().product::<usize>() as f64;
        let dense_divisor = (TARGET_PARALLEL_WORKGROUPS as f64 / visits_per_point)
            .ceil()
            .clamp(1.0, f64::from(u32::MAX)) as u32;
        let layout = Self {
            width,
            run,
            groups_0,
            block,
            blocks,
            bin_side,
            bins,
            full,
            reach_bins,
            workgroup_size,
            row_batch,
            chunk,
            heavy_records,
            dense_divisor,
            block_fraction,
            fine_shape: fine_shape.to_vec(),
            record_steps,
        };
        let block_count = layout
            .blocks
            .iter()
            .try_fold(1usize, |count, &n| count.checked_mul(n))
            .ok_or(NufftError::LengthOverflow {
                context: "rank-generic spread block count",
            })?;
        u32::try_from(block_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic spread block index space",
        })?;
        Ok(layout)
    }

    /// Cells per bin along each axis, for the point binning.
    pub(crate) fn bin_side(&self) -> &[usize] {
        &self.bin_side
    }

    fn block_count(&self) -> usize {
        self.blocks.iter().product()
    }

    /// Records whose support reaches one block, for `point_count` evenly
    /// spread points: the block edge plus the kernel width along every axis.
    fn block_records(&self, point_count: usize) -> u64 {
        (point_count as f64 * self.block_fraction).ceil() as u64
    }

    /// Records in reach above which a block goes to the heavy passes, for
    /// `point_count` points; the classify pass derives the same value.
    fn light_limit(&self, point_count: usize) -> u64 {
        (point_count as u64 / u64::from(self.dense_divisor))
            .clamp(MIN_PART_RECORDS, u64::from(self.heavy_records))
    }

    /// Invocations that stage one record and share its kernel weights.
    /// `F32` weights are cheap, and one invocation per record measured
    /// fastest; `F64` and `Df64` Horner weights are shared by the workgroup.
    fn record_lanes(&self, precision: FftPrecision) -> usize {
        match precision {
            FftPrecision::F32 => 1,
            FftPrecision::F64 | FftPrecision::Df64 => (self.workgroup_size / self.chunk).max(1),
        }
    }

    /// Light workgroups per dispatch for `point_count` evenly spread points:
    /// each block's reach then holds its share of the grid volume, and never
    /// more than the heavy limit.
    fn light_workgroups_per_dispatch(&self, point_count: usize) -> u64 {
        let records = self
            .block_records(point_count)
            .min(self.light_limit(point_count))
            .max(1);
        let rows = self.reach_bins[1..].iter().product::<usize>().max(1) as u64;
        let steps = records * self.record_steps + rows * self.workgroup_size as u64;
        (DISPATCH_WORK_BUDGET / steps).max(MIN_WORKGROUPS_PER_DISPATCH)
    }

    /// Heavy part workgroups per dispatch: a part holds at most
    /// `part_records` records.
    fn part_workgroups_per_dispatch(&self, part_records: u64) -> u64 {
        let steps = part_records.max(1).saturating_mul(self.record_steps);
        (DISPATCH_WORK_BUDGET / steps).max(MIN_WORKGROUPS_PER_DISPATCH)
    }

    fn block_cells(&self) -> usize {
        self.workgroup_size * self.run
    }

    /// Blocks whose reach can include one bin, bounding how often a record is
    /// counted over all reaches.
    fn blocks_per_bin(&self) -> u64 {
        (0..self.block.len())
            .map(|axis| {
                if self.full[axis] {
                    self.blocks[axis] as u64
                } else {
                    let edge = self.block[axis];
                    let span = self.bin_side[axis] + edge + self.width + 2;
                    (span.div_ceil(edge) + 1).min(self.blocks[axis]) as u64
                }
            })
            .product()
    }
}

/// Invocations of a spreading workgroup, when the device allows them. Every
/// invocation checks every staged record, so workgroups that are small
/// relative to a record's reach waste fewer checks; `Df64` plans instead
/// share their costly kernel weights across a large workgroup, as do native
/// `F64` plans from four dimensions on. Chosen by measurement.
fn target_workgroup_size(dimensions: usize, precision: FftPrecision) -> usize {
    match (dimensions, precision) {
        (1, _) => 64,
        (2, FftPrecision::F32) => 32,
        (2 | 3, FftPrecision::F64) | (_, FftPrecision::F32) => 64,
        _ => 256,
    }
}

/// Cells per invocation along axis zero. Long runs reuse every staged record
/// over more cells, which pays from three dimensions on, and `F32` sums fit
/// more registers; in one and two dimensions a record reaches few cells of a
/// long run, so short runs keep more invocations busy.
fn max_run(dimensions: usize, precision: FftPrecision) -> usize {
    match (dimensions, precision) {
        (1, _) | (2, FftPrecision::Df64) => 2,
        (2, _) => 4,
        (3, _) | (_, FftPrecision::F64 | FftPrecision::Df64) => 8,
        _ => 16,
    }
}

/// Records staged at a time. Two-dimensional `F32` and `Df64` blocks gain
/// from shorter chunks, which leave more workgroups resident; the others,
/// native `F64` especially, share each chunk's weight evaluation better when
/// it is long. Chosen by measurement.
fn max_chunk(dimensions: usize, precision: FftPrecision) -> usize {
    match (dimensions, precision) {
        (2, FftPrecision::F32 | FftPrecision::Df64) => MAX_CHUNK / 2,
        _ => MAX_CHUNK,
    }
}

fn largest_power_of_two_at_most(value: usize) -> usize {
    if value == 0 {
        1
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

/// Heavy-pass geometry of one execution.
struct HeavyGeometry {
    /// Most blocks the classify pass can list.
    capacity: u64,
    /// Part slots per vector in the partial-sum scratch; every listed block
    /// needs at least one.
    max_parts: u64,
    /// The most records one part can hold: the plan pass derives its part
    /// size from the listed records, which never exceed the bound behind
    /// [`part_records`], and may double it once to absorb `f32` rounding.
    part_records: u64,
}

impl HeavyGeometry {
    /// All reaches hold at most `blocks_per_bin * point_count` records, so at
    /// most `bound / (limit + 1)` blocks exceed the limit. The capacity also
    /// keeps one part per listed block within `max_scratch_bytes`; when that
    /// binds below the block count, the classify pass raises its limit to
    /// `bound / (capacity + 1)`, so it never lists more blocks than the
    /// scratch holds.
    fn new(
        layout: &NdSpreadLayout,
        point_count: usize,
        active_batch: usize,
        complex_bytes: u64,
        max_scratch_bytes: u64,
    ) -> Self {
        let blocks_per_bin = layout.blocks_per_bin();
        let bound = blocks_per_bin.saturating_mul(point_count as u64);
        let per_block = active_batch as u64 * layout.block_cells() as u64 * complex_bytes;
        let capacity = (bound / (layout.light_limit(point_count) + 1))
            .min(layout.block_count() as u64)
            .min(max_scratch_bytes / per_block);
        let budget = HEAVY_SCRATCH_BUDGET_BYTES.min(max_scratch_bytes);
        // A slot for every listable block, and the budget's worth of slots
        // for a few heavy blocks to split into many parts; `(blocks_per_bin
        // + 1) * spare` must fit a u32 for the plan pass's arithmetic.
        let spare_limit = u64::from(u32::MAX) / (blocks_per_bin + 1);
        let max_parts = (budget / per_block)
            .min(max_scratch_bytes / per_block)
            .min(capacity + spare_limit)
            .max(capacity);
        let part_records = part_records(point_count as u64, max_parts - capacity, blocks_per_bin)
            .saturating_mul(2);
        Self {
            capacity,
            max_parts,
            part_records,
        }
    }
}

/// Records per heavy part if every reach held its most records: all reaches
/// hold at most `blocks_per_bin * point_count`, and each listed block adds at
/// most one rounded-up part, so parts of this size never need more than
/// `spare + capacity` slots. The plan pass sizes parts from the actual
/// listed records, which is never larger.
fn part_records(point_count: u64, spare: u64, blocks_per_bin: u64) -> u64 {
    let unlimited = u64::from(u32::MAX);
    if spare == 0 {
        return unlimited;
    }
    let per = point_count / spare;
    if per >= unlimited / blocks_per_bin.max(1) {
        return unlimited;
    }
    let extra = (blocks_per_bin * (point_count % spare)).div_ceil(spare);
    (blocks_per_bin * per + extra).clamp(MIN_PART_RECORDS, unlimited)
}

pub(crate) struct NdBlockSpread {
    layout: NdSpreadLayout,
    complex_bytes: u64,
    spread_pipeline: wgpu::ComputePipeline,
    spread_layout: wgpu::BindGroupLayout,
    part_pipeline: wgpu::ComputePipeline,
    part_layout: wgpu::BindGroupLayout,
    classify_pipeline: wgpu::ComputePipeline,
    classify_layout: wgpu::BindGroupLayout,
    reduce_pipeline: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    plan_pipeline: wgpu::ComputePipeline,
    plan_layout: wgpu::BindGroupLayout,
    reset_pipeline: wgpu::ComputePipeline,
    reset_layout: wgpu::BindGroupLayout,
    /// Records in each block's reach, or [`HEAVY_STATE`], rewritten by every
    /// classify pass.
    block_state: wgpu::Buffer,
    /// A count, then `(block, records)` for every listed heavy block.
    heavy_list: GrowOnlyBuffer,
    /// The first part slot of every listed block, then the total.
    part_offsets: GrowOnlyBuffer,
    /// Partial sums of the heavy parts.
    partials: Mutex<Option<wgpu::Buffer>>,
    ranged: RangedDispatch,
    device: wgpu::Device,
    max_workgroups_per_dimension: u32,
}

impl NdBlockSpread {
    pub(crate) fn new(
        device: &wgpu::Device,
        kernel: EsKernel,
        fine_shape: &[usize],
        precision: FftPrecision,
        layout: NdSpreadLayout,
    ) -> Result<Self> {
        let types = NdWgsl::new(precision);
        let shaders = ShaderSource {
            types,
            kernel,
            fine_shape,
            layout: &layout,
        };
        let spread_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.nd_spread.light",
            &shaders.spread(SpreadMode::Light),
        );
        let part_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.nd_spread.heavy_part",
            &shaders.spread(SpreadMode::HeavyPart),
        );
        let classify_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.nd_spread.classify", &shaders.classify());
        let reduce_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.nd_spread.heavy_reduce",
            &shaders.reduce(),
        );
        let plan_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.nd_spread.heavy_plan", &shaders.plan());
        let reset_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.nd_spread.heavy_reset", HEAVY_RESET_WGSL);
        let block_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.nd_spread.block_state"),
            size: layout.block_count() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(Self {
            complex_bytes: types.complex_bytes() as u64,
            spread_layout: spread_pipeline.get_bind_group_layout(0),
            spread_pipeline,
            part_layout: part_pipeline.get_bind_group_layout(0),
            part_pipeline,
            classify_layout: classify_pipeline.get_bind_group_layout(0),
            classify_pipeline,
            reduce_layout: reduce_pipeline.get_bind_group_layout(0),
            reduce_pipeline,
            plan_layout: plan_pipeline.get_bind_group_layout(0),
            plan_pipeline,
            reset_layout: reset_pipeline.get_bind_group_layout(0),
            reset_pipeline,
            block_state,
            heavy_list: GrowOnlyBuffer::new(device, "wgpu_nufft.nd_spread.heavy_list"),
            part_offsets: GrowOnlyBuffer::new(device, "wgpu_nufft.nd_spread.part_offsets"),
            partials: Mutex::new(None),
            ranged: RangedDispatch::new(device),
            device: device.clone(),
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
            layout,
        })
    }

    /// Spreads `active_batch` transform-major strength vectors of the
    /// `point_count > 0` points prepared in `prepared` and `bin_offsets` into
    /// `fine_grid`, overwriting every active cell. With profiling, the end of
    /// the spread is written at offset 5 of the type-1 layout.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_spread(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        prepared: &NdPreparedPoints,
        bin_offsets: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        strength_bytes: u64,
        fine_grid: &wgpu::Buffer,
        active_fine_bytes: u64,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        self.encode_spread_passes(
            device,
            recorder,
            active_batch,
            point_count,
            prepared,
            bin_offsets,
            strengths,
            strength_bytes,
            fine_grid,
            active_fine_bytes,
        )?;
        #[cfg(feature = "gpu-profiling")]
        profile.encode_marker(recorder, None, Some(5));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_spread_passes(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        active_batch: usize,
        point_count: usize,
        prepared: &NdPreparedPoints,
        bin_offsets: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        strength_bytes: u64,
        fine_grid: &wgpu::Buffer,
        active_fine_bytes: u64,
    ) -> Result<()> {
        debug_assert!(point_count > 0);
        let limits = self.device.limits();
        let heavy = HeavyGeometry::new(
            &self.layout,
            point_count,
            active_batch,
            self.complex_bytes,
            limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
        );
        // The classify pass lists at most `capacity` blocks; bind exactly
        // that many entries after the count, so every pass can derive it.
        let heavy_bytes = (2 * heavy.capacity + 1) * 4;
        let heavy_list = self.heavy_list.get(heavy_bytes);
        let block_count = self.layout.block_count();
        let overflow = |context: &'static str| NufftError::LengthOverflow { context };

        let reset_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_spread.heavy_reset.bind_group"),
            layout: &self.reset_layout,
            entries: &[binding_entry(0, &heavy_list, 4)],
        });
        recorder.dispatch(&self.reset_pipeline, &reset_bind_group, (1, 1, 1));
        let classify_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_spread.classify.bind_group"),
            layout: &self.classify_layout,
            entries: &[
                binding_entry(0, bin_offsets, bin_offsets.size()),
                binding_entry(1, &heavy_list, heavy_bytes),
                binding_entry(2, &self.block_state, self.block_state.size()),
            ],
        });
        let classify_workgroups =
            u32::try_from(block_count.div_ceil(CLASSIFY_WORKGROUP_SIZE as usize))
                .map_err(|_| overflow("rank-generic spread classify workgroup count"))?;
        encode_pass(
            recorder,
            "wgpu_nufft.nd_spread.classify.pass",
            &self.classify_pipeline,
            &classify_bind_group,
            split_workgroups(classify_workgroups, self.max_workgroups_per_dimension)?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );

        let spread_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_spread.light.bind_group"),
            layout: &self.spread_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, &prepared.offsets, prepared.offset_bytes),
                binding_entry(2, strengths, strength_bytes),
                binding_entry(3, bin_offsets, bin_offsets.size()),
                binding_entry(4, fine_grid, active_fine_bytes),
                binding_entry(5, &self.block_state, self.block_state.size()),
            ],
        });
        let light_workgroups = block_count
            .checked_mul(active_batch)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| overflow("batched rank-generic spread workgroup count"))?;
        self.ranged.encode(
            device,
            recorder,
            &self.spread_pipeline,
            &spread_bind_group,
            u64::from(light_workgroups),
            self.layout.light_workgroups_per_dispatch(point_count),
        )?;
        if heavy.capacity == 0 {
            // No block can exceed the limit.
            return Ok(());
        }

        let block_cells = self.layout.block_cells() as u64;
        let partial_bytes =
            heavy.max_parts * active_batch as u64 * block_cells * self.complex_bytes;
        let partials = {
            let mut cached = self
                .partials
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match cached.as_ref() {
                Some(buffer) if buffer.size() >= partial_bytes => buffer.clone(),
                _ => {
                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("wgpu_nufft.nd_spread.heavy_partials"),
                        size: partial_bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    *cached = Some(buffer.clone());
                    buffer
                }
            }
        };
        let offset_bytes = (heavy.capacity + 1) * 4;
        let part_offsets = self.part_offsets.get(offset_bytes);
        let plan_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_spread.heavy_plan.bind_group"),
            layout: &self.plan_layout,
            entries: &[
                binding_entry(0, &heavy_list, heavy_bytes),
                binding_entry(1, &part_offsets, offset_bytes),
                binding_entry(2, &partials, partial_bytes),
                binding_entry(3, fine_grid, active_fine_bytes),
            ],
        });
        recorder.dispatch(&self.plan_pipeline, &plan_bind_group, (1, 1, 1));
        let part_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_spread.heavy_part.bind_group"),
            layout: &self.part_layout,
            entries: &[
                binding_entry(0, &prepared.starts, prepared.start_bytes),
                binding_entry(1, &prepared.offsets, prepared.offset_bytes),
                binding_entry(2, strengths, strength_bytes),
                binding_entry(3, bin_offsets, bin_offsets.size()),
                binding_entry(4, fine_grid, active_fine_bytes),
                binding_entry(5, &heavy_list, heavy_bytes),
                binding_entry(6, &partials, partial_bytes),
                binding_entry(7, &part_offsets, offset_bytes),
            ],
        });
        let reduce_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.nd_spread.heavy_reduce.bind_group"),
            layout: &self.reduce_layout,
            entries: &[
                binding_entry(0, fine_grid, active_fine_bytes),
                binding_entry(1, &heavy_list, heavy_bytes),
                binding_entry(2, &partials, partial_bytes),
                binding_entry(3, &part_offsets, offset_bytes),
            ],
        });
        let part_workgroups = heavy.max_parts * active_batch as u64;
        let reduce_workgroups = heavy.capacity
            * active_batch as u64
            * block_cells.div_ceil(u64::from(REDUCE_WORKGROUP_SIZE));
        let workgroup_count = |count: u64| {
            u32::try_from(count).map_err(|_| overflow("batched rank-generic heavy workgroup count"))
        };
        self.ranged.encode(
            device,
            recorder,
            &self.part_pipeline,
            &part_bind_group,
            u64::from(workgroup_count(part_workgroups)?),
            self.layout.part_workgroups_per_dispatch(heavy.part_records),
        )?;
        encode_pass(
            recorder,
            "wgpu_nufft.nd_spread.heavy_reduce.pass",
            &self.reduce_pipeline,
            &reduce_bind_group,
            split_workgroups(
                workgroup_count(reduce_workgroups)?,
                self.max_workgroups_per_dimension,
            )?,
            #[cfg(feature = "gpu-profiling")]
            None,
        );
        Ok(())
    }
}

/// Which records a spreading workgroup walks and where its sums go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpreadMode {
    /// One block and vector per workgroup, writing the fine grid; heavy
    /// blocks are left to the heavy passes.
    Light,
    /// One part of a listed heavy block per workgroup, writing partial sums.
    HeavyPart,
}

/// Resets the heavy-block count before the classify pass lists blocks.
const HEAVY_RESET_WGSL: &str =
    "@group(0) @binding(0) var<storage, read_write> heavy_list: array<u32>;

@compute @workgroup_size(1)
fn main() {
    heavy_list[0] = 0u;
}
";

/// Bins covering the cells `[low, high]` of a periodic axis as up to two
/// ranges, or every bin of the axis.
const BIN_RANGES_WGSL: &str = r#"
struct BinRanges {
    first_0: i32,
    last_0: i32,
    first_1: i32,
    last_1: i32,
}

fn bin_ranges(low: i32, high: i32, fine_length: i32, side: i32, bins: i32, full: bool) -> BinRanges {
    if (full) { return BinRanges(0, bins - 1, 0, -1); }
    if (low < 0) { return BinRanges((low + fine_length) / side, bins - 1, 0, high / side); }
    if (high >= fine_length) {
        return BinRanges(low / side, bins - 1, 0, (high - fine_length) / side);
    }
    return BinRanges(low / side, high / side, 0, -1);
}

fn bin_range_length(ranges: BinRanges) -> i32 {
    return (ranges.last_0 - ranges.first_0 + 1) + max(ranges.last_1 - ranges.first_1 + 1, 0);
}

fn bin_range_item(ranges: BinRanges, item: i32) -> i32 {
    let first_length = ranges.last_0 - ranges.first_0 + 1;
    if (item < first_length) { return ranges.first_0 + item; }
    return ranges.first_1 + (item - first_length);
}
"#;

struct ShaderSource<'a> {
    types: NdWgsl,
    kernel: EsKernel,
    fine_shape: &'a [usize],
    layout: &'a NdSpreadLayout,
}

impl ShaderSource<'_> {
    fn dimensions(&self) -> usize {
        self.fine_shape.len()
    }

    /// Constants of the block and bin geometry.
    fn geometry_constants(&self) -> String {
        let layout = self.layout;
        let width = layout.width;
        let mut source = format!(
            "const DIMS: u32 = {dimensions}u;
const STRIDE: u32 = {stride}u;
const RUN: i32 = {run}i;
const GROUPS_0: u32 = {groups_0}u;
const BLOCK_COUNT: u32 = {block_count}u;
const BLOCK_CELLS: u32 = {block_cells}u;
const REACH_BELOW: i32 = {reach_below}i;
const REACH_ABOVE: i32 = {reach_above}i;
const HEAVY_STATE: u32 = {HEAVY_STATE}u;
",
            dimensions = self.dimensions(),
            stride = self.dimensions() + 1,
            run = layout.run,
            groups_0 = layout.groups_0,
            block_count = layout.block_count(),
            block_cells = layout.block_cells(),
            // Bins hold support starts: a record reaches the block exactly
            // when its start lies at most `width - 1` cells below the block.
            reach_below = width - 1,
            reach_above = 0,
        );
        let mut bin_stride = 1usize;
        for axis in 0..self.dimensions() {
            let _ = write!(
                source,
                "const BLOCK_{axis}: i32 = {block}i;
const BLOCKS_{axis}: u32 = {blocks}u;
const BIN_SIDE_{axis}: i32 = {side}i;
const BINS_{axis}: i32 = {bins}i;
const BIN_STRIDE_{axis}: u32 = {bin_stride}u;
const FULL_{axis}: bool = {full};
",
                block = layout.block[axis],
                blocks = layout.blocks[axis],
                side = layout.bin_side[axis],
                bins = layout.bins[axis],
                full = layout.full[axis],
            );
            bin_stride *= layout.bins[axis];
        }
        source
    }

    /// Statements deriving `origin_{a}` from `block`.
    fn origin_statements(&self) -> String {
        let last = self.dimensions() - 1;
        let mut source = String::from("    var block_rest = block;\n");
        for axis in 0..=last {
            if axis == last {
                let _ = writeln!(
                    source,
                    "    let origin_{axis} = i32(block_rest) * BLOCK_{axis};"
                );
            } else {
                let _ = writeln!(
                    source,
                    "    let origin_{axis} = i32(block_rest % BLOCKS_{axis}) * BLOCK_{axis};\n    block_rest = block_rest / BLOCKS_{axis};"
                );
            }
        }
        source
    }

    /// Statements deriving the invocation's `run_origin` (cells along axis
    /// zero, relative to the block) and `local_{a}` for the other axes from
    /// `invocation`.
    fn local_statements(&self) -> String {
        let last = self.dimensions() - 1;
        let mut source = String::from(
            "    var local_rest = invocation;\n    let run_origin = i32(local_rest % GROUPS_0) * RUN;\n    local_rest = local_rest / GROUPS_0;\n",
        );
        for axis in 1..=last {
            if axis == last {
                let _ = writeln!(source, "    let local_{axis} = i32(local_rest);");
            } else {
                let _ = writeln!(
                    source,
                    "    let local_{axis} = i32(local_rest % u32(BLOCK_{axis}));\n    local_rest = local_rest / u32(BLOCK_{axis});"
                );
            }
        }
        source
    }

    /// Statements computing `ranges_{a}` for every axis, `row_count_{a}` for
    /// the axes above zero, and `rows`.
    fn reach_statements(&self) -> String {
        let mut source = String::new();
        for axis in 0..self.dimensions() {
            let _ = writeln!(
                source,
                "    let ranges_{axis} = bin_ranges(origin_{axis} - REACH_BELOW, origin_{axis} + BLOCK_{axis} - 1 + REACH_ABOVE, FINE_{axis}_I32, BIN_SIDE_{axis}, BINS_{axis}, FULL_{axis});"
            );
        }
        let mut rows = Vec::new();
        for axis in 1..self.dimensions() {
            let _ = writeln!(
                source,
                "    let row_count_{axis} = bin_range_length(ranges_{axis});"
            );
            rows.push(format!("row_count_{axis}"));
        }
        let rows = if rows.is_empty() {
            "1".to_owned()
        } else {
            rows.join(" * ")
        };
        let _ = writeln!(source, "    var rows = u32({rows});");
        source
    }

    /// Statements deriving `row_bin`, the first bin of row `row` of the
    /// reach, and loading its axis-zero segments into `start_a`, `length_a`,
    /// `start_b` and `length_b`.
    fn row_statements(&self, indent: &str) -> String {
        let last = self.dimensions() - 1;
        let mut source = format!("{indent}var row_rest = i32(row);\n");
        let mut terms = Vec::new();
        for axis in 1..=last {
            if axis == last {
                let _ = writeln!(
                    source,
                    "{indent}let bin_{axis} = bin_range_item(ranges_{axis}, row_rest);"
                );
            } else {
                let _ = writeln!(
                    source,
                    "{indent}let bin_{axis} = bin_range_item(ranges_{axis}, row_rest % row_count_{axis});\n{indent}row_rest = row_rest / row_count_{axis};"
                );
            }
            terms.push(format!("u32(bin_{axis}) * BIN_STRIDE_{axis}"));
        }
        let row_bin = if terms.is_empty() {
            "0u".to_owned()
        } else {
            terms.join(" + ")
        };
        let _ = write!(
            source,
            "{indent}let row_bin = {row_bin};
{indent}start_a = bin_offsets[row_bin + u32(ranges_0.first_0)];
{indent}length_a = bin_offsets[row_bin + u32(ranges_0.last_0) + 1u] - start_a;
{indent}if (ranges_0.last_1 >= ranges_0.first_1) {{
{indent}    start_b = bin_offsets[row_bin + u32(ranges_0.first_1)];
{indent}    length_b = bin_offsets[row_bin + u32(ranges_0.last_1) + 1u] - start_b;
{indent}}}
"
        );
        source
    }

    fn spread(&self, mode: SpreadMode) -> String {
        let types = self.types;
        let layout = self.layout;
        let dimensions = self.dimensions();
        let complex = types.complex_type();
        let zero = types.complex_zero();
        let run = layout.run;
        let initialize: String = (0..run)
            .map(|cell| format!("    var sum_{cell} = {zero};\n"))
            .collect();

        // Loading one record into chunk slot `slot`.
        let relatives: String = (0..dimensions)
            .map(|axis| {
                format!(
                    "                let relative_{axis} = relative_start(prepared_starts[start_base + {axis}u], origin_{axis}, FINE_{axis}_I32);\n"
                )
            })
            .collect();
        let keep = (0..dimensions)
            .map(|axis| format!("relative_{axis} < BLOCK_{axis}"))
            .collect::<Vec<_>>()
            .join(" && ");
        let store_relatives: String = (0..dimensions)
            .map(|axis| {
                format!(
                    "                chunk_starts[slot * DIMS + {axis}u] = select(SKIPPED, relative_{axis}, keep);\n"
                )
            })
            .collect();

        // The record's kernel weights, shared by its lanes.
        let weights = if types.precision() == FftPrecision::F32 {
            // Constant loops let the compiler overlap the `exp` evaluations.
            (0..dimensions)
                .map(|axis| {
                    format!(
                        "                    let offset_{axis} = prepared_offsets[sorted_slot * DIMS + {axis}u];
                    for (var support = 0u; support < WIDTH; support = support + 1u) {{
                        chunk_weights[(slot * DIMS + {axis}u) * WIDTH + support] = support_weight(offset_{axis}, support);
                    }}
"
                    )
                })
                .collect()
        } else {
            // One loop over every (axis, support) weight, with a bound the
            // compiler cannot unroll: unrolled Horner evaluations made these
            // shaders slow to compile on DX12.
            "                    for (var entry = lane; entry < weight_entries; entry = entry + RECORD_LANES) {{
                        let axis = entry / WIDTH;
                        chunk_weights[slot * DIMS * WIDTH + entry] = support_weight(
                            prepared_offsets[sorted_slot * DIMS + axis],
                            entry - axis * WIDTH,
                        );
                    }}
"
            .to_owned()
        };

        // Adding one staged record to this invocation's cells.
        let deltas: String = (1..dimensions)
            .map(|axis| {
                format!(
                    "                let delta_{axis} = local_{axis} - chunk_starts[base + {axis}u];\n"
                )
            })
            .collect();
        let inside = if dimensions == 1 {
            "true".to_owned()
        } else {
            (1..dimensions)
                .map(|axis| format!("u32(delta_{axis}) < WIDTH"))
                .collect::<Vec<_>>()
                .join(" && ")
        };
        let value = if dimensions == 1 {
            "chunk_values[slot]".to_owned()
        } else {
            let mut weight = "chunk_weights[(base + 1u) * WIDTH + u32(delta_1)]".to_owned();
            for axis in 2..dimensions {
                weight = types.weight_product(
                    &weight,
                    &format!("chunk_weights[(base + {axis}u) * WIDTH + u32(delta_{axis})]"),
                );
            }
            types.complex_scale("chunk_values[slot]", &weight)
        };
        let accumulate: String = (0..run)
            .map(|cell| {
                format!(
                    "                    {{ let dx = delta_0 + {cell}; if (u32(dx) < WIDTH) {{ sum_{cell} = {}; }} }}\n",
                    types.complex_add(
                        &format!("sum_{cell}"),
                        &types.complex_scale("value", "chunk_weights[weights_0 + u32(dx)]")
                    )
                )
            })
            .collect();

        let (mode_bindings, locate, store) = match mode {
            SpreadMode::Light => {
                let cells: String = (1..dimensions)
                    .map(|axis| format!("    let cell_{axis} = origin_{axis} + local_{axis};\n"))
                    .collect();
                let in_grid = if dimensions == 1 {
                    "true".to_owned()
                } else {
                    (1..dimensions)
                        .map(|axis| format!("cell_{axis} < FINE_{axis}_I32"))
                        .collect::<Vec<_>>()
                        .join(" && ")
                };
                let row_index = if dimensions == 1 {
                    "0u".to_owned()
                } else {
                    (1..dimensions)
                        .map(|axis| format!("u32(cell_{axis}) * FINE_STRIDE_{axis}"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
                let writes: String = (0..run)
                    .map(|cell| {
                        format!(
                            "        {{ let cell_0 = origin_0 + run_origin + {cell}; if (cell_0 < FINE_0_I32) {{ fine_grid[row_index + u32(cell_0)] = sum_{cell}; }} }}\n"
                        )
                    })
                    .collect();
                (
                    "@group(0) @binding(5) var<storage, read> block_state: array<u32>;\n"
                        .to_owned(),
                    "    if (wg_flat >= BLOCK_COUNT * total_vectors) { return; }
    let vector_index = wg_flat / BLOCK_COUNT;
    let block = wg_flat - vector_index * BLOCK_COUNT;
    let state = block_state[block];
    // Heavy blocks belong to the heavy passes.
    if (state == HEAVY_STATE) { return; }
    let records_start = 0u;
    let records_end = state;
"
                    .to_owned(),
                    format!(
                        "{cells}    if ({in_grid}) {{
        let row_index = vector_index * FINE_COUNT + {row_index};
{writes}    }}
"
                    ),
                )
            }
            SpreadMode::HeavyPart => {
                let writes: String = (0..run)
                    .map(|cell| format!("    partials[partial_base + {cell}u] = sum_{cell};\n"))
                    .collect();
                (
                    format!(
                        "@group(0) @binding(5) var<storage, read> heavy_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> partials: array<{complex}>;
@group(0) @binding(7) var<storage, read> part_offsets: array<u32>;
"
                    ),
                    "    // Workgroups enumerate (vector, part slot); slots past the planned parts
    // return. The partials binding fixes the slots per vector.
    let capacity = (arrayLength(&heavy_list) - 1u) / 2u;
    let count = min(heavy_list[0], capacity);
    let max_parts = arrayLength(&partials) / (total_vectors * BLOCK_CELLS);
    let vector_index = wg_flat / max_parts;
    let slot = wg_flat - vector_index * max_parts;
    if (vector_index >= total_vectors || slot >= part_offsets[count]) { return; }
    // The listed block of this slot: the last one whose parts start at or
    // before it.
    var low = 0u;
    var high = count - 1u;
    loop {
        if (low >= high) { break; }
        let middle = (low + high + 1u) / 2u;
        if (part_offsets[middle] <= slot) { low = middle; } else { high = middle - 1u; }
    }
    let entry = low;
    let block = heavy_list[1u + 2u * entry];
    let block_records = heavy_list[2u + 2u * entry];
    let parts = part_offsets[entry + 1u] - part_offsets[entry];
    let part = slot - part_offsets[entry];
    // Part `part` of `parts` equal runs of the reach, in reach order.
    let records_start = part * (block_records / parts) + min(part, block_records % parts);
    let records_end = (part + 1u) * (block_records / parts) + min(part + 1u, block_records % parts);
"
                    .to_owned(),
                    format!(
                        "    // Partial sums of this part, RUN cells per invocation.
    let partial_base = (vector_index * max_parts + slot) * BLOCK_CELLS + invocation * u32(RUN);
{writes}"
                    ),
                )
            }
        };

        let entry = format!(
            r#"const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const ROW_BATCH: u32 = {row_batch}u;
const CHUNK: u32 = {chunk}u;
const RECORD_LANES: u32 = {record_lanes}u;
// The relative start of a staged record whose support misses the block.
const SKIPPED: i32 = 0x3fffffff;
{geometry}
@group(0) @binding(0) var<storage, read> prepared_starts: array<i32>;
@group(0) @binding(1) var<storage, read> prepared_offsets: array<{offset_type}>;
@group(0) @binding(2) var<storage, read> strengths: array<{complex}>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<{complex}>;
{mode_bindings}{DISPATCH_RANGE_WGSL}
var<workgroup> row_start_a: array<u32, {row_batch}>;
var<workgroup> row_length_a: array<u32, {row_batch}>;
var<workgroup> row_start_b: array<u32, {row_batch}>;
var<workgroup> row_prefix: array<u32, {row_batch}>;
var<workgroup> chunk_starts: array<i32, {chunk_starts}>;
var<workgroup> chunk_values: array<{complex}, {chunk}>;
var<workgroup> chunk_weights: array<{weight_type}, {chunk_weights}>;
{BIN_RANGES_WGSL}
// A support start relative to the block origin, as its representative in
// [-(w-1), n-w]; supports that reach the block start in [-(w-1), BLOCK-1].
fn relative_start(start: i32, origin: i32, fine_length: i32) -> i32 {{
    var relative = start - origin;
    if (relative < 0) {{ relative = relative + fine_length; }}
    if (relative < 0) {{ relative = relative + fine_length; }}
    if (relative >= fine_length) {{ relative = relative - fine_length; }}
    if (relative > fine_length - WIDTH_I32) {{ relative = relative - fine_length; }}
    return relative;
}}

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(local_invocation_index) invocation: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = dispatch_range.first + (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let point_count = arrayLength(&prepared_starts) / STRIDE;
    // DIMS * WIDTH, but not a constant the compiler can unroll loops by:
    // point counts stay below 2^31.
    let weight_entries = DIMS * WIDTH + (point_count >> 31u);
{locate}{origins}{locals}{reach}    if (records_end == 0u) {{ rows = 0u; }}
{initialize}
    // Records already walked in earlier row batches.
    var consumed = 0u;
    for (var batch_first = 0u; batch_first < rows; batch_first = batch_first + ROW_BATCH) {{
        if (consumed >= records_end) {{ break; }}
        if (invocation < ROW_BATCH) {{
            let row = batch_first + invocation;
            var start_a = 0u;
            var length_a = 0u;
            var start_b = 0u;
            var length_b = 0u;
            if (row < rows) {{
{row}            }}
            row_start_a[invocation] = start_a;
            row_length_a[invocation] = length_a;
            row_start_b[invocation] = start_b;
            row_prefix[invocation] = length_a + length_b;
        }}
        workgroupBarrier();
        // Inclusive scan of the row lengths.
        for (var offset = 1u; offset < ROW_BATCH; offset = offset * 2u) {{
            var addend = 0u;
            if (invocation < ROW_BATCH && invocation >= offset) {{
                addend = row_prefix[invocation - offset];
            }}
            workgroupBarrier();
            if (invocation < ROW_BATCH) {{
                row_prefix[invocation] = row_prefix[invocation] + addend;
            }}
            workgroupBarrier();
        }}
        let batch_records = workgroupUniformLoad(&row_prefix[ROW_BATCH - 1u]);
        let batch_end = consumed + batch_records;
        var local_first = 0u;
        var local_last = 0u;
        if (records_start < batch_end && records_end > consumed) {{
            local_first = max(records_start, consumed) - consumed;
            local_last = min(records_end, batch_end) - consumed;
        }}
        for (var chunk_first = local_first; chunk_first < local_last; chunk_first = chunk_first + CHUNK) {{
            let chunk_length = min(CHUNK, local_last - chunk_first);
            // RECORD_LANES invocations stage each record and share its kernel
            // weights; every chunk slot is filled, whether or not CHUNK exceeds
            // the workgroup.
            for (var lane_index = invocation; lane_index < chunk_length * RECORD_LANES;
                 lane_index = lane_index + WORKGROUP_SIZE) {{
                let slot = lane_index / RECORD_LANES;
                let lane = lane_index - slot * RECORD_LANES;
                let record = chunk_first + slot;
                // First row whose inclusive prefix exceeds `record`.
                var low = 0u;
                var high = ROW_BATCH - 1u;
                loop {{
                    if (low >= high) {{ break; }}
                    let middle = (low + high) / 2u;
                    if (row_prefix[middle] > record) {{ high = middle; }} else {{ low = middle + 1u; }}
                }}
                let within = record - select(0u, row_prefix[max(low, 1u) - 1u], low > 0u);
                var sorted_slot = row_start_a[low] + within;
                if (within >= row_length_a[low]) {{
                    sorted_slot = row_start_b[low] + (within - row_length_a[low]);
                }}
                let start_base = sorted_slot * STRIDE;
{relatives}                let keep = {keep};
                if (lane == 0u) {{
{store_relatives}                    if (keep) {{
                        let point_index = bitcast<u32>(prepared_starts[start_base + DIMS]);
                        chunk_values[slot] = strengths[vector_index * point_count + point_index];
                    }}
                }}
                if (keep) {{
{weights}                }}
            }}
            workgroupBarrier();
            for (var slot = 0u; slot < chunk_length; slot = slot + 1u) {{
                let base = slot * DIMS;
                let delta_0 = run_origin - chunk_starts[base];
{deltas}                // The record's support must overlap this invocation's run.
                if (u32(delta_0 + RUN - 1) < WIDTH + u32(RUN) - 1u && {inside}) {{
                    let value = {value};
                    let weights_0 = base * WIDTH;
{accumulate}                }}
            }}
            workgroupBarrier();
        }}
        consumed = batch_end;
        workgroupBarrier();
    }}

{store}}}
"#,
            workgroup_size = layout.workgroup_size,
            row_batch = layout.row_batch,
            chunk = layout.chunk,
            record_lanes = layout.record_lanes(types.precision()),
            geometry = self.geometry_constants(),
            offset_type = types.offset_type(),
            weight_type = types.weight_type(),
            chunk_starts = layout.chunk * dimensions,
            chunk_weights = layout.chunk * dimensions * layout.width,
            origins = self.origin_statements(),
            locals = self.local_statements(),
            reach = self.reach_statements(),
            row = self.row_statements("                "),
        );
        types.with_library(&format!(
            "{}\n{}\n{entry}",
            position_wgsl(self.fine_shape, self.kernel, types.precision()),
            weight_wgsl(self.kernel, types.precision()),
        ))
    }

    /// Counts the records in every block's reach, flags the blocks above the
    /// light limit and lists them, one invocation per block.
    ///
    /// All reaches hold at most `BLOCKS_PER_BIN * point_count` records, so at
    /// most `capacity` blocks exceed `floor(that / (capacity + 1))`: raising
    /// the limit to it keeps the list within its binding and depends only on
    /// the point count and the binding.
    fn classify(&self) -> String {
        let entry = format!(
            r#"const WORKGROUP_SIZE: u32 = {CLASSIFY_WORKGROUP_SIZE}u;
const HEAVY_RECORDS: u32 = {heavy_records}u;
const MIN_PART_RECORDS: u32 = {MIN_PART_RECORDS}u;
const DENSE_DIVISOR: u32 = {dense_divisor}u;
const BLOCKS_PER_BIN: u32 = {blocks_per_bin}u;
{geometry}
@group(0) @binding(0) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> heavy_list: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> block_state: array<u32>;
{BIN_RANGES_WGSL}
@compute @workgroup_size({CLASSIFY_WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) invocation: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let block = wg_flat * WORKGROUP_SIZE + invocation;
    if (block >= BLOCK_COUNT) {{ return; }}
{origins}{reach}    var total = 0u;
    for (var row = 0u; row < rows; row = row + 1u) {{
        var start_a = 0u;
        var length_a = 0u;
        var start_b = 0u;
        var length_b = 0u;
{row}        total = total + length_a + length_b;
    }}

    let point_count = bin_offsets[arrayLength(&bin_offsets) - 1u];
    let capacity = (arrayLength(&heavy_list) - 1u) / 2u;
    // The same limit as the host's: lower on grids with few blocks. A list
    // with room for every block needs no raise.
    let base_limit = clamp(point_count / DENSE_DIVISOR, MIN_PART_RECORDS, HEAVY_RECORDS);
    var limit = HEAVY_STATE;
    if (capacity >= BLOCK_COUNT) {{
        limit = base_limit;
    }} else {{
        let per_entry = point_count / (capacity + 1u);
        if (capacity > 0u && per_entry < 0xffffffffu / BLOCKS_PER_BIN) {{
            limit = max(base_limit, BLOCKS_PER_BIN * per_entry
                + BLOCKS_PER_BIN * (point_count % (capacity + 1u)) / (capacity + 1u));
        }}
    }}
    var state = min(total, HEAVY_STATE - 1u);
    if (total > limit) {{
        let slot = atomicAdd(&heavy_list[0], 1u);
        if (slot < capacity) {{
            atomicStore(&heavy_list[1u + 2u * slot], block);
            atomicStore(&heavy_list[2u + 2u * slot], total);
            state = HEAVY_STATE;
        }}
    }}
    block_state[block] = state;
}}
"#,
            heavy_records = self.layout.heavy_records,
            dense_divisor = self.layout.dense_divisor,
            blocks_per_bin = u32::try_from(self.layout.blocks_per_bin()).unwrap_or(u32::MAX),
            geometry = self.geometry_constants(),
            origins = self.origin_statements(),
            reach = self.reach_statements(),
            row = self.row_statements("        "),
        );
        let position = position_wgsl(self.fine_shape, self.kernel, self.types.precision());
        self.types.with_library(&format!("{position}\n{entry}"))
    }

    /// Plans the heavy parts in one workgroup. The part size comes from the
    /// listed records, an exact 64-bit sum that does not depend on the list
    /// order: it starts at the size that fills every part slot and grows in
    /// small steps while the parts, `ceil(records / part size)` per block,
    /// outgrow the slots, up to the size that leaves one slot per block for
    /// rounding up, which always fits; doubling covers `f32` rounding. An
    /// exclusive scan of the part counts then gives every block its first
    /// part slot, then the total.
    fn plan(&self) -> String {
        let entry = format!(
            r#"const WORKGROUP_SIZE: u32 = {PLAN_WORKGROUP_SIZE}u;
const MIN_PART_RECORDS: u32 = {MIN_PART_RECORDS}u;
const BLOCK_CELLS: u32 = {block_cells}u;
const FINE_COUNT: u32 = {fine_count}u;

@group(0) @binding(0) var<storage, read> heavy_list: array<u32>;
@group(0) @binding(1) var<storage, read_write> part_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> partials: array<{complex}>;
@group(0) @binding(3) var<storage, read> fine_grid: array<{complex}>;

var<workgroup> scan: array<u32, {PLAN_WORKGROUP_SIZE}>;
var<workgroup> scan_high: array<u32, {PLAN_WORKGROUP_SIZE}>;
var<workgroup> shared_count: u32;
var<workgroup> shared_records: u32;
var<workgroup> shared_safe: u32;
var<workgroup> running: u32;

fn saturating_add(a: u32, b: u32) -> u32 {{
    return select(a + b, 0xffffffffu, a > 0xffffffffu - b);
}}

fn parts_of(total: u32, records: u32) -> u32 {{
    return max(1u, total / records + select(0u, 1u, total % records != 0u));
}}

@compute @workgroup_size({PLAN_WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) lid: u32) {{
    let capacity = (arrayLength(&heavy_list) - 1u) / 2u;
    let vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let max_parts = arrayLength(&partials) / (vectors * BLOCK_CELLS);
    if (lid == 0u) {{
        shared_count = min(heavy_list[0], capacity);
        running = 0u;
    }}
    let count = workgroupUniformLoad(&shared_count);

    // Exact 64-bit sum of the listed records.
    var low = 0u;
    var high = 0u;
    for (var entry = lid; entry < count; entry = entry + WORKGROUP_SIZE) {{
        let sum = low + heavy_list[2u + 2u * entry];
        high = high + select(0u, 1u, sum < low);
        low = sum;
    }}
    scan[lid] = low;
    scan_high[lid] = high;
    workgroupBarrier();
    for (var half = WORKGROUP_SIZE / 2u; half > 0u; half = half / 2u) {{
        if (lid < half) {{
            let sum = scan[lid] + scan[lid + half];
            scan_high[lid] = scan_high[lid] + scan_high[lid + half] + select(0u, 1u, sum < scan[lid]);
            scan[lid] = sum;
        }}
        workgroupBarrier();
    }}
    if (lid == 0u) {{
        // First the part size that fills every slot, which fits when the
        // blocks divide evenly; then the one that leaves a slot per block for
        // rounding up, which always fits. `count` never exceeds `capacity`,
        // nor `capacity` the slots.
        let listed = f32(scan_high[0]) * 4294967296.0 + f32(scan[0]);
        let spare = max_parts - count;
        var safe = 0xffffffffu;
        if (spare > 0u) {{
            safe = u32(clamp(ceil(listed / f32(spare)), f32(MIN_PART_RECORDS), 4294967040.0));
        }}
        shared_records = u32(clamp(ceil(listed / f32(max_parts)), f32(MIN_PART_RECORDS), 4294967040.0));
        shared_safe = safe;
    }}
    var records = workgroupUniformLoad(&shared_records);
    let safe = workgroupUniformLoad(&shared_safe);
    for (var attempt = 0u; attempt < 64u; attempt = attempt + 1u) {{
        var parts = 0u;
        for (var entry = lid; entry < count; entry = entry + WORKGROUP_SIZE) {{
            parts = saturating_add(parts, parts_of(heavy_list[2u + 2u * entry], records));
        }}
        workgroupBarrier();
        scan[lid] = parts;
        workgroupBarrier();
        for (var half = WORKGROUP_SIZE / 2u; half > 0u; half = half / 2u) {{
            if (lid < half) {{ scan[lid] = saturating_add(scan[lid], scan[lid + half]); }}
            workgroupBarrier();
        }}
        let needed = workgroupUniformLoad(&scan[0]);
        if (needed <= max_parts) {{ break; }}
        if (records < safe) {{
            // Grow by about 3%: the smallest fitting part size keeps the most
            // parts, and every wave of workgroups short.
            records = min(safe, max(records + 1u, records + records / 32u));
        }} else {{
            records = select(records * 2u, 0xffffffffu, records >= 0x80000000u);
        }}
    }}

    for (var first = 0u; first < count; first = first + WORKGROUP_SIZE) {{
        let entry = first + lid;
        var parts = 0u;
        if (entry < count) {{ parts = parts_of(heavy_list[2u + 2u * entry], records); }}
        workgroupBarrier();
        scan[lid] = parts;
        workgroupBarrier();
        for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {{
            var addend = 0u;
            if (lid >= offset) {{ addend = scan[lid - offset]; }}
            workgroupBarrier();
            scan[lid] = scan[lid] + addend;
            workgroupBarrier();
        }}
        let before = workgroupUniformLoad(&running);
        if (entry < count) {{ part_offsets[entry] = before + scan[lid] - parts; }}
        workgroupBarrier();
        if (lid == WORKGROUP_SIZE - 1u) {{ running = before + scan[lid]; }}
    }}
    let total = workgroupUniformLoad(&running);
    if (lid == 0u) {{ part_offsets[count] = total; }}
}}
"#,
            block_cells = self.layout.block_cells(),
            fine_count = self.fine_shape.iter().product::<usize>(),
            complex = self.types.complex_type(),
        );
        self.types.with_library(&entry)
    }

    /// Adds the parts of every cell of every listed block in part order and
    /// writes the cell, one invocation per (listed block, vector, cell).
    fn reduce(&self) -> String {
        let types = self.types;
        let dimensions = self.dimensions();
        let cells: String = (1..dimensions)
            .map(|axis| format!("    let cell_{axis} = origin_{axis} + local_{axis};\n"))
            .collect();
        let in_grid = std::iter::once("cell_0 < FINE_0_I32".to_owned())
            .chain((1..dimensions).map(|axis| format!("cell_{axis} < FINE_{axis}_I32")))
            .collect::<Vec<_>>()
            .join(" && ");
        let index = std::iter::once("u32(cell_0)".to_owned())
            .chain((1..dimensions).map(|axis| format!("u32(cell_{axis}) * FINE_STRIDE_{axis}")))
            .collect::<Vec<_>>()
            .join(" + ");
        let sum = types.complex_add("sum", "partials[base + slot * BLOCK_CELLS]");
        let entry = format!(
            r#"const WORKGROUP_SIZE: u32 = {REDUCE_WORKGROUP_SIZE}u;
{geometry}
@group(0) @binding(0) var<storage, read_write> fine_grid: array<{complex}>;
@group(0) @binding(1) var<storage, read> heavy_list: array<u32>;
@group(0) @binding(2) var<storage, read> partials: array<{complex}>;
@group(0) @binding(3) var<storage, read> part_offsets: array<u32>;

@compute @workgroup_size({REDUCE_WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let capacity = (arrayLength(&heavy_list) - 1u) / 2u;
    let max_parts = arrayLength(&partials) / (total_vectors * BLOCK_CELLS);
    let groups = (BLOCK_CELLS + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let entry = wg_flat / (total_vectors * groups);
    if (entry >= min(heavy_list[0], capacity)) {{ return; }}
    let vector_index = (wg_flat / groups) % total_vectors;
    let cell = (wg_flat % groups) * WORKGROUP_SIZE + lid;
    if (cell >= BLOCK_CELLS) {{ return; }}
    let base = vector_index * max_parts * BLOCK_CELLS + cell;
    var sum = {zero};
    for (var slot = part_offsets[entry]; slot < part_offsets[entry + 1u]; slot = slot + 1u) {{
        sum = {sum};
    }}
    // Cells follow the spreading invocations: RUN cells per invocation.
    let block = heavy_list[1u + 2u * entry];
{origins}    let invocation = cell / u32(RUN);
{locals}    let cell_0 = origin_0 + run_origin + i32(cell % u32(RUN));
{cells}    if ({in_grid}) {{
        fine_grid[vector_index * FINE_COUNT + {index}] = sum;
    }}
}}
"#,
            geometry = self.geometry_constants(),
            complex = types.complex_type(),
            zero = types.complex_zero(),
            origins = self.origin_statements(),
            locals = self.local_statements(),
        );
        let position = position_wgsl(self.fine_shape, self.kernel, types.precision());
        types.with_library(&format!("{position}\n{entry}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_nd_wgsl::assert_valid_nd_wgsl;

    fn shaders(precision: FftPrecision, fine_shape: &[usize], eps: f64) -> Vec<String> {
        let kernel = EsKernel::for_tolerance(eps, 2.0).unwrap();
        let layout =
            NdSpreadLayout::for_grid(kernel, fine_shape, precision, &wgpu::Limits::default())
                .unwrap();
        let source = ShaderSource {
            types: NdWgsl::new(precision),
            kernel,
            fine_shape,
            layout: &layout,
        };
        vec![
            source.spread(SpreadMode::Light),
            source.spread(SpreadMode::HeavyPart),
            source.classify(),
            source.reduce(),
            source.plan(),
        ]
    }

    #[test]
    fn spread_shaders_validate_in_every_precision_and_rank() {
        for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
            for shape in [
                vec![64],
                vec![24, 20],
                vec![32, 16, 20],
                vec![24, 20, 16, 16],
                vec![16, 14, 14, 14, 14],
                vec![14; 6],
                vec![10, 8, 8, 8, 8, 8, 8, 8],
            ] {
                let eps = if shape.len() > 5 { 1.0e-3 } else { 1.0e-6 };
                for source in shaders(precision, &shape, eps) {
                    assert_valid_nd_wgsl(precision, &source);
                }
            }
        }
        crate::wgsl_validation::assert_valid_wgsl(HEAVY_RESET_WGSL);
    }

    #[test]
    fn light_shader_writes_each_cell_once_without_atomics() {
        let sources = shaders(FftPrecision::F32, &[24, 20, 16, 16], 1.0e-6);
        let light = &sources[0];
        assert!(!light.contains("atomic"));
        assert!(light.contains("if (state == HEAVY_STATE) { return; }"));
        assert!(light.contains("fine_grid[row_index + u32(cell_0)] = sum_"));
        assert!(!sources[1].contains("fine_grid[row_index"));
        assert!(!sources[1].contains("atomic"));
        assert_eq!(sources[2].matches("atomic").count(), 4);
    }

    #[test]
    fn layouts_keep_support_starts_unique_and_fit_webgpu_defaults() {
        let limits = wgpu::Limits::default();
        for (shape, eps) in [
            (vec![128usize], 1.0e-6),
            (vec![8, 8, 8, 8], 1.0e-3),
            (vec![32, 32, 32, 32], 1.0e-6),
            (vec![20; 5], 1.0e-6),
            (vec![14; 6], 1.0e-6),
            (vec![8; 8], 1.0e-3),
            (vec![32; 8], 1.0e-12),
        ] {
            for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
                let kernel = EsKernel::for_tolerance(eps, 2.0).unwrap();
                let layout = NdSpreadLayout::for_grid(kernel, &shape, precision, &limits).unwrap();
                let width = kernel.width();
                for (axis, &length) in shape.iter().enumerate() {
                    assert!(
                        layout.block[axis] + width - 1 <= length,
                        "{shape:?} axis {axis}"
                    );
                }
                assert!(layout.workgroup_size <= 256);
                let types = NdWgsl::new(precision);
                let bytes = 16 * layout.row_batch
                    + layout.chunk
                        * (4 * shape.len()
                            + types.complex_bytes()
                            + shape.len() * width * types.weight_bytes());
                assert!(bytes as u32 <= limits.max_compute_workgroup_storage_size);
                assert!(layout.heavy_records >= MIN_HEAVY_RECORDS as u32);
            }
        }
    }

    #[test]
    fn four_dimensional_f32_layout_uses_long_runs() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let layout = NdSpreadLayout::for_grid(
            kernel,
            &[64, 64, 64, 64],
            FftPrecision::F32,
            &wgpu::Limits::default(),
        )
        .unwrap();
        assert_eq!(layout.run, 16);
        assert_eq!(layout.workgroup_size, 64);
        assert_eq!(layout.block, vec![16, 4, 4, 4]);
        assert_eq!(layout.block_cells(), 1024);
    }

    #[test]
    fn low_rank_layouts_use_short_runs() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let limits = wgpu::Limits::default();
        let one = NdSpreadLayout::for_grid(kernel, &[4096], FftPrecision::Df64, &limits).unwrap();
        assert_eq!((one.run, one.workgroup_size), (2, 64));
        assert_eq!(one.block, vec![128]);
        let two =
            NdSpreadLayout::for_grid(kernel, &[512, 512], FftPrecision::F32, &limits).unwrap();
        assert_eq!((two.run, two.workgroup_size), (4, 32));
        assert_eq!(two.block, vec![8, 16]);
    }

    #[test]
    fn bins_coarsen_to_bound_reach_rows_and_bin_count() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let limits = wgpu::Limits::default();
        let layout = |shape: &[usize]| {
            NdSpreadLayout::for_grid(kernel, shape, FftPrecision::F32, &limits).unwrap()
        };
        // Single cells above axis zero while a reach spans few bin rows.
        assert_eq!(layout(&[2000, 2000]).bin_side, vec![1, 1]);
        assert_eq!(layout(&[200, 200, 200]).bin_side, vec![2, 1, 1]);
        assert_eq!(layout(&[32; 4]).bin_side, vec![2, 1, 1, 1]);
        // More than `MAX_BINS` bins coarsen along the widest reaches.
        assert_eq!(layout(&[400, 400, 400]).bin_side, vec![2, 2, 2]);
        // Single cells along four axes would span 6400 rows per reach.
        let five = layout(&[24; 5]);
        assert_eq!(five.block, vec![16, 4, 4, 2, 2]);
        assert_eq!(five.bin_side, vec![2, 2, 2, 2, 2]);
    }
}
