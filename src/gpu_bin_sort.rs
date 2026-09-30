//! Parallel, deterministic ordering of large coarse bins.
//!
//! The binning passes place the points of every bin at slots taken with
//! integer atomics, so the order inside a bin depends on scheduling.
//! Consumers restore original point order: bins of up to [`SMALL_BIN`]
//! points are ordered by ranking (each point counts the smaller point
//! indices of its bin), and [`LargeBinSort`] sorts the larger ones.
//!
//! A single invocation per bin would serialize a cluster of millions of
//! points and could run long enough to trip the Windows TDR watchdog, so the
//! sort spreads every bin across all invocations. A first stage finds every
//! slot's bin once, by binary search over the bin offsets, and ranks each
//! point inside its [`CHUNK`]-slot chunk of the bin, which leaves sorted runs
//! of [`CHUNK`]. Each following stage merges groups of [`FAN_IN`] runs: a
//! point keeps its offset inside its own run and adds, for every other run of
//! its group, how many of that run's indices are smaller (a binary search).
//! Few runs per group keep the searches per point low; runs grow fourfold per
//! stage, and one pipeline serves every stage, reading its run length from a
//! uniform at bind group 1.
//!
//! The stage count is fixed at encode time from the point count. The binning
//! passes set a flag, the last word of a buffer they share with the sort,
//! when any bin exceeds [`SMALL_BIN`] points, to at least half the size of
//! the largest bin. Without it, every stage returns at once, so evenly spread
//! points pay only for a few empty dispatches; with it, merge stages whose
//! runs already hold the largest bin return at once too, and the final copy
//! follows the parity of the merges that ran.
//! [`generate_small_bin_sort_wgsl`] is such a pass for binnings that do not
//! rank small bins elsewhere.

use std::num::NonZeroU64;
use std::sync::Mutex;

use wgpu::util::DeviceExt;

use crate::error::{NufftError, Result};
use crate::gpu_recorder::GpuRecorder;

/// Largest bin that consumers order by ranking instead of this sort.
pub(crate) const SMALL_BIN: u32 = 64;
/// Slots ranked together by the first stage: the first run length.
const CHUNK: u32 = 64;
/// Runs merged per stage.
const FAN_IN: u32 = 4;
/// Merge stages for bins of up to `CHUNK * FAN_IN^MAX_MERGE_STAGES` = 2^30
/// points; no storage binding holds more point indices. The stage uniforms
/// hold one more run length, the one after every merge, for the copy.
const MAX_MERGE_STAGES: u32 = 12;
const WORKGROUP_SIZE: u32 = 256;
/// Upper bound on dispatched workgroups; invocations loop over the slots.
/// 1024 workgroups of 256 invocations fill a large GPU once, and a smaller
/// grid keeps the flag-only dispatches of evenly spread points cheap.
const MAX_WORKGROUPS: u32 = 1024;
const U32_BYTES: u64 = 4;
/// Bytes of one stage uniform: the run length and padding.
const STAGE_BYTES: u64 = 16;

/// Sorts, in place, the point indices of every bin above [`SMALL_BIN`]
/// points into ascending order.
pub(crate) struct LargeBinSort {
    layout: wgpu::BindGroupLayout,
    chunk: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    /// Bind group 1 of merge stage `k`: runs of `CHUNK * FAN_IN^k` indices.
    /// The copy after `k` merges takes entry `k`.
    stages: Vec<wgpu::BindGroup>,
    copy: wgpu::ComputePipeline,
    scratch: Mutex<Option<wgpu::Buffer>>,
    /// The bin of every slot, found by the first stage.
    slot_bins: Mutex<Option<wgpu::Buffer>>,
    device: wgpu::Device,
}

impl LargeBinSort {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wgpu_nufft.bin_sort.layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
            ],
        });
        let stage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wgpu_nufft.bin_sort.stage_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(STAGE_BYTES),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wgpu_nufft.bin_sort.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let merge_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wgpu_nufft.bin_sort.merge_pipeline_layout"),
            bind_group_layouts: &[Some(&layout), Some(&stage_layout)],
            immediate_size: 0,
        });
        let pipeline = |label: &str, source: String, layout: &wgpu::PipelineLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let chunk = pipeline(
            "wgpu_nufft.bin_sort.chunk",
            generate_stage_wgsl(StageKind::Chunk),
            &pipeline_layout,
        );
        let merge = pipeline(
            "wgpu_nufft.bin_sort.merge",
            generate_stage_wgsl(StageKind::Merge),
            &merge_layout,
        );
        let copy = pipeline(
            "wgpu_nufft.bin_sort.copy",
            generate_stage_wgsl(StageKind::Copy),
            &merge_layout,
        );
        // One uniform per merge stage, each at an aligned offset.
        let stride =
            u64::from(device.limits().min_uniform_buffer_offset_alignment).max(STAGE_BYTES);
        let mut contents = vec![0u8; (stride * u64::from(MAX_MERGE_STAGES + 1)) as usize];
        for stage in 0..=MAX_MERGE_STAGES {
            let run = CHUNK * FAN_IN.pow(stage);
            let offset = (stride * u64::from(stage)) as usize;
            contents[offset..offset + 4].copy_from_slice(&run.to_le_bytes());
        }
        let runs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.bin_sort.runs"),
            contents: &contents,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let stages = (0..=MAX_MERGE_STAGES)
            .map(|stage| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wgpu_nufft.bin_sort.stage_bind_group"),
                    layout: &stage_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &runs,
                            offset: stride * u64::from(stage),
                            size: NonZeroU64::new(STAGE_BYTES),
                        }),
                    }],
                })
            })
            .collect();
        Self {
            layout,
            chunk,
            merge,
            stages,
            copy,
            scratch: Mutex::new(None),
            slot_bins: Mutex::new(None),
            device: device.clone(),
        }
    }

    /// Records the sort of the first `point_count` entries of
    /// `sorted_indices`. The first `bin_offset_bytes` of `bin_offsets` hold
    /// every bin's first slot and a terminal total. The last word of
    /// `large_bin_flag` is zero when no bin exceeds [`SMALL_BIN`] points, and
    /// otherwise at least half the size of the largest bin.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        bin_offsets: &wgpu::Buffer,
        bin_offset_bytes: u64,
        large_bin_flag: &wgpu::Buffer,
        sorted_indices: &wgpu::Buffer,
        point_count: usize,
    ) -> Result<()> {
        let stages = merge_stages(point_count);
        if stages == 0 {
            // No bin can exceed SMALL_BIN points.
            return Ok(());
        }
        let point_count_u32 =
            u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
                context: "large-bin sort point count",
            })?;
        let index_bytes = u64::from(point_count_u32) * U32_BYTES;
        let scratch = self.buffer(&self.scratch, "wgpu_nufft.bin_sort.scratch", index_bytes);
        let slot_bins = self.buffer(
            &self.slot_bins,
            "wgpu_nufft.bin_sort.slot_bins",
            index_bytes,
        );
        let bind_group = |source: &wgpu::Buffer, destination: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.bin_sort.bind_group"),
                layout: &self.layout,
                entries: &[
                    sized_binding(0, bin_offsets, bin_offset_bytes),
                    whole_binding(1, large_bin_flag),
                    sized_binding(2, source, index_bytes),
                    sized_binding(3, destination, index_bytes),
                    sized_binding(4, &slot_bins, index_bytes),
                ],
            })
        };
        let forward = bind_group(sorted_indices, &scratch);
        let backward = bind_group(&scratch, sorted_indices);
        let workgroups = (
            point_count_u32.div_ceil(WORKGROUP_SIZE).min(MAX_WORKGROUPS),
            1,
            1,
        );
        // Chunk ranking writes the scratch; the merges alternate from there,
        // each with its run length at bind group 1.
        recorder.dispatch(&self.chunk, &forward, workgroups);
        for (stage, stage_group) in self.stages.iter().take(stages as usize).enumerate() {
            let bind_group = if stage % 2 == 0 { &backward } else { &forward };
            recorder.dispatch_ranged(&self.merge, bind_group, stage_group, workgroups);
        }
        // The merges that ran leave the result in the scratch when they were
        // even in number; the copy works that out from the flag.
        recorder.dispatch_ranged(
            &self.copy,
            &backward,
            &self.stages[stages as usize],
            workgroups,
        );
        Ok(())
    }

    fn buffer(
        &self,
        cache: &Mutex<Option<wgpu::Buffer>>,
        label: &'static str,
        required_bytes: u64,
    ) -> wgpu::Buffer {
        let mut cached = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match cached.as_ref() {
            Some(buffer) if buffer.size() >= required_bytes => buffer.clone(),
            _ => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
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

/// Merge stages after chunk ranking that sort any bin of `point_count`
/// points: runs of `CHUNK * FAN_IN^k` after `k` merges. Zero when no bin can
/// exceed [`SMALL_BIN`].
fn merge_stages(point_count: usize) -> u32 {
    if point_count <= SMALL_BIN as usize {
        return 0;
    }
    let mut stages = 0;
    let mut sorted_run = u64::from(CHUNK);
    while sorted_run < point_count as u64 && stages < MAX_MERGE_STAGES {
        stages += 1;
        sorted_run *= u64::from(FAN_IN);
    }
    stages
}

fn whole_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn sized_binding(binding: u32, buffer: &wgpu::Buffer, bytes: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(bytes),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageKind {
    /// Finds every slot's bin and ranks the slots of each chunk.
    Chunk,
    /// Merges groups of `FAN_IN` runs of the uniform's run length.
    Merge,
    /// Copies the sorted slots back when an even number of merges ran.
    Copy,
}

/// A stage shader: every invocation walks slots with a grid stride, skips
/// the slots of small bins, and runs the stage's body for the rest, with the
/// slot's bin in `range`.
fn generate_stage_wgsl(kind: StageKind) -> String {
    let stage_uniform = "struct Stage {
    run: u32,
    unused_0: u32,
    unused_1: u32,
    unused_2: u32,
}

@group(1) @binding(0) var<uniform> stage: Stage;
";
    let (uniform, gate, locate, body) = match kind {
        StageKind::Chunk => (
            "",
            "",
            "        let bin = bin_of(slot);\n        slot_bins[slot] = bin;\n",
            CHUNK_BODY,
        ),
        StageKind::Merge => (
            stage_uniform,
            MERGE_GATE,
            "        let bin = slot_bins[slot];\n",
            MERGE_BODY,
        ),
        StageKind::Copy => (
            stage_uniform,
            COPY_GATE,
            "        let bin = slot_bins[slot];\n",
            COPY_BODY,
        ),
    };
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const SMALL_BIN: u32 = {SMALL_BIN}u;
const CHUNK: u32 = {CHUNK}u;
const FAN_IN: u32 = {FAN_IN}u;

@group(0) @binding(0) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> large_bin_flag: array<u32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read_write> destination: array<u32>;
@group(0) @binding(4) var<storage, read_write> slot_bins: array<u32>;
{uniform}
// The bin holding `slot`. Empty bins share their first slot with the next
// bin, so this is the last bin starting at or before it.
fn bin_of(slot: u32) -> u32 {{
    var low = 0u;
    var high = arrayLength(&bin_offsets) - 1u;
    loop {{
        if (high - low <= 1u) {{ break; }}
        let middle = low + (high - low) / 2u;
        if (bin_offsets[middle] <= slot) {{ low = middle; }} else {{ high = middle; }}
    }}
    return low;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    // Set by the binning passes only when some bin holds more than
    // SMALL_BIN points, to at least half the size of the largest bin.
    let flag = large_bin_flag[arrayLength(&large_bin_flag) - 1u];
    if (flag == 0u) {{ return; }}
{gate}    let total = arrayLength(&source);
    let slot_step = nwg.x * WORKGROUP_SIZE;
    for (var slot = gid.x; slot < total; slot = slot + slot_step) {{
{locate}        let range = vec2<u32>(bin_offsets[bin], bin_offsets[bin + 1u]);
        if (range.y - range.x <= SMALL_BIN) {{ continue; }}
{body}    }}
}}
"#,
    )
}

/// Ranks the slot's index inside its `CHUNK`-slot chunk of the bin.
const CHUNK_BODY: &str = r#"        let chunk_start = range.x + ((slot - range.x) / CHUNK) * CHUNK;
        let chunk_end = min(chunk_start + CHUNK, range.y);
        let value = source[slot];
        var rank = 0u;
        for (var other = chunk_start; other < chunk_end; other = other + 1u) {
            if (source[other] < value) { rank = rank + 1u; }
        }
        destination[chunk_start + rank] = value;
"#;

/// A merge stage whose runs already hold the largest bin has nothing to do.
const MERGE_GATE: &str = "    if (stage.run >= 2u * flag) { return; }\n";

/// The merges that ran are those whose runs stayed below the largest bin; an
/// odd number of them leaves the result in place. The copy's uniform holds
/// the run length after every recorded merge.
const COPY_GATE: &str = r#"    let limit = min(2u * flag, stage.run);
    var merges = 0u;
    var merged_run = CHUNK;
    loop {
        if (merged_run >= limit) { break; }
        merges = merges + 1u;
        merged_run = merged_run * FAN_IN;
    }
    if (merges % 2u == 1u) { return; }
"#;

/// Merges groups of `FAN_IN` sorted runs of `stage.run` indices. Runs stay
/// below 2^30 indices, so a group's span fits a `u32` whenever the bin
/// reaches past it.
const MERGE_BODY: &str = r#"        let run = stage.run;
        let offset = slot - range.x;
        let group_start = range.x + (offset / run / FAN_IN) * FAN_IN * run;
        let remaining = range.y - group_start;
        let group_end = group_start + select(FAN_IN * run, remaining, remaining / FAN_IN < run);
        let own_start = group_start + ((slot - group_start) / run) * run;
        let value = source[slot];
        var output_slot = group_start + (slot - own_start);
        for (var run_start = group_start; run_start < group_end; run_start = run_start + run) {
            if (run_start == own_start) { continue; }
            // Indices of this run below `value`: a lower bound in sorted order.
            var low = run_start;
            var high = min(run_start + run, group_end);
            loop {
                if (low >= high) { break; }
                let middle = low + (high - low) / 2u;
                if (source[middle] < value) { low = middle + 1u; } else { high = middle; }
            }
            output_slot = output_slot + (low - run_start);
        }
        destination[output_slot] = value;
"#;

const COPY_BODY: &str = "        destination[slot] = source[slot];\n";

/// Per-bin sort for binnings that leave small bins unordered: one invocation
/// per bin heap-sorts a bin of at most [`SMALL_BIN`] point indices and flags
/// a larger bin for [`LargeBinSort`] in the last word of `large_bin_flag`.
pub(crate) fn generate_small_bin_sort_wgsl(workgroup_size: u32) -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const SMALL_BIN: u32 = {SMALL_BIN}u;

@group(0) @binding(0) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> large_bin_flag: array<atomic<u32>>;

fn swap_indices(left: u32, right: u32) {{
    let temporary = sorted_indices[left];
    sorted_indices[left] = sorted_indices[right];
    sorted_indices[right] = temporary;
}}

fn sift_down(start: u32, count: u32, initial_root: u32) {{
    var root = initial_root;
    loop {{
        let child = root * 2u + 1u;
        if (child >= count) {{ break; }}
        var greatest = root;
        if (sorted_indices[start + greatest] < sorted_indices[start + child]) {{
            greatest = child;
        }}
        if (child + 1u < count &&
            sorted_indices[start + greatest] < sorted_indices[start + child + 1u]) {{
            greatest = child + 1u;
        }}
        if (greatest == root) {{ break; }}
        swap_indices(start + root, start + greatest);
        root = greatest;
    }}
}}

fn heap_sort(start: u32, end: u32) {{
    let count = end - start;
    if (count < 2u) {{ return; }}
    var root = count / 2u;
    loop {{
        if (root == 0u) {{ break; }}
        root = root - 1u;
        sift_down(start, count, root);
    }}
    var remaining = count;
    loop {{
        if (remaining <= 1u) {{ break; }}
        remaining = remaining - 1u;
        swap_indices(start, start + remaining);
        sift_down(start, remaining, 0u);
    }}
}}

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total_bins = arrayLength(&bin_offsets) - 1u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total_bins - 1u) / WORKGROUP_SIZE) {{ return; }}
    let bin = wg_flat * WORKGROUP_SIZE + lid.x;
    if (bin >= total_bins) {{ return; }}
    let start = bin_offsets[bin];
    let end = bin_offsets[bin + 1u];
    if (end - start <= SMALL_BIN) {{
        heap_sort(start, end);
    }} else {{
        atomicMax(&large_bin_flag[arrayLength(&large_bin_flag) - 1u], end - start);
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl_validation::assert_valid_wgsl;

    #[test]
    fn stage_shaders_validate() {
        for kind in [StageKind::Chunk, StageKind::Merge, StageKind::Copy] {
            assert_valid_wgsl(&generate_stage_wgsl(kind));
        }
        assert_valid_wgsl(&generate_small_bin_sort_wgsl(64));
    }

    #[test]
    fn merge_stages_cover_the_largest_possible_bin() {
        assert_eq!(merge_stages(0), 0);
        assert_eq!(merge_stages(64), 0);
        assert_eq!(merge_stages(65), 1);
        assert_eq!(merge_stages(256), 1);
        assert_eq!(merge_stages(257), 2);
        assert_eq!(merge_stages(1 << 18), 6);
        assert_eq!(merge_stages((1 << 18) + 1), 7);
        assert_eq!(merge_stages(1 << 30), MAX_MERGE_STAGES);
        assert_eq!(merge_stages(usize::MAX), MAX_MERGE_STAGES);
        // The last stage's groups still fit a u32 span.
        assert!(u64::from(CHUNK) * u64::from(FAN_IN).pow(MAX_MERGE_STAGES) <= 1 << 30);
    }

    /// Bin offsets and, per bin, its point indices in shuffled order.
    fn shuffled_bins(sizes: &[usize], seed: u32) -> (Vec<u32>, Vec<u32>) {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut offsets = vec![0u32];
        let mut indices = Vec::new();
        let mut base = 0u32;
        for &size in sizes {
            let mut bin: Vec<u32> = (base..base + size as u32).collect();
            for i in (1..bin.len()).rev() {
                let j = next() as usize % (i + 1);
                bin.swap(i, j);
            }
            indices.extend(bin);
            base += size as u32;
            offsets.push(indices.len() as u32);
        }
        (offsets, indices)
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_stages_sort_exactly_the_large_bins() {
        if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
            eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
            return;
        }
        use wgpu::util::DeviceExt;
        let context = pollster::block_on(wgpu_fft::device::request_default_device())
            .expect("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
        let device = &context.device;
        let sort = LargeBinSort::new(device);
        // Four merges plus the copy back; seven merges; one merge of four with
        // the rest skipped; two merges and the copy back. The flag-off run
        // must leave every bin untouched.
        let bins_of_300 = [300usize; 20];
        let bins_of_100 = [100usize; 60];
        let cases: [(&[usize], u32); 5] = [
            (&[3, 65, 0, 200, 5_000, 64, 17], 1),
            (&[300_000, 10, 70_000, 129], 1),
            (&bins_of_100, 1),
            (&bins_of_300, 1),
            (&[3, 5_000, 64], 0),
        ];
        for (sizes, on) in cases {
            // As the count pass sets it: the largest power-of-two rank.
            let largest = *sizes.iter().max().unwrap() as u32;
            let flag = if on == 1 {
                1u32 << (31 - (largest - 1).leading_zeros())
            } else {
                0
            };
            let (offsets, indices) = shuffled_bins(sizes, 0x2545_f491);
            let storage = |label: &str, words: &[u32], usage: wgpu::BufferUsages| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(words),
                    usage: wgpu::BufferUsages::STORAGE | usage,
                })
            };
            let bin_offsets = storage("offsets", &offsets, wgpu::BufferUsages::empty());
            // Only the last word of the flag buffer matters.
            let large_bin_flag = storage("flag", &[7, flag], wgpu::BufferUsages::empty());
            let sorted = storage("indices", &indices, wgpu::BufferUsages::COPY_SRC);
            let bytes = (indices.len() * 4) as u64;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut recorder = GpuRecorder::new(&mut encoder);
                sort.encode(
                    device,
                    &mut recorder,
                    &bin_offsets,
                    bin_offsets.size(),
                    &large_bin_flag,
                    &sorted,
                    indices.len(),
                )
                .unwrap();
            }
            encoder.copy_buffer_to_buffer(&sorted, 0, &readback, 0, bytes);
            context.queue.submit([encoder.finish()]);
            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let output: Vec<u32> =
                bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
            readback.unmap();
            for window in offsets.windows(2) {
                let (start, end) = (window[0] as usize, window[1] as usize);
                if on == 1 && end - start > SMALL_BIN as usize {
                    let expected: Vec<u32> = (start as u32..end as u32).collect();
                    assert_eq!(output[start..end], expected[..], "bin of {}", end - start);
                } else {
                    assert_eq!(
                        output[start..end],
                        indices[start..end],
                        "bin of {}",
                        end - start
                    );
                }
            }
        }
        // Mirror the other GPU tests, which leak their devices to avoid a
        // teardown stall on Windows.
        std::mem::forget(context);
    }

    /// Replays the stage arithmetic on the CPU for bins of many sizes.
    #[test]
    fn stage_arithmetic_sorts_every_large_bin() {
        let sizes = [0usize, 3, 64, 65, 200, 4096, 4097, 5000, 70_000];
        let (offsets, indices) = shuffled_bins(&sizes, 0x9e37_79b9);
        let total = indices.len();
        let bin_range = |slot: usize| {
            let bin = offsets.partition_point(|&start| start as usize <= slot) - 1;
            (offsets[bin] as usize, offsets[bin + 1] as usize)
        };
        let (chunk, fan) = (CHUNK as usize, FAN_IN as usize);
        let mut source = indices.clone();
        let mut destination = indices.clone();
        let mut run = 1usize;
        for stage in 0..=merge_stages(total) {
            for slot in 0..total {
                let (start, end) = bin_range(slot);
                if end - start <= SMALL_BIN as usize {
                    continue;
                }
                let value = source[slot];
                let output = if stage == 0 {
                    let chunk_start = start + (slot - start) / chunk * chunk;
                    let chunk_end = (chunk_start + chunk).min(end);
                    chunk_start
                        + source[chunk_start..chunk_end]
                            .iter()
                            .filter(|&&other| other < value)
                            .count()
                } else {
                    let span = run * fan;
                    let group_start = start + (slot - start) / span * span;
                    let group_end = (group_start + span).min(end);
                    let own_start = group_start + (slot - group_start) / run * run;
                    let mut output = group_start + (slot - own_start);
                    let mut run_start = group_start;
                    while run_start < group_end {
                        if run_start != own_start {
                            let run_end = (run_start + run).min(group_end);
                            output += source[run_start..run_end].partition_point(|&x| x < value);
                        }
                        run_start += run;
                    }
                    output
                };
                destination[output] = value;
            }
            std::mem::swap(&mut source, &mut destination);
            run = if stage == 0 { chunk } else { run * fan };
        }
        for window in offsets.windows(2) {
            let (start, end) = (window[0] as usize, window[1] as usize);
            if end - start > SMALL_BIN as usize {
                assert!(source[start..end].windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
}
