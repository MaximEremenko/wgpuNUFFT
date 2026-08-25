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
//! sort spreads every bin across all invocations. A first stage ranks every
//! point inside its 64-slot chunk of the bin, which leaves sorted runs of 64.
//! Each following stage merges groups of 64 runs: a point keeps its offset
//! inside its own run and adds, for every other run of its group, how many of
//! that run's indices are smaller (a binary search). Runs grow 64-fold per
//! stage, so four stages cover 2^24 points per bin.
//!
//! The stage count is fixed at encode time from the point count. The binning
//! passes set a flag, the last word of a buffer they share with the sort,
//! when any bin exceeds [`SMALL_BIN`] points; without it, every stage returns
//! at once, so evenly spread points pay only for a few empty dispatches.
//! [`generate_small_bin_sort_wgsl`] is such a pass for binnings that do not
//! rank small bins elsewhere.

use std::num::NonZeroU64;
use std::sync::Mutex;

use crate::error::{NufftError, Result};
use crate::gpu_recorder::GpuRecorder;

/// Largest bin that consumers order by ranking instead of this sort.
pub(crate) const SMALL_BIN: u32 = 64;
/// Runs merged per stage, and the first run length.
const FAN_IN: u32 = 64;
/// Merge stages for point counts up to `FAN_IN^(MAX_MERGE_STAGES + 1)`.
const MAX_MERGE_STAGES: u32 = 4;
const WORKGROUP_SIZE: u32 = 256;
/// Upper bound on dispatched workgroups; invocations loop over the slots.
/// 1024 workgroups of 256 invocations fill a large GPU once, and a smaller
/// grid keeps the flag-only dispatches of evenly spread points cheap.
const MAX_WORKGROUPS: u32 = 1024;
const U32_BYTES: u64 = 4;

/// Sorts, in place, the point indices of every bin above [`SMALL_BIN`]
/// points into ascending order.
pub(crate) struct LargeBinSort {
    layout: wgpu::BindGroupLayout,
    chunk: wgpu::ComputePipeline,
    /// Stage `k` merges runs of `FAN_IN^(k + 1)` indices.
    merges: Vec<wgpu::ComputePipeline>,
    copy: wgpu::ComputePipeline,
    scratch: Mutex<Option<wgpu::Buffer>>,
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
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wgpu_nufft.bin_sort.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label: &str, body: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(generate_stage_wgsl(body).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let chunk = pipeline("wgpu_nufft.bin_sort.chunk", &chunk_body());
        let merges = (0..MAX_MERGE_STAGES)
            .map(|stage| {
                pipeline(
                    "wgpu_nufft.bin_sort.merge",
                    &merge_body(FAN_IN.pow(stage + 1)),
                )
            })
            .collect();
        let copy = pipeline("wgpu_nufft.bin_sort.copy", COPY_BODY);
        Self {
            layout,
            chunk,
            merges,
            copy,
            scratch: Mutex::new(None),
            device: device.clone(),
        }
    }

    /// Records the sort of the first `point_count` entries of
    /// `sorted_indices`. The first `bin_offset_bytes` of `bin_offsets` hold
    /// every bin's first slot and a terminal total. The last word of
    /// `large_bin_flag` is nonzero when some bin exceeds [`SMALL_BIN`] points.
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
        let scratch = self.scratch(index_bytes);
        let bind_group = |source: &wgpu::Buffer, destination: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.bin_sort.bind_group"),
                layout: &self.layout,
                entries: &[
                    sized_binding(0, bin_offsets, bin_offset_bytes),
                    whole_binding(1, large_bin_flag),
                    sized_binding(2, source, index_bytes),
                    sized_binding(3, destination, index_bytes),
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
        // Chunk ranking writes the scratch; the merges alternate from there.
        recorder.dispatch(&self.chunk, &forward, workgroups);
        for (stage, pipeline) in self.merges.iter().take(stages as usize).enumerate() {
            let bind_group = if stage % 2 == 0 { &backward } else { &forward };
            recorder.dispatch(pipeline, bind_group, workgroups);
        }
        if stages.is_multiple_of(2) {
            // An even number of merges leaves the result in the scratch.
            recorder.dispatch(&self.copy, &backward, workgroups);
        }
        Ok(())
    }

    fn scratch(&self, required_bytes: u64) -> wgpu::Buffer {
        let mut cached = self
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match cached.as_ref() {
            Some(buffer) if buffer.size() >= required_bytes => buffer.clone(),
            _ => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wgpu_nufft.bin_sort.scratch"),
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
/// points: runs of `FAN_IN^(k + 1)` after `k` merges. Zero when no bin can
/// exceed [`SMALL_BIN`].
fn merge_stages(point_count: usize) -> u32 {
    if point_count <= SMALL_BIN as usize {
        return 0;
    }
    let mut stages = 1;
    let mut sorted_run = u64::from(FAN_IN).pow(2);
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

/// A stage shader: every invocation walks slots with a grid stride, skips
/// the slots of small bins, and runs `body` for the rest, with the slot's
/// bin in `range`.
fn generate_stage_wgsl(body: &str) -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const SMALL_BIN: u32 = {SMALL_BIN}u;
const FAN_IN: u32 = {FAN_IN}u;

@group(0) @binding(0) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> large_bin_flag: array<u32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read_write> destination: array<u32>;

// Slots [start, end) of the bin holding `slot`. Empty bins share their first
// slot with the next bin, so this is the last bin starting at or before it.
fn bin_range(slot: u32) -> vec2<u32> {{
    var low = 0u;
    var high = arrayLength(&bin_offsets) - 1u;
    loop {{
        if (high - low <= 1u) {{ break; }}
        let middle = low + (high - low) / 2u;
        if (bin_offsets[middle] <= slot) {{ low = middle; }} else {{ high = middle; }}
    }}
    return vec2<u32>(bin_offsets[low], bin_offsets[low + 1u]);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    // Set by the binning passes only when some bin holds more than
    // SMALL_BIN points.
    if (large_bin_flag[arrayLength(&large_bin_flag) - 1u] == 0u) {{ return; }}
    let total = arrayLength(&source);
    let slot_step = nwg.x * WORKGROUP_SIZE;
    for (var slot = gid.x; slot < total; slot = slot + slot_step) {{
        let range = bin_range(slot);
        if (range.y - range.x <= SMALL_BIN) {{ continue; }}
{body}    }}
}}
"#,
    )
}

/// Ranks the slot's index inside its 64-slot chunk of the bin.
fn chunk_body() -> String {
    r#"        let chunk_start = range.x + ((slot - range.x) / FAN_IN) * FAN_IN;
        let chunk_end = min(chunk_start + FAN_IN, range.y);
        let value = source[slot];
        var rank = 0u;
        for (var other = chunk_start; other < chunk_end; other = other + 1u) {
            if (source[other] < value) { rank = rank + 1u; }
        }
        destination[chunk_start + rank] = value;
"#
    .to_string()
}

/// Merges groups of `FAN_IN` sorted runs of `run` indices.
fn merge_body(run: u32) -> String {
    format!(
        r#"        let group_span = {run}u * FAN_IN;
        let group_start = range.x + ((slot - range.x) / group_span) * group_span;
        let group_end = min(group_start + group_span, range.y);
        let own_start = group_start + ((slot - group_start) / {run}u) * {run}u;
        let value = source[slot];
        var output_slot = group_start + (slot - own_start);
        for (var run_start = group_start; run_start < group_end; run_start = run_start + {run}u) {{
            if (run_start == own_start) {{ continue; }}
            // Indices of this run below `value`: a lower bound in sorted order.
            var low = run_start;
            var high = min(run_start + {run}u, group_end);
            loop {{
                if (low >= high) {{ break; }}
                let middle = low + (high - low) / 2u;
                if (source[middle] < value) {{ low = middle + 1u; }} else {{ high = middle; }}
            }}
            output_slot = output_slot + (low - run_start);
        }}
        destination[output_slot] = value;
"#
    )
}

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
        atomicMax(&large_bin_flag[arrayLength(&large_bin_flag) - 1u], 1u);
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
        assert_valid_wgsl(&generate_stage_wgsl(&chunk_body()));
        for stage in 0..MAX_MERGE_STAGES {
            assert_valid_wgsl(&generate_stage_wgsl(&merge_body(FAN_IN.pow(stage + 1))));
        }
        assert_valid_wgsl(&generate_stage_wgsl(COPY_BODY));
        assert_valid_wgsl(&generate_small_bin_sort_wgsl(64));
    }

    #[test]
    fn merge_stages_cover_the_largest_possible_bin() {
        assert_eq!(merge_stages(0), 0);
        assert_eq!(merge_stages(64), 0);
        assert_eq!(merge_stages(65), 1);
        assert_eq!(merge_stages(4096), 1);
        assert_eq!(merge_stages(4097), 2);
        assert_eq!(merge_stages(1 << 18), 2);
        assert_eq!(merge_stages((1 << 18) + 1), 3);
        assert_eq!(merge_stages(1 << 24), 3);
        assert_eq!(merge_stages((1 << 24) + 1), 4);
        assert_eq!(merge_stages(usize::MAX), MAX_MERGE_STAGES);
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
        // Two merges plus the copy back, then three merges; the flag-off run
        // must leave every bin untouched.
        let cases: [(&[usize], u32); 3] = [
            (&[3, 65, 0, 200, 5_000, 64, 17], 1),
            (&[300_000, 10, 70_000, 129], 1),
            (&[3, 5_000, 64], 0),
        ];
        for (sizes, flag) in cases {
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
                if flag == 1 && end - start > SMALL_BIN as usize {
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
        let fan = FAN_IN as usize;
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
                    let chunk_start = start + (slot - start) / fan * fan;
                    let chunk_end = (chunk_start + fan).min(end);
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
            run = if stage == 0 { fan } else { run * fan };
        }
        for window in offsets.windows(2) {
            let (start, end) = (window[0] as usize, window[1] as usize);
            if end - start > SMALL_BIN as usize {
                assert!(source[start..end].windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
}
