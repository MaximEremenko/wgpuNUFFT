use wgpu::util::DeviceExt;

use crate::error::{NufftError, Result};
use crate::gpu_recorder::GpuRecorder;

/// Declares the first flat workgroup of a ranged dispatch, at bind group 1.
/// Shaders add `dispatch_range.first` to their flat workgroup index.
pub(crate) const DISPATCH_RANGE_WGSL: &str = "
struct DispatchRange {
    first: u32,
    unused_0: u32,
    unused_1: u32,
    unused_2: u32,
}

@group(1) @binding(0) var<uniform> dispatch_range: DispatchRange;
";

/// Bytes of one [`DISPATCH_RANGE_WGSL`] uniform.
const RANGE_BYTES: u64 = 16;

/// Splits one logical dispatch into dispatches of bounded work.
///
/// A long dispatch can hold the GPU past the operating system's watchdog on
/// devices that preempt only between dispatches, so large rank-generic
/// workloads are recorded as consecutive ranges of their flat workgroups.
/// Every range reads its first workgroup from a uniform at bind group 1;
/// results do not depend on the split, since each workgroup computes the
/// same thing wherever its range starts.
pub(crate) struct RangedDispatch {
    /// The uniform of a single range starting at workgroup zero.
    zero: wgpu::Buffer,
    stride: u64,
    max_workgroups_per_dimension: u32,
}

impl RangedDispatch {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            zero: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu_nufft.dispatch_range.zero"),
                contents: &[0u8; RANGE_BYTES as usize],
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            stride: u64::from(limits.min_uniform_buffer_offset_alignment).max(RANGE_BYTES),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Records `total` flat workgroups of `pipeline` in ranges of at most
    /// `per_dispatch` workgroups.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        recorder: &mut GpuRecorder<'_>,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        total: u64,
        per_dispatch: u64,
    ) -> Result<()> {
        if total == 0 {
            return Ok(());
        }
        let per_dispatch = per_dispatch.clamp(1, u64::from(u32::MAX));
        let layout = pipeline.get_bind_group_layout(1);
        let ranges = total.div_ceil(per_dispatch);
        let overflow = || NufftError::LengthOverflow {
            context: "ranged dispatch workgroup count",
        };
        if ranges == 1 {
            let range_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.dispatch_range.bind_group"),
                layout: &layout,
                entries: &[range_entry(&self.zero, 0)],
            });
            let workgroups = u32::try_from(total).map_err(|_| overflow())?;
            recorder.dispatch_ranged(
                pipeline,
                bind_group,
                &range_group,
                split_workgroups(workgroups, self.max_workgroups_per_dimension)?,
            );
            return Ok(());
        }
        let stride = usize::try_from(self.stride).map_err(|_| overflow())?;
        let range_count = usize::try_from(ranges).map_err(|_| overflow())?;
        let mut contents = vec![0u8; range_count * stride];
        for range in 0..range_count {
            let first = u32::try_from(range as u64 * per_dispatch).map_err(|_| overflow())?;
            contents[range * stride..range * stride + 4].copy_from_slice(&first.to_le_bytes());
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.dispatch_range.ranges"),
            contents: &contents,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        for range in 0..range_count {
            let first = range as u64 * per_dispatch;
            let count = per_dispatch.min(total - first);
            let range_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.dispatch_range.bind_group"),
                layout: &layout,
                entries: &[range_entry(&buffer, range as u64 * self.stride)],
            });
            let workgroups = u32::try_from(count).map_err(|_| overflow())?;
            recorder.dispatch_ranged(
                pipeline,
                bind_group,
                &range_group,
                split_workgroups(workgroups, self.max_workgroups_per_dimension)?,
            );
        }
        Ok(())
    }
}

fn range_entry(buffer: &wgpu::Buffer, offset: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: std::num::NonZeroU64::new(RANGE_BYTES),
        }),
    }
}

/// Splits a flat workgroup count into a u32-safe 3D dispatch grid.
///
/// This is deliberately local rather than reaching through wgpu-fft's
/// crate-private runtime module or promoting raw WGSL string fragments into
/// its public API. NUFFT needs its own structured error, and barrier-using
/// kernels require a workgroup-uniform padding guard instead of the FFT
/// helper's lane-level element guard. Reconsider promotion when another
/// non-FFT consumer needs a stable typed dispatch abstraction.
pub(crate) fn split_workgroups(workgroups: u32, max_per_dimension: u32) -> Result<(u32, u32, u32)> {
    if max_per_dimension == 0 {
        return Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension,
        });
    }
    if workgroups == 0 {
        return Ok((0, 1, 1));
    }

    let total = u64::from(workgroups);
    let maximum = u64::from(max_per_dimension);
    let per_slice_limit = maximum * maximum;
    let z = total.div_ceil(per_slice_limit);
    if z > maximum {
        return Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension,
        });
    }

    let workgroups_per_slice = total.div_ceil(z);
    let y = workgroups_per_slice.div_ceil(maximum);
    let x = workgroups_per_slice.div_ceil(y);
    let covered = x
        .checked_mul(y)
        .and_then(|xy| xy.checked_mul(z))
        .unwrap_or(u64::MAX);
    if x <= maximum && y <= maximum && covered <= u64::from(u32::MAX) + 1 {
        return Ok((x as u32, y as u32, z as u32));
    }

    // Exactly covers the full u32 index space while keeping flattened WGSL
    // workgroup arithmetic in range.
    if max_per_dimension >= 2048 {
        return Ok((2048, 2048, 1024));
    }

    Err(NufftError::GpuDispatchUnsupported {
        workgroups,
        max_per_dimension,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_grid_covers_scan_and_full_u32_boundaries() {
        for workgroups in [0, 1, 65_535, 65_536, 65_537, 1 << 19, u32::MAX] {
            let (x, y, z) = split_workgroups(workgroups, 65_535).unwrap();
            assert!(x <= 65_535 && y <= 65_535 && z <= 65_535);
            let covered = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(covered >= u64::from(workgroups));
            assert!(covered <= u64::from(u32::MAX) + 1);
        }
        assert_eq!(
            split_workgroups(u32::MAX, 65_535).unwrap(),
            (2048, 2048, 1024)
        );
    }

    #[test]
    fn split_grid_rejects_zero_or_insufficient_dimension_limits() {
        assert!(matches!(
            split_workgroups(1, 0),
            Err(NufftError::GpuDispatchUnsupported { .. })
        ));
        assert!(matches!(
            split_workgroups(2, 1),
            Err(NufftError::GpuDispatchUnsupported { .. })
        ));
    }

    #[test]
    fn future_three_dimensional_scan_uses_split_dispatch() {
        let bins = 1u32 << 27;
        let block_count = bins.div_ceil(2048);
        assert_eq!(block_count, 65_536);
        let grid = split_workgroups(block_count, 65_535).unwrap();
        assert_ne!(grid, (block_count, 1, 1));
        assert!(grid.0 <= 65_535 && grid.1 <= 65_535 && grid.2 <= 65_535);
        assert!(
            u64::from(grid.0) * u64::from(grid.1) * u64::from(grid.2) >= u64::from(block_count)
        );
    }
}
