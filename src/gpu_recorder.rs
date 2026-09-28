//! Command recording that batches consecutive dispatches into one compute
//! pass.

/// Records one NUFFT execution, batching consecutive dispatches, FFTs
/// included, into a single compute pass.
///
/// Every separate compute pass adds about 4 µs of
/// latency, because wgpu records each pass and its barriers into command
/// buffers of their own, against under 1 µs per dispatch inside a shared
/// pass. wgpu still orders dependent dispatches within one pass, so every
/// dispatch sees the previous dispatches' writes. The pass is wgpu-fft's
/// shared pass, so FFT executions join it too; clears, copies and profiled
/// dispatches end it.
pub(crate) struct GpuRecorder<'a> {
    inner: wgpu_fft::FftRecorder<'a>,
}

impl<'a> GpuRecorder<'a> {
    pub(crate) fn new(encoder: &'a mut wgpu::CommandEncoder) -> Self {
        Self {
            inner: wgpu_fft::FftRecorder::new(encoder),
        }
    }

    /// The wgpu-fft recorder, so FFT executions join the shared pass. They
    /// leave their own pipeline and bind group set, which is harmless here
    /// because every dispatch sets both.
    pub(crate) fn fft(&mut self) -> &mut wgpu_fft::FftRecorder<'a> {
        &mut self.inner
    }

    /// Returns the command encoder for clears, copies and queries, ending
    /// any open compute pass.
    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.inner.encoder()
    }

    pub(crate) fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        self.encoder().clear_buffer(buffer, offset, size);
    }

    /// Records one dispatch into the shared compute pass, opening it on first
    /// use.
    pub(crate) fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let pass = self.inner.compute_pass();
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    /// Records one dispatch. With profiling, a dispatch that writes
    /// timestamps gets a compute pass of its own named `label`, so the
    /// timestamps bracket exactly it; every other dispatch joins the shared
    /// pass.
    pub(crate) fn dispatch_profiled(
        &mut self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
        #[cfg(feature = "gpu-profiling")] timestamp_writes: Option<
            wgpu::ComputePassTimestampWrites<'_>,
        >,
    ) {
        #[cfg(feature = "gpu-profiling")]
        if timestamp_writes.is_some() {
            let mut pass = self
                .encoder()
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            return;
        }
        let _ = label;
        self.dispatch(pipeline, bind_group, workgroups);
    }
}
