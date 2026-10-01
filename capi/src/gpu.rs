//! The process-wide GPU device and host transfers.

use std::panic::AssertUnwindSafe;
use std::sync::{mpsc, Arc, Mutex, PoisonError};

use wgpu_nufft::{wgpu, DoubleFloat, FftPrecision};

use crate::error::{Error, Result};

pub(crate) struct GpuContext {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// The first wgpu error raised outside an error scope, which the next
    /// guarded operation reports instead of wgpu's default panic.
    uncaptured_error: Arc<Mutex<Option<String>>>,
    pub(crate) name: String,
}

impl GpuContext {
    pub(crate) fn supports_f64(&self) -> bool {
        self.device.features().contains(wgpu::Features::SHADER_F64)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // Plan creation fills wgpu-fft's thread-local pipeline cache; release
        // its device-owned entries before the device goes.
        wgpu_nufft::wgpu_fft::clear_thread_local_pipeline_cache(&self.device);
    }
}

/// The shared device, or why there is none. A failed request is remembered,
/// so plans that fall back to the CPU do not ask for an adapter every time.
static CONTEXT: Mutex<Option<std::result::Result<Arc<GpuContext>, String>>> = Mutex::new(None);

pub(crate) fn context() -> Result<Arc<GpuContext>> {
    let mut slot = CONTEXT.lock().unwrap_or_else(PoisonError::into_inner);
    if slot.is_none() {
        *slot = Some(create_context().map(Arc::new));
    }
    match slot.as_ref() {
        Some(Ok(context)) => Ok(Arc::clone(context)),
        Some(Err(reason)) => Err(Error::GpuUnavailable(reason.clone())),
        None => Err(Error::Internal("GPU context slot is empty".to_owned())),
    }
}

/// Forgets the shared device; plans that hold it keep it alive.
pub(crate) fn release_context() {
    CONTEXT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
}

fn create_context() -> std::result::Result<GpuContext, String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(descriptor.with_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
        compatible_surface: None,
    }))
    .map_err(|error| format!("no GPU adapter: {error}"))?;
    let required_features = adapter.features() & wgpu::Features::SHADER_F64;
    let descriptor = |limits: wgpu::Limits| wgpu::DeviceDescriptor {
        label: Some("wgpu_nufft.c.device"),
        required_features,
        required_limits: limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) =
        match pollster::block_on(adapter.request_device(&descriptor(adapter.limits()))) {
            Ok(pair) => pair,
            Err(maximum_error) => pollster::block_on(
                adapter.request_device(&descriptor(wgpu::Limits::default())),
            )
            .map_err(|default_error| {
                format!(
                    "GPU device request failed at the adapter's limits ({maximum_error}) and at the defaults ({default_error})"
                )
            })?,
        };
    // wgpu's default handler panics on errors outside an error scope.
    let uncaptured_error = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&uncaptured_error);
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        sink.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(|| error.to_string());
    }));
    let info = adapter.get_info();
    Ok(GpuContext {
        _instance: instance,
        _adapter: adapter,
        device,
        queue,
        uncaptured_error,
        name: format!("{} ({:?})", info.name, info.backend),
    })
}

/// Runs GPU work so that wgpu errors and panics come back as errors.
pub(crate) fn guarded<T>(context: &GpuContext, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let device = &context.device;
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let result = std::panic::catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        Err(Error::Gpu(format!(
            "GPU operation panicked: {}",
            panic_message(&payload)
        )))
    });
    // Scopes pop in reverse order of their creation.
    let scoped = [
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
        pollster::block_on(out_of_memory.pop()),
    ];
    let uncaptured = context
        .uncaptured_error
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(error) = scoped.into_iter().flatten().next() {
        return Err(Error::Gpu(error.to_string()));
    }
    if let Some(error) = uncaptured {
        return Err(Error::Gpu(error));
    }
    result
}

pub(crate) fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
        })
        .unwrap_or_else(|| "unknown panic".to_owned())
}

/// A device buffer reused while it is large enough.
#[derive(Default)]
pub(crate) struct BufferSlot {
    buffer: Option<wgpu::Buffer>,
}

impl BufferSlot {
    fn ensure(
        &mut self,
        context: &GpuContext,
        bytes: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Result<wgpu::Buffer> {
        let bytes = bytes.max(4);
        let limits = context.device.limits();
        let storage = usage.contains(wgpu::BufferUsages::STORAGE);
        if bytes > limits.max_buffer_size
            || (storage && bytes > limits.max_storage_buffer_binding_size)
        {
            return Err(Error::Plan(format!(
                "{label} needs {bytes} bytes, beyond the GPU's buffer limits"
            )));
        }
        match &self.buffer {
            Some(buffer) if buffer.size() >= bytes => Ok(buffer.clone()),
            _ => {
                let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage,
                    mapped_at_creation: false,
                });
                self.buffer = Some(buffer.clone());
                Ok(buffer)
            }
        }
    }

    /// The buffer of the last upload.
    pub(crate) fn current(&self) -> Result<wgpu::Buffer> {
        self.buffer
            .clone()
            .ok_or_else(|| Error::Internal("no buffer uploaded yet".to_owned()))
    }

    /// Uploads `words`, which must take exactly `expected` bytes.
    pub(crate) fn upload(
        &mut self,
        context: &GpuContext,
        words: &[u8],
        expected: u64,
        label: &'static str,
    ) -> Result<wgpu::Buffer> {
        if words.len() as u64 != expected {
            return Err(Error::Internal(format!(
                "{label}: the plan expects {expected} bytes, the host prepared {}",
                words.len()
            )));
        }
        let buffer = self.ensure(
            context,
            expected,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label,
        )?;
        if !words.is_empty() {
            context.queue.write_buffer(&buffer, 0, words);
        }
        Ok(buffer)
    }
}

/// An output buffer and its readback twin.
#[derive(Default)]
pub(crate) struct OutputSlots {
    output: BufferSlot,
    readback: BufferSlot,
}

impl OutputSlots {
    pub(crate) fn output(&mut self, context: &GpuContext, bytes: u64) -> Result<wgpu::Buffer> {
        self.output.ensure(
            context,
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "wgpu_nufft.c.output",
        )
    }

    /// Submits `encoder` after copying `bytes` of `output` for readback, and
    /// returns them once the GPU has finished.
    pub(crate) fn submit_and_read(
        &mut self,
        context: &GpuContext,
        mut encoder: wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        bytes: u64,
    ) -> Result<Vec<u8>> {
        let readback = self.readback.ensure(
            context,
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            "wgpu_nufft.c.readback",
        )?;
        if bytes > 0 {
            encoder.copy_buffer_to_buffer(output, 0, &readback, 0, bytes);
        }
        let submission = context.queue.submit([encoder.finish()]);
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let slice = readback.slice(0..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| Error::Gpu(format!("waiting for the GPU failed: {error}")))?;
        receiver
            .recv()
            .map_err(|_| Error::Gpu("the GPU readback was dropped".to_owned()))?
            .map_err(|error| Error::Gpu(format!("mapping the GPU result failed: {error}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| Error::Gpu(format!("reading the GPU result failed: {error}")))?;
        let words = mapped.to_vec();
        drop(mapped);
        readback.unmap();
        Ok(words)
    }
}

/// Words of `values` in a plan's precision: `f32`, `f64`, or a `Df64`
/// `(hi, lo)` pair of `f32` per value. Points and interleaved complex values
/// convert alike.
pub(crate) fn device_words(values: &[f64], precision: FftPrecision) -> Vec<u8> {
    match precision {
        FftPrecision::F32 => {
            let words: Vec<f32> = values.iter().map(|&value| value as f32).collect();
            bytemuck::cast_slice(&words).to_vec()
        }
        FftPrecision::F64 => bytemuck::cast_slice(values).to_vec(),
        FftPrecision::Df64 => {
            let words: Vec<f32> = values
                .iter()
                .flat_map(|&value| {
                    let pair = DoubleFloat::from_f64(value);
                    [pair.hi, pair.lo]
                })
                .collect();
            bytemuck::cast_slice(&words).to_vec()
        }
    }
}

/// Values of device `words` in a plan's precision; see [`device_words`].
pub(crate) fn host_values(words: &[u8], precision: FftPrecision) -> Vec<f64> {
    match precision {
        FftPrecision::F32 => words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f64::from(f32::from_le_bytes(word)))
            .collect(),
        FftPrecision::F64 => words
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&word| f64::from_le_bytes(word))
            .collect(),
        FftPrecision::Df64 => words
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| {
                let hi = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                let lo = f32::from_le_bytes([word[4], word[5], word[6], word[7]]);
                DoubleFloat::new(hi, lo).to_f64()
            })
            .collect(),
    }
}
