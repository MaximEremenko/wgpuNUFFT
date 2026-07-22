//! Native NumPy bindings for `wgpu-nufft`.

use std::f64::consts::PI;
use std::sync::{mpsc, Arc, Mutex};

use numpy::{
    ndarray::{ArrayD, IxDyn},
    Complex32, Element, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use wgpu_nufft::{
    ModeOrder, NufftConfig, NufftInterval, NufftPlan, NufftType3Config, NufftType3Plan,
};

create_exception!(wgpu_nufft, WgpuNufftError, PyRuntimeError);

struct ContextInner {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    backend: String,
    device_type: String,
    driver: String,
    driver_info: String,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // Plans populate wgpu-fft's thread-local pipeline cache. Release those
        // device-owned entries before the device itself is destroyed; leaving
        // them to process teardown can crash native graphics drivers.
        wgpu_fft::clear_thread_local_pipeline_cache(&self.device);
    }
}

/// Clears wgpu-fft's cache on the exact thread that populated it.
///
/// Python extension objects may be destroyed on a different thread, so
/// relying only on `ContextInner::drop` can leave thread-local driver handles
/// alive until process teardown.
struct PipelineCacheClearGuard<'a>(&'a wgpu::Device);

impl Drop for PipelineCacheClearGuard<'_> {
    fn drop(&mut self) {
        wgpu_fft::clear_thread_local_pipeline_cache(self.0);
    }
}

/// Persistent native wgpu device shared by reusable NUFFT plans.
#[pyclass(name = "Context", module = "wgpu_nufft._native", frozen)]
struct PyContext {
    inner: Arc<ContextInner>,
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (power_preference="high-performance", force_fallback=false))]
    fn new(py: Python<'_>, power_preference: &str, force_fallback: bool) -> PyResult<Self> {
        let preference = parse_power_preference(power_preference)?;
        let inner = py
            .detach(move || initialize_gpu(preference, force_fallback))
            .map_err(WgpuNufftError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    fn adapter_name(&self) -> String {
        self.inner.adapter_name.clone()
    }

    #[getter]
    fn backend(&self) -> String {
        self.inner.backend.clone()
    }

    #[getter]
    fn device_type(&self) -> String {
        self.inner.device_type.clone()
    }

    #[getter]
    fn driver(&self) -> String {
        self.inner.driver.clone()
    }

    #[getter]
    fn driver_info(&self) -> String {
        self.inner.driver_info.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Context(adapter_name={:?}, backend={:?}, device_type={:?})",
            self.inner.adapter_name, self.inner.backend, self.inner.device_type
        )
    }
}

fn parse_power_preference(value: &str) -> PyResult<wgpu::PowerPreference> {
    match value {
        "high-performance" | "high" => Ok(wgpu::PowerPreference::HighPerformance),
        "low-power" | "low" => Ok(wgpu::PowerPreference::LowPower),
        "none" | "default" => Ok(wgpu::PowerPreference::None),
        _ => Err(PyValueError::new_err(
            "power_preference must be 'high-performance', 'low-power', or 'none'",
        )),
    }
}

fn initialize_gpu(
    power_preference: wgpu::PowerPreference,
    force_fallback: bool,
) -> Result<ContextInner, String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(descriptor.with_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        force_fallback_adapter: force_fallback,
        apply_limit_buckets: false,
        compatible_surface: None,
    }))
    .map_err(|error| format!("WebGPU adapter request failed: {error}"))?;

    let required_features = adapter.features() & wgpu::Features::SHADER_F64;
    let maximum_descriptor = wgpu::DeviceDescriptor {
        label: Some("wgpu_nufft.python.device"),
        required_features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = match pollster::block_on(adapter.request_device(&maximum_descriptor)) {
        Ok(pair) => pair,
        Err(maximum_error) => {
            let default_descriptor = wgpu::DeviceDescriptor {
                label: Some("wgpu_nufft.python.default_device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            };
            pollster::block_on(adapter.request_device(&default_descriptor)).map_err(
                |default_error| {
                    format!(
                        "WebGPU device request failed at adapter limits ({maximum_error}) and defaults ({default_error})"
                    )
                },
            )?
        }
    };

    let info = adapter.get_info();
    Ok(ContextInner {
        _instance: instance,
        _adapter: adapter,
        device,
        queue,
        adapter_name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: info.driver,
        driver_info: info.driver_info,
    })
}

#[derive(Default)]
struct BufferSlot {
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
}

impl BufferSlot {
    fn ensure(
        &mut self,
        context: &ContextInner,
        requested: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
        storage: bool,
    ) -> Result<wgpu::Buffer, String> {
        let allocated = requested.max(4);
        let limits = context.device.limits();
        if allocated > limits.max_buffer_size {
            return Err(format!(
                "{label} requires {allocated} bytes, exceeding max_buffer_size {}",
                limits.max_buffer_size
            ));
        }
        if storage && allocated > limits.max_storage_buffer_binding_size {
            return Err(format!(
                "{label} requires {allocated} bytes, exceeding max_storage_buffer_binding_size {}",
                limits.max_storage_buffer_binding_size
            ));
        }
        if self.buffer.is_none() || self.capacity < allocated {
            self.buffer = Some(context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: allocated,
                usage,
                mapped_at_creation: false,
            }));
            self.capacity = allocated;
        }
        self.buffer
            .clone()
            .ok_or_else(|| format!("internal error while allocating {label}"))
    }
}

#[derive(Default)]
struct ExecutionBuffers {
    points: BufferSlot,
    values: BufferSlot,
    targets: BufferSlot,
    output: BufferSlot,
    readback: BufferSlot,
}

fn upload_words(
    context: &ContextInner,
    slot: &mut BufferSlot,
    values: &[f32],
    expected_bytes: u64,
    label: &'static str,
) -> Result<wgpu::Buffer, String> {
    let bytes = bytemuck::cast_slice(values);
    let actual = u64::try_from(bytes.len()).map_err(|_| format!("{label} byte length overflow"))?;
    if actual != expected_bytes {
        return Err(format!(
            "internal {label} size mismatch: core requires {expected_bytes} bytes, host prepared {actual}"
        ));
    }
    let buffer = slot.ensure(
        context,
        expected_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        label,
        true,
    )?;
    if !bytes.is_empty() {
        context.queue.write_buffer(&buffer, 0, bytes);
    }
    Ok(buffer)
}

fn output_buffers(
    context: &ContextInner,
    buffers: &mut ExecutionBuffers,
    output_bytes: u64,
) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
    let output = buffers.output.ensure(
        context,
        output_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "wgpu_nufft.python.output",
        true,
    )?;
    let readback = buffers.readback.ensure(
        context,
        output_bytes,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        "wgpu_nufft.python.readback",
        false,
    )?;
    Ok((output, readback))
}

fn submit_and_read(
    context: &ContextInner,
    mut encoder: wgpu::CommandEncoder,
    output: &wgpu::Buffer,
    readback: &wgpu::Buffer,
    output_bytes: u64,
) -> Result<Vec<Complex32>, String> {
    if output_bytes == 0 {
        return Ok(Vec::new());
    }
    if !output_bytes.is_multiple_of(8) {
        return Err(format!(
            "internal complex output byte length {output_bytes} is not divisible by 8"
        ));
    }
    encoder.copy_buffer_to_buffer(output, 0, readback, 0, output_bytes);
    let submission = context.queue.submit([encoder.finish()]);
    let slice = readback.slice(0..output_bytes);
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
        .map_err(|error| format!("GPU polling failed: {error}"))?;
    receiver
        .recv()
        .map_err(|_| "GPU readback callback was dropped".to_owned())?
        .map_err(|error| format!("GPU readback mapping failed: {error}"))?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|error| format!("GPU readback range failed: {error}"))?;
    let words = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    Ok(words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Complex32::new(pair[0], pair[1]))
        .collect())
}

fn copy_c_array<T>(name: &str, array: PyReadonlyArrayDyn<'_, T>) -> PyResult<(Vec<usize>, Vec<T>)>
where
    T: Element + Copy,
{
    let shape = array.shape().to_vec();
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous; use numpy.ascontiguousarray"
        )));
    }
    let values = array
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "{name} must be an aligned C-contiguous NumPy array"
            ))
        })?
        .to_vec();
    Ok((shape, values))
}

fn complex_words(values: &[Complex32]) -> Vec<f32> {
    let mut words = Vec::with_capacity(values.len().saturating_mul(2));
    for value in values {
        words.push(value.re);
        words.push(value.im);
    }
    words
}

fn validate_complex(name: &str, values: &[Complex32]) -> PyResult<()> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "{name} contains a non-finite value at flat index {index}"
        )));
    }
    Ok(())
}

fn validate_type12_points(
    name: &str,
    shape: &[usize],
    values: &[f32],
    dimensions: usize,
) -> PyResult<usize> {
    let point_count = match shape {
        [count] if dimensions == 1 => *count,
        [count, width] if *width == dimensions => *count,
        _ => {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {dimensions}){}",
                if dimensions == 1 { " or (M,)" } else { "" }
            )))
        }
    };
    let limit = 3.0 * PI;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains a non-finite coordinate at flat index {index}"
            )));
        }
        if f64::from(value).abs() > limit {
            return Err(PyValueError::new_err(format!(
                "{name} coordinate at flat index {index} is outside [-3*pi, 3*pi]"
            )));
        }
    }
    Ok(point_count)
}

fn validate_type3_points(
    name: &str,
    shape: &[usize],
    values: &[f32],
    bounds: &[NufftInterval],
) -> PyResult<usize> {
    let dimensions = bounds.len();
    let point_count = match shape {
        [count] if dimensions == 1 => *count,
        [count, width] if *width == dimensions => *count,
        _ => {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {dimensions}){}",
                if dimensions == 1 { " or (M,)" } else { "" }
            )))
        }
    };
    for (index, point) in values.chunks_exact(dimensions).enumerate() {
        for (axis, (&value, &interval)) in point.iter().zip(bounds).enumerate() {
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains a non-finite coordinate at point {index}, axis {axis}"
                )));
            }
            if !interval.contains(f64::from(value)) {
                return Err(PyValueError::new_err(format!(
                    "{name} coordinate at point {index}, axis {axis} is outside [{}, {}]",
                    interval.lower(),
                    interval.upper()
                )));
            }
        }
    }
    Ok(point_count)
}

fn vector_batch(
    name: &str,
    shape: &[usize],
    vector_len: usize,
    capacity: usize,
) -> PyResult<(usize, bool)> {
    let (batch, explicit) = match shape {
        [length] if *length == vector_len => (1, false),
        [batch, length] if *length == vector_len => (*batch, true),
        _ => {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({vector_len},) or (B, {vector_len})"
            )))
        }
    };
    if batch == 0 || batch > capacity {
        return Err(PyValueError::new_err(format!(
            "{name} batch {batch} is outside the plan capacity 1..={capacity}"
        )));
    }
    Ok((batch, explicit))
}

fn mode_batch(
    name: &str,
    shape: &[usize],
    n_modes: &[usize],
    capacity: usize,
) -> PyResult<(usize, bool)> {
    if shape == n_modes {
        return Ok((1, false));
    }
    if shape.len() == n_modes.len() + 1 && &shape[1..] == n_modes {
        let batch = shape[0];
        if batch >= 1 && batch <= capacity {
            return Ok((batch, true));
        }
        return Err(PyValueError::new_err(format!(
            "{name} batch {batch} is outside the plan capacity 1..={capacity}"
        )));
    }
    Err(PyValueError::new_err(format!(
        "{name} must have shape {:?} or (B, {:?})",
        n_modes, n_modes
    )))
}

fn parse_sign(isign: i32) -> PyResult<i32> {
    if matches!(isign, -1 | 1) {
        Ok(isign)
    } else {
        Err(PyValueError::new_err("isign must be exactly -1 or 1"))
    }
}

fn parse_mode_order(value: &str) -> PyResult<ModeOrder> {
    match value {
        "centered" => Ok(ModeOrder::Centered),
        "fft" => Ok(ModeOrder::Fft),
        _ => Err(PyValueError::new_err(
            "mode_order must be 'centered' or 'fft'",
        )),
    }
}

fn checked_mode_count(n_modes: &[usize]) -> PyResult<usize> {
    n_modes.iter().try_fold(1usize, |count, &length| {
        count
            .checked_mul(length)
            .ok_or_else(|| PyValueError::new_err("total Fourier mode count overflows usize"))
    })
}

fn core_to_python_index(mut core_index: usize, shape: &[usize]) -> usize {
    let mut python_index = 0usize;
    for &length in shape {
        let coordinate = core_index % length;
        core_index /= length;
        python_index = python_index * length + coordinate;
    }
    python_index
}

fn pack_python_modes(values: &[Complex32], shape: &[usize], batch: usize) -> Vec<Complex32> {
    let mode_count = values.len() / batch;
    let mut packed = vec![Complex32::new(0.0, 0.0); values.len()];
    for transform in 0..batch {
        let offset = transform * mode_count;
        for core_index in 0..mode_count {
            let python_index = core_to_python_index(core_index, shape);
            packed[offset + core_index] = values[offset + python_index];
        }
    }
    packed
}

fn unpack_core_modes(values: Vec<Complex32>, shape: &[usize], batch: usize) -> Vec<Complex32> {
    let mode_count = values.len() / batch;
    let mut unpacked = vec![Complex32::new(0.0, 0.0); values.len()];
    for transform in 0..batch {
        let offset = transform * mode_count;
        for core_index in 0..mode_count {
            let python_index = core_to_python_index(core_index, shape);
            unpacked[offset + python_index] = values[offset + core_index];
        }
    }
    unpacked
}

fn complex_output<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    values: Vec<Complex32>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let array = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
        WgpuNufftError::new_err(format!("could not construct output array: {error}"))
    })?;
    Ok(array.into_pyarray(py))
}

struct Type12State {
    plan: NufftPlan,
    buffers: ExecutionBuffers,
}

#[derive(Clone, Copy)]
enum Type12Kind {
    Type1,
    Type2,
}

#[allow(clippy::too_many_arguments)]
fn build_type12_plan(
    py: Python<'_>,
    context: &PyContext,
    n_modes: Vec<usize>,
    eps: f64,
    isign: i32,
    mode_order: &str,
    batch: usize,
    kind: Type12Kind,
) -> PyResult<(Arc<ContextInner>, Arc<Mutex<Type12State>>, ModeOrder)> {
    let isign = parse_sign(isign)?;
    let order = parse_mode_order(mode_order)?;
    let config = NufftConfig::new(n_modes, eps)
        .with_isign(isign)
        .with_mode_order(order)
        .with_batch(batch);
    config
        .validate()
        .map_err(|error| PyValueError::new_err(format!("invalid NUFFT plan: {error}")))?;

    let runtime = Arc::clone(&context.inner);
    let plan_runtime = Arc::clone(&runtime);
    let plan = py
        .detach(move || {
            let _cache_guard = PipelineCacheClearGuard(&plan_runtime.device);
            match kind {
                Type12Kind::Type1 => {
                    NufftPlan::type1_gpu(&plan_runtime.device, &plan_runtime.queue, config)
                }
                Type12Kind::Type2 => {
                    NufftPlan::type2_gpu(&plan_runtime.device, &plan_runtime.queue, config)
                }
            }
        })
        .map_err(|error| WgpuNufftError::new_err(format!("GPU plan creation failed: {error}")))?;

    Ok((
        runtime,
        Arc::new(Mutex::new(Type12State {
            plan,
            buffers: ExecutionBuffers::default(),
        })),
        order,
    ))
}

fn run_type1(
    context: &ContextInner,
    state: &mut Type12State,
    point_count: usize,
    active_batch: usize,
    points: &[f32],
    strengths: &[Complex32],
) -> Result<Vec<Complex32>, String> {
    let point_bytes = state
        .plan
        .required_point_buffer_size_bytes(point_count)
        .map_err(|error| format!("type-1 point size failed: {error}"))?;
    let strength_bytes = state
        .plan
        .required_type1_strength_buffer_size_bytes_for_batch(point_count, active_batch)
        .map_err(|error| format!("type-1 input size failed: {error}"))?;
    let output_bytes = state
        .plan
        .required_type1_output_buffer_size_bytes_for_batch(active_batch)
        .map_err(|error| format!("type-1 output size failed: {error}"))?;

    let point_buffer = upload_words(
        context,
        &mut state.buffers.points,
        points,
        point_bytes,
        "wgpu_nufft.python.type1.points",
    )?;
    let strength_words = complex_words(strengths);
    let strength_buffer = upload_words(
        context,
        &mut state.buffers.values,
        &strength_words,
        strength_bytes,
        "wgpu_nufft.python.type1.strengths",
    )?;
    let (output, readback) = output_buffers(context, &mut state.buffers, output_bytes)?;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_nufft.python.type1.encoder"),
        });
    state
        .plan
        .encode_type1_gpu_batch(
            &context.device,
            &mut encoder,
            point_count,
            active_batch,
            &point_buffer,
            &strength_buffer,
            &output,
        )
        .map_err(|error| format!("type-1 execution encoding failed: {error}"))?;
    submit_and_read(context, encoder, &output, &readback, output_bytes)
}

fn run_type2(
    context: &ContextInner,
    state: &mut Type12State,
    n_modes: &[usize],
    point_count: usize,
    active_batch: usize,
    points: &[f32],
    coefficients: &[Complex32],
) -> Result<Vec<Complex32>, String> {
    let point_bytes = state
        .plan
        .required_point_buffer_size_bytes(point_count)
        .map_err(|error| format!("type-2 point size failed: {error}"))?;
    let coefficient_bytes = state
        .plan
        .required_type2_coefficient_buffer_size_bytes_for_batch(active_batch)
        .map_err(|error| format!("type-2 input size failed: {error}"))?;
    let output_bytes = state
        .plan
        .required_type2_output_buffer_size_bytes_for_batch(point_count, active_batch)
        .map_err(|error| format!("type-2 output size failed: {error}"))?;

    let point_buffer = upload_words(
        context,
        &mut state.buffers.points,
        points,
        point_bytes,
        "wgpu_nufft.python.type2.points",
    )?;
    let packed = pack_python_modes(coefficients, n_modes, active_batch);
    let coefficient_words = complex_words(&packed);
    let coefficient_buffer = upload_words(
        context,
        &mut state.buffers.values,
        &coefficient_words,
        coefficient_bytes,
        "wgpu_nufft.python.type2.coefficients",
    )?;
    let (output, readback) = output_buffers(context, &mut state.buffers, output_bytes)?;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_nufft.python.type2.encoder"),
        });
    state
        .plan
        .encode_type2_gpu_batch(
            &context.device,
            &mut encoder,
            point_count,
            active_batch,
            &point_buffer,
            &coefficient_buffer,
            &output,
        )
        .map_err(|error| format!("type-2 execution encoding failed: {error}"))?;
    submit_and_read(context, encoder, &output, &readback, output_bytes)
}

fn type1_output_shape(n_modes: &[usize], batch: usize, explicit_batch: bool) -> Vec<usize> {
    if explicit_batch {
        let mut shape = Vec::with_capacity(n_modes.len() + 1);
        shape.push(batch);
        shape.extend_from_slice(n_modes);
        shape
    } else {
        n_modes.to_vec()
    }
}

fn point_output_shape(point_count: usize, batch: usize, explicit_batch: bool) -> Vec<usize> {
    if explicit_batch {
        vec![batch, point_count]
    } else {
        vec![point_count]
    }
}

/// Reusable f32 type-1 NUFFT plan with NumPy host-array transfer.
#[pyclass(module = "wgpu_nufft._native", frozen)]
struct Type1Plan {
    context: Arc<ContextInner>,
    state: Arc<Mutex<Type12State>>,
    n_modes: Vec<usize>,
    mode_count: usize,
    eps: f64,
    isign: i32,
    mode_order: ModeOrder,
    batch_capacity: usize,
}

#[pymethods]
impl Type1Plan {
    #[new]
    #[pyo3(signature = (context, n_modes, eps=1.0e-6, isign=1, mode_order="centered", batch=1))]
    fn new(
        py: Python<'_>,
        context: PyRef<'_, PyContext>,
        n_modes: Vec<usize>,
        eps: f64,
        isign: i32,
        mode_order: &str,
        batch: usize,
    ) -> PyResult<Self> {
        let mode_count = checked_mode_count(&n_modes)?;
        let (context, state, mode_order_value) = build_type12_plan(
            py,
            &context,
            n_modes.clone(),
            eps,
            isign,
            mode_order,
            batch,
            Type12Kind::Type1,
        )?;
        Ok(Self {
            context,
            state,
            n_modes,
            mode_count,
            eps,
            isign,
            mode_order: mode_order_value,
            batch_capacity: batch,
        })
    }

    /// Execute one transform or a transform-major batch sharing one point set.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArrayDyn<'py, f32>,
        strengths: PyReadonlyArrayDyn<'py, Complex32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
        let (point_shape, points) = copy_c_array("points", points)?;
        let point_count =
            validate_type12_points("points", &point_shape, &points, self.n_modes.len())?;
        let (strength_shape, strengths) = copy_c_array("strengths", strengths)?;
        validate_complex("strengths", &strengths)?;
        let (active_batch, explicit_batch) = vector_batch(
            "strengths",
            &strength_shape,
            point_count,
            self.batch_capacity,
        )?;
        let output_shape = type1_output_shape(&self.n_modes, active_batch, explicit_batch);
        if point_count == 0 {
            return complex_output(
                py,
                output_shape,
                vec![Complex32::new(0.0, 0.0); active_batch * self.mode_count],
            );
        }

        let context = Arc::clone(&self.context);
        let state = Arc::clone(&self.state);
        let n_modes = self.n_modes.clone();
        let output = py
            .detach(move || {
                let mut state = state
                    .lock()
                    .map_err(|_| "type-1 plan lock is poisoned".to_owned())?;
                let core = run_type1(
                    &context,
                    &mut state,
                    point_count,
                    active_batch,
                    &points,
                    &strengths,
                )?;
                Ok::<_, String>(unpack_core_modes(core, &n_modes, active_batch))
            })
            .map_err(WgpuNufftError::new_err)?;
        complex_output(py, output_shape, output)
    }

    #[getter]
    fn n_modes(&self) -> Vec<usize> {
        self.n_modes.clone()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.n_modes.len()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.eps
    }

    #[getter]
    fn isign(&self) -> i32 {
        self.isign
    }

    #[getter]
    fn mode_order(&self) -> &'static str {
        match self.mode_order {
            ModeOrder::Centered => "centered",
            ModeOrder::Fft => "fft",
        }
    }

    #[getter]
    fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }
}

/// Reusable f32 type-2 NUFFT plan with NumPy host-array transfer.
#[pyclass(module = "wgpu_nufft._native", frozen)]
struct Type2Plan {
    context: Arc<ContextInner>,
    state: Arc<Mutex<Type12State>>,
    n_modes: Vec<usize>,
    eps: f64,
    isign: i32,
    mode_order: ModeOrder,
    batch_capacity: usize,
}

#[pymethods]
impl Type2Plan {
    #[new]
    #[pyo3(signature = (context, n_modes, eps=1.0e-6, isign=1, mode_order="centered", batch=1))]
    fn new(
        py: Python<'_>,
        context: PyRef<'_, PyContext>,
        n_modes: Vec<usize>,
        eps: f64,
        isign: i32,
        mode_order: &str,
        batch: usize,
    ) -> PyResult<Self> {
        checked_mode_count(&n_modes)?;
        let (context, state, mode_order_value) = build_type12_plan(
            py,
            &context,
            n_modes.clone(),
            eps,
            isign,
            mode_order,
            batch,
            Type12Kind::Type2,
        )?;
        Ok(Self {
            context,
            state,
            n_modes,
            eps,
            isign,
            mode_order: mode_order_value,
            batch_capacity: batch,
        })
    }

    /// Execute one transform or a transform-major batch sharing one point set.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArrayDyn<'py, f32>,
        coefficients: PyReadonlyArrayDyn<'py, Complex32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
        let (point_shape, points) = copy_c_array("points", points)?;
        let point_count =
            validate_type12_points("points", &point_shape, &points, self.n_modes.len())?;
        let (coefficient_shape, coefficients) = copy_c_array("coefficients", coefficients)?;
        validate_complex("coefficients", &coefficients)?;
        let (active_batch, explicit_batch) = mode_batch(
            "coefficients",
            &coefficient_shape,
            &self.n_modes,
            self.batch_capacity,
        )?;
        let output_shape = point_output_shape(point_count, active_batch, explicit_batch);
        if point_count == 0 {
            return complex_output(py, output_shape, Vec::new());
        }

        let context = Arc::clone(&self.context);
        let state = Arc::clone(&self.state);
        let n_modes = self.n_modes.clone();
        let output = py
            .detach(move || {
                let mut state = state
                    .lock()
                    .map_err(|_| "type-2 plan lock is poisoned".to_owned())?;
                run_type2(
                    &context,
                    &mut state,
                    &n_modes,
                    point_count,
                    active_batch,
                    &points,
                    &coefficients,
                )
            })
            .map_err(WgpuNufftError::new_err)?;
        complex_output(py, output_shape, output)
    }

    #[getter]
    fn n_modes(&self) -> Vec<usize> {
        self.n_modes.clone()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.n_modes.len()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.eps
    }

    #[getter]
    fn isign(&self) -> i32 {
        self.isign
    }

    #[getter]
    fn mode_order(&self) -> &'static str {
        match self.mode_order {
            ModeOrder::Centered => "centered",
            ModeOrder::Fft => "fft",
        }
    }

    #[getter]
    fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }
}

struct Type3State {
    plan: NufftType3Plan,
    buffers: ExecutionBuffers,
}

fn copy_bounds(name: &str, array: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Vec<NufftInterval>> {
    let (shape, values) = copy_c_array(name, array)?;
    if shape.len() != 2 || shape[1] != 2 || shape[0] == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (d, 2) with d >= 1"
        )));
    }
    let mut intervals = Vec::with_capacity(shape[0]);
    for (axis, endpoints) in values.as_chunks::<2>().0.iter().enumerate() {
        let lower = endpoints[0];
        let upper = endpoints[1];
        if !lower.is_finite() || !upper.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} axis {axis} endpoints must be finite"
            )));
        }
        if lower > upper {
            return Err(PyValueError::new_err(format!(
                "{name} axis {axis} lower endpoint exceeds its upper endpoint"
            )));
        }
        intervals.push(NufftInterval::new(lower, upper));
    }
    Ok(intervals)
}

#[allow(clippy::too_many_arguments)]
fn run_type3(
    context: &ContextInner,
    state: &mut Type3State,
    active_batch: usize,
    source_count: usize,
    source_points: &[f32],
    strengths: &[Complex32],
    target_count: usize,
    target_points: &[f32],
) -> Result<Vec<Complex32>, String> {
    let source_bytes = state
        .plan
        .required_source_point_buffer_size_bytes(source_count)
        .map_err(|error| format!("type-3 source point size failed: {error}"))?;
    let target_bytes = state
        .plan
        .required_target_point_buffer_size_bytes(target_count)
        .map_err(|error| format!("type-3 target point size failed: {error}"))?;
    let strength_bytes = state
        .plan
        .required_strength_buffer_size_bytes_for_batch(source_count, active_batch)
        .map_err(|error| format!("type-3 input size failed: {error}"))?;
    let output_bytes = state
        .plan
        .required_output_buffer_size_bytes_for_batch(target_count, active_batch)
        .map_err(|error| format!("type-3 output size failed: {error}"))?;

    let source_buffer = upload_words(
        context,
        &mut state.buffers.points,
        source_points,
        source_bytes,
        "wgpu_nufft.python.type3.source_points",
    )?;
    let strength_words = complex_words(strengths);
    let strength_buffer = upload_words(
        context,
        &mut state.buffers.values,
        &strength_words,
        strength_bytes,
        "wgpu_nufft.python.type3.strengths",
    )?;
    let target_buffer = upload_words(
        context,
        &mut state.buffers.targets,
        target_points,
        target_bytes,
        "wgpu_nufft.python.type3.target_points",
    )?;
    let (output, readback) = output_buffers(context, &mut state.buffers, output_bytes)?;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_nufft.python.type3.encoder"),
        });
    state
        .plan
        .encode_gpu_batch(
            &context.device,
            &mut encoder,
            active_batch,
            source_count,
            &source_buffer,
            &strength_buffer,
            target_count,
            &target_buffer,
            &output,
        )
        .map_err(|error| format!("type-3 execution encoding failed: {error}"))?;
    submit_and_read(context, encoder, &output, &readback, output_bytes)
}

/// Reusable f32 type-3 NUFFT plan with fixed source and target bounds.
#[pyclass(module = "wgpu_nufft._native", frozen)]
struct Type3Plan {
    context: Arc<ContextInner>,
    state: Arc<Mutex<Type3State>>,
    source_bounds: Vec<NufftInterval>,
    target_bounds: Vec<NufftInterval>,
    outer_grid_shape: Vec<usize>,
    eps: f64,
    isign: i32,
    batch_capacity: usize,
}

#[pymethods]
impl Type3Plan {
    #[new]
    #[pyo3(signature = (context, source_bounds, target_bounds, eps=1.0e-6, isign=1, batch=1))]
    fn new(
        py: Python<'_>,
        context: PyRef<'_, PyContext>,
        source_bounds: PyReadonlyArrayDyn<'_, f64>,
        target_bounds: PyReadonlyArrayDyn<'_, f64>,
        eps: f64,
        isign: i32,
        batch: usize,
    ) -> PyResult<Self> {
        let isign = parse_sign(isign)?;
        let source_bounds = copy_bounds("source_bounds", source_bounds)?;
        let target_bounds = copy_bounds("target_bounds", target_bounds)?;
        let config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), eps)
            .with_isign(isign)
            .with_batch(batch);
        config
            .validate()
            .map_err(|error| PyValueError::new_err(format!("invalid type-3 plan: {error}")))?;

        let runtime = Arc::clone(&context.inner);
        let plan_runtime = Arc::clone(&runtime);
        let plan = py
            .detach(move || {
                let _cache_guard = PipelineCacheClearGuard(&plan_runtime.device);
                NufftType3Plan::new_gpu(&plan_runtime.device, &plan_runtime.queue, config)
            })
            .map_err(|error| {
                WgpuNufftError::new_err(format!("type-3 GPU plan creation failed: {error}"))
            })?;
        let outer_grid_shape = plan.outer_grid_shape();

        Ok(Self {
            context: runtime,
            state: Arc::new(Mutex::new(Type3State {
                plan,
                buffers: ExecutionBuffers::default(),
            })),
            source_bounds,
            target_bounds,
            outer_grid_shape,
            eps,
            isign,
            batch_capacity: batch,
        })
    }

    /// Execute one transform or a transform-major batch for the supplied point sets.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        source_points: PyReadonlyArrayDyn<'py, f32>,
        strengths: PyReadonlyArrayDyn<'py, Complex32>,
        target_points: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
        let (source_shape, source_points) = copy_c_array("source_points", source_points)?;
        let source_count = validate_type3_points(
            "source_points",
            &source_shape,
            &source_points,
            &self.source_bounds,
        )?;
        let (target_shape, target_points) = copy_c_array("target_points", target_points)?;
        let target_count = validate_type3_points(
            "target_points",
            &target_shape,
            &target_points,
            &self.target_bounds,
        )?;
        let (strength_shape, strengths) = copy_c_array("strengths", strengths)?;
        validate_complex("strengths", &strengths)?;
        let (active_batch, explicit_batch) = vector_batch(
            "strengths",
            &strength_shape,
            source_count,
            self.batch_capacity,
        )?;
        let output_shape = point_output_shape(target_count, active_batch, explicit_batch);
        if target_count == 0 {
            return complex_output(py, output_shape, Vec::new());
        }
        if source_count == 0 {
            return complex_output(
                py,
                output_shape,
                vec![Complex32::new(0.0, 0.0); active_batch * target_count],
            );
        }

        let context = Arc::clone(&self.context);
        let state = Arc::clone(&self.state);
        let output = py
            .detach(move || {
                let mut state = state
                    .lock()
                    .map_err(|_| "type-3 plan lock is poisoned".to_owned())?;
                run_type3(
                    &context,
                    &mut state,
                    active_batch,
                    source_count,
                    &source_points,
                    &strengths,
                    target_count,
                    &target_points,
                )
            })
            .map_err(WgpuNufftError::new_err)?;
        complex_output(py, output_shape, output)
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.source_bounds.len()
    }

    #[getter]
    fn source_bounds(&self) -> Vec<(f64, f64)> {
        self.source_bounds
            .iter()
            .map(|interval| (interval.lower(), interval.upper()))
            .collect()
    }

    #[getter]
    fn target_bounds(&self) -> Vec<(f64, f64)> {
        self.target_bounds
            .iter()
            .map(|interval| (interval.lower(), interval.upper()))
            .collect()
    }

    #[getter]
    fn outer_grid_shape(&self) -> Vec<usize> {
        self.outer_grid_shape.clone()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.eps
    }

    #[getter]
    fn isign(&self) -> i32 {
        self.isign
    }

    #[getter]
    fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("WgpuNufftError", m.py().get_type::<WgpuNufftError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyContext>()?;
    m.add_class::<Type1Plan>()?;
    m.add_class::<Type2Plan>()?;
    m.add_class::<Type3Plan>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_square_mode_repacking_is_inverse() {
        let shape = [2, 3];
        let values = (0..6)
            .map(|value| Complex32::new(value as f32, 0.0))
            .collect::<Vec<_>>();
        let packed = pack_python_modes(&values, &shape, 1);
        assert_eq!(
            packed.iter().map(|value| value.re).collect::<Vec<_>>(),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
        assert_eq!(unpack_core_modes(packed, &shape, 1), values);
    }
}
