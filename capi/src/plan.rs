//! Plans over host memory, on the shared GPU device or on the CPU.

use std::f64::consts::PI;
use std::ffi::CStr;
use std::sync::Arc;

use wgpu_nufft::{
    wgpu, CpuNufftPlan, CpuNufftPoints, CpuNufftType3Plan, CpuNufftType3Points, FftPrecision,
    ModeOrder, NufftConfig, NufftInterval, NufftPlan, NufftType3Config, NufftType3Plan,
    MAX_NUFFT_DIMENSIONS,
};

use crate::adapter::{AdapterSelection, PciAddress};
use crate::error::{invalid, Error, Result};
use crate::gpu::{self, BufferSlot, GpuContext, OutputSlots};

pub(crate) const BACKEND_AUTO: i32 = 0;
pub(crate) const BACKEND_GPU: i32 = 1;
pub(crate) const BACKEND_CPU: i32 = 2;

pub(crate) const PRECISION_AUTO: i32 = 0;
pub(crate) const PRECISION_F64: i32 = 1;
pub(crate) const PRECISION_DF64: i32 = 2;
pub(crate) const PRECISION_F32: i32 = 3;

/// Bytes of `wgpu_nufft_opts.adapter_name`, the terminating NUL included.
pub const ADAPTER_NAME_SIZE: usize = 256;
/// Bytes of `wgpu_nufft_opts.adapter_pci_bus_id`, the terminating NUL included.
pub const PCI_BUS_ID_SIZE: usize = 32;

/// `wgpu_nufft_opts`; the strings are NUL-terminated `char` arrays in C.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opts {
    pub backend: i32,
    pub precision: i32,
    pub mode_order: i32,
    pub threads: i32,
    pub sigma: f64,
    pub adapter_index: i32,
    pub adapter_name: [u8; ADAPTER_NAME_SIZE],
    pub adapter_pci_bus_id: [u8; PCI_BUS_ID_SIZE],
}

/// All zero, which C callers may rely on.
impl Default for Opts {
    fn default() -> Self {
        Self {
            backend: BACKEND_AUTO,
            precision: PRECISION_AUTO,
            mode_order: 0,
            threads: 0,
            sigma: 0.0,
            adapter_index: 0,
            adapter_name: [0; ADAPTER_NAME_SIZE],
            adapter_pci_bus_id: [0; PCI_BUS_ID_SIZE],
        }
    }
}

/// The text of a NUL-terminated `char` array field.
fn text_field<'a>(bytes: &'a [u8], name: &str) -> Result<&'a str> {
    CStr::from_bytes_until_nul(bytes)
        .map_err(|_| {
            invalid(format!(
                "{name} must end with a NUL within its {} bytes",
                bytes.len()
            ))
        })?
        .to_str()
        .map_err(|_| invalid(format!("{name} must be UTF-8")))
}

impl Opts {
    fn adapter_selection(&self) -> Result<AdapterSelection> {
        let index = u32::try_from(self.adapter_index).map_err(|_| {
            invalid(format!(
                "adapter_index must not be negative, not {}",
                self.adapter_index
            ))
        })?;
        let address = text_field(&self.adapter_pci_bus_id, "adapter_pci_bus_id")?.trim();
        let pci_bus_id = if address.is_empty() {
            None
        } else {
            Some(PciAddress::parse(address).ok_or_else(|| {
                invalid(format!(
                    "adapter_pci_bus_id \"{address}\" is not a PCI address such as 0000:01:00.0"
                ))
            })?)
        };
        Ok(AdapterSelection {
            name: text_field(&self.adapter_name, "adapter_name")?
                .trim()
                .to_owned(),
            pci_bus_id,
            index,
        })
    }

    fn validate(&self) -> Result<()> {
        if !(BACKEND_AUTO..=BACKEND_CPU).contains(&self.backend) {
            return Err(invalid(format!("unknown backend option {}", self.backend)));
        }
        if !(PRECISION_AUTO..=PRECISION_F32).contains(&self.precision) {
            return Err(invalid(format!(
                "unknown precision option {}",
                self.precision
            )));
        }
        if !(0..=1).contains(&self.mode_order) {
            return Err(invalid(format!(
                "unknown mode order option {}",
                self.mode_order
            )));
        }
        if self.threads < 0 {
            return Err(invalid("threads must not be negative"));
        }
        if !(self.sigma == 0.0 || (self.sigma.is_finite() && self.sigma > 1.0)) {
            return Err(invalid(
                "sigma must be 0 (the default) or a finite value above 1",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    Type1,
    Type2,
    Type3,
}

impl Kind {
    pub(crate) fn from_code(code: i32) -> Result<Self> {
        match code {
            1 => Ok(Self::Type1),
            2 => Ok(Self::Type2),
            3 => Ok(Self::Type3),
            _ => Err(invalid(format!("type must be 1, 2 or 3, not {code}"))),
        }
    }
}

/// The element type of a plan's arrays at the interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Data {
    F32,
    F64,
}

/// What a plan computes, as `makeplan` received it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Spec {
    pub(crate) kind: Kind,
    pub(crate) dimensions: usize,
    pub(crate) n_modes: Vec<usize>,
    pub(crate) isign: i32,
    pub(crate) ntrans: usize,
    pub(crate) eps: f64,
    pub(crate) data: Data,
    /// The options, with the adapter fields cleared: they are in `adapter`,
    /// where equal selections compare equal.
    pub(crate) opts: Opts,
    pub(crate) adapter: AdapterSelection,
}

impl Spec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kind: Kind,
        dimensions: i32,
        n_modes: &[i64],
        isign: i32,
        ntrans: i64,
        eps: f64,
        data: Data,
        opts: Opts,
    ) -> Result<Self> {
        opts.validate()?;
        let adapter = opts.adapter_selection()?;
        let opts = Opts {
            adapter_index: 0,
            adapter_name: [0; ADAPTER_NAME_SIZE],
            adapter_pci_bus_id: [0; PCI_BUS_ID_SIZE],
            ..opts
        };
        let dimensions = usize::try_from(dimensions)
            .ok()
            .filter(|d| (1..=MAX_NUFFT_DIMENSIONS).contains(d))
            .ok_or_else(|| {
                invalid(format!(
                    "dim must be between 1 and {MAX_NUFFT_DIMENSIONS}, not {dimensions}"
                ))
            })?;
        let n_modes = if kind == Kind::Type3 {
            Vec::new()
        } else {
            if n_modes.len() != dimensions {
                return Err(invalid("n_modes must hold one mode count per dimension"));
            }
            n_modes
                .iter()
                .map(|&n| {
                    usize::try_from(n)
                        .ok()
                        .filter(|&n| n >= 1)
                        .ok_or_else(|| invalid(format!("mode counts must be positive, not {n}")))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let ntrans = usize::try_from(ntrans)
            .ok()
            .filter(|&n| n >= 1)
            .ok_or_else(|| invalid(format!("ntrans must be positive, not {ntrans}")))?;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(invalid(format!(
                "eps must be positive and finite, not {eps}"
            )));
        }
        Ok(Self {
            kind,
            dimensions,
            n_modes,
            isign: if isign >= 0 { 1 } else { -1 },
            ntrans,
            eps,
            data,
            opts,
            adapter,
        })
    }

    fn mode_count(&self) -> Result<usize> {
        self.n_modes
            .iter()
            .try_fold(1usize, |count, &n| count.checked_mul(n))
            .ok_or_else(|| invalid("the mode count overflows"))
    }

    fn mode_order(&self) -> ModeOrder {
        if self.opts.mode_order == 1 {
            ModeOrder::Fft
        } else {
            ModeOrder::Centered
        }
    }

    fn type12_config(&self, precision: FftPrecision) -> NufftConfig {
        let config = NufftConfig::new(self.n_modes.clone(), self.eps)
            .with_isign(self.isign)
            .with_mode_order(self.mode_order())
            .with_batch(self.ntrans)
            .with_precision(precision);
        if self.opts.sigma > 0.0 {
            config.with_sigma(self.opts.sigma)
        } else {
            config
        }
    }

    fn type3_config(
        &self,
        sources: Vec<NufftInterval>,
        targets: Vec<NufftInterval>,
        precision: FftPrecision,
    ) -> NufftType3Config {
        let config = NufftType3Config::new(sources, targets, self.eps)
            .with_isign(self.isign)
            .with_batch(self.ntrans)
            .with_precision(precision);
        if self.opts.sigma > 0.0 {
            config.with_sigma(self.opts.sigma)
        } else {
            config
        }
    }

    /// The GPU arithmetic for this plan on a device with or without `f64`.
    fn gpu_precision(&self, supports_f64: bool) -> Result<FftPrecision> {
        Ok(match (self.opts.precision, self.data) {
            (PRECISION_F32, _) | (PRECISION_AUTO, Data::F32) => FftPrecision::F32,
            (PRECISION_DF64, _) => FftPrecision::Df64,
            (PRECISION_F64, _) if !supports_f64 => {
                return Err(Error::GpuUnavailable(
                    "the GPU lacks native f64; use the Df64 or automatic precision".to_owned(),
                ))
            }
            (_, _) if supports_f64 => FftPrecision::F64,
            _ => FftPrecision::Df64,
        })
    }

    /// The CPU arithmetic: `f32`, or `f64` for every double format.
    fn cpu_precision(&self) -> FftPrecision {
        match (self.opts.precision, self.data) {
            (PRECISION_F32, _) | (PRECISION_AUTO, Data::F32) => FftPrecision::F32,
            _ => FftPrecision::F64,
        }
    }

    fn threads(&self) -> Option<usize> {
        (self.opts.threads > 0).then_some(self.opts.threads as usize)
    }
}

/// A plan with its engine and the points last set.
pub(crate) struct Plan {
    spec: Spec,
    mode_count: usize,
    engine: Engine,
    /// Source points and type-3 targets of the last `setpts`.
    counts: Option<(usize, usize)>,
}

enum Engine {
    Gpu(Box<GpuEngine>),
    Cpu(Box<CpuEngine>),
}

struct GpuEngine {
    context: Arc<GpuContext>,
    precision: FftPrecision,
    plan: GpuPlan,
    points: BufferSlot,
    targets: BufferSlot,
    values: BufferSlot,
    outputs: OutputSlots,
}

enum GpuPlan {
    Type12(NufftPlan),
    /// Built from the bounds of the points at `setpts`.
    Type3(Option<(NufftType3Plan, Bounds)>),
}

// Boxed as a whole in `Engine`.
#[allow(clippy::large_enum_variant)]
enum CpuEngine {
    Type12 {
        plan: Box<CpuNufftPlan>,
        points: Option<CpuNufftPoints>,
    },
    Type3 {
        plan: Option<Box<(CpuNufftType3Plan, Bounds)>>,
        points: Option<CpuNufftType3Points>,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct Bounds {
    sources: Vec<NufftInterval>,
    targets: Vec<NufftInterval>,
}

impl Bounds {
    fn of(points: &[f64], targets: &[f64], dimensions: usize) -> Self {
        Self {
            sources: intervals(points, dimensions),
            targets: intervals(targets, dimensions),
        }
    }

    fn contains(&self, other: &Self) -> bool {
        let inside = |outer: &[NufftInterval], inner: &[NufftInterval]| {
            outer
                .iter()
                .zip(inner)
                .all(|(a, b)| a.lower() <= b.lower() && b.upper() <= a.upper())
        };
        inside(&self.sources, &other.sources) && inside(&self.targets, &other.targets)
    }
}

/// The smallest interval per axis holding every point, `[0, 0]` without any.
fn intervals(points: &[f64], dimensions: usize) -> Vec<NufftInterval> {
    (0..dimensions)
        .map(|axis| {
            let mut lower = f64::INFINITY;
            let mut upper = f64::NEG_INFINITY;
            for point in points.chunks_exact(dimensions) {
                lower = lower.min(point[axis]);
                upper = upper.max(point[axis]);
            }
            if lower > upper {
                NufftInterval::new(0.0, 0.0)
            } else {
                NufftInterval::new(lower, upper)
            }
        })
        .collect()
}

/// Creates a plan on the shared device: wgpu-fft caches pipelines per
/// thread, and the cache is cleared afterwards so that no device-owned entry
/// outlives a plan on a thread the caller may end.
fn on_gpu<T>(context: &GpuContext, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let result = gpu::guarded(context, work);
    wgpu_nufft::wgpu_fft::clear_thread_local_pipeline_cache(&context.device);
    result
}

impl Plan {
    pub(crate) fn new(spec: Spec) -> Result<Self> {
        let mode_count = spec.mode_count()?;
        let selection = &spec.adapter;
        let engine = match spec.opts.backend {
            BACKEND_CPU => Engine::Cpu(Box::new(cpu_engine(&spec)?)),
            BACKEND_GPU => Engine::Gpu(Box::new(gpu_engine(&spec, gpu::context(selection)?)?)),
            _ => match gpu::context(selection) {
                // A selected adapter is never traded for the CPU.
                Err(error) if !selection.is_default() => return Err(error),
                context => match context.and_then(|context| gpu_engine(&spec, context)) {
                    Ok(engine) => Engine::Gpu(Box::new(engine)),
                    // The CPU backend serves what the GPU cannot.
                    Err(gpu_error) => {
                        Engine::Cpu(Box::new(cpu_engine(&spec).map_err(|_| gpu_error)?))
                    }
                },
            },
        };
        Ok(Self {
            spec,
            mode_count,
            engine,
            counts: None,
        })
    }

    pub(crate) fn spec(&self) -> &Spec {
        &self.spec
    }

    pub(crate) fn backend(&self) -> i32 {
        match self.engine {
            Engine::Gpu(_) => BACKEND_GPU,
            Engine::Cpu(_) => BACKEND_CPU,
        }
    }

    pub(crate) fn precision(&self) -> i32 {
        let precision = match &self.engine {
            Engine::Gpu(engine) => engine.precision,
            Engine::Cpu(_) => self.spec.cpu_precision(),
        };
        match precision {
            FftPrecision::F32 => PRECISION_F32,
            FftPrecision::F64 => PRECISION_F64,
            FftPrecision::Df64 => PRECISION_DF64,
        }
    }

    pub(crate) fn dimensions(&self) -> usize {
        self.spec.dimensions
    }

    /// `(input, output)` lengths of an execution in real words: two per
    /// complex value.
    pub(crate) fn execution_lengths(&self) -> Result<(usize, usize)> {
        let (points, targets) = self
            .counts
            .ok_or_else(|| invalid("set the points before executing the plan"))?;
        let ntrans = self.spec.ntrans;
        let words = |count: usize| {
            count
                .checked_mul(ntrans)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| invalid("the array length overflows"))
        };
        Ok(match self.spec.kind {
            Kind::Type1 => (words(points)?, words(self.mode_count)?),
            Kind::Type2 => (words(self.mode_count)?, words(points)?),
            Kind::Type3 => (words(points)?, words(targets)?),
        })
    }

    /// Sets `points` (point-major, `dimensions` coordinates per point) and,
    /// for type 3, the target frequencies `targets`.
    pub(crate) fn set_points(&mut self, points: &[f64], targets: &[f64]) -> Result<()> {
        let dimensions = self.spec.dimensions;
        let point_count = points.len() / dimensions;
        let target_count = targets.len() / dimensions;
        self.counts = None;
        // `f32` arithmetic sees rounded coordinates, which type-3 bounds
        // must hold.
        let (rounded_points, rounded_targets);
        let (points, targets) = if self.precision() == PRECISION_F32 {
            rounded_points = round_to_f32(points);
            rounded_targets = round_to_f32(targets);
            (&rounded_points[..], &rounded_targets[..])
        } else {
            (points, targets)
        };
        if self.spec.kind == Kind::Type3 {
            check_finite("target", targets, dimensions)?;
            check_finite("point", points, dimensions)?;
        } else {
            check_type12_points(points, dimensions)?;
        }
        let spec = &self.spec;
        match &mut self.engine {
            Engine::Gpu(engine) => {
                let fallback = engine.set_points(spec, points, targets, point_count, target_count);
                match fallback {
                    Ok(()) => {}
                    // A type-3 plan the GPU cannot build, such as one whose
                    // phases are too large for Df64, falls back to the CPU
                    // when the backend is automatic.
                    Err(error)
                        if spec.kind == Kind::Type3
                            && spec.opts.backend == BACKEND_AUTO
                            && matches!(error, Error::Plan(_) | Error::GpuUnavailable(_)) =>
                    {
                        let mut cpu = cpu_engine(spec)?;
                        cpu.set_points(spec, points, targets).map_err(|_| error)?;
                        self.engine = Engine::Cpu(Box::new(cpu));
                    }
                    Err(error) => return Err(error),
                }
            }
            Engine::Cpu(engine) => engine.set_points(spec, points, targets)?,
        }
        self.counts = Some((point_count, target_count));
        Ok(())
    }

    /// Runs the batch on `input` into `output`, both interleaved complex
    /// values of [`Self::execution_lengths`].
    pub(crate) fn execute(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        let (input_len, output_len) = self.execution_lengths()?;
        if input.len() != input_len || output.len() != output_len {
            return Err(Error::Internal(
                "execution array lengths disagree".to_owned(),
            ));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "the input holds a value that is not finite at complex index {}",
                index / 2
            )));
        }
        let (points, targets) = self.counts.unwrap_or((0, 0));
        match &mut self.engine {
            Engine::Gpu(engine) => engine.execute(&self.spec, points, targets, input, output),
            Engine::Cpu(engine) => engine.execute(input, output),
        }
    }
}

fn check_type12_points(points: &[f64], dimensions: usize) -> Result<()> {
    let limit = 3.0 * PI;
    for (index, &value) in points.iter().enumerate() {
        if value.is_nan() || value.abs() > limit {
            return Err(invalid(format!(
                "point {} has coordinate {value} on axis {}, outside [-3*pi, 3*pi]",
                index / dimensions,
                index % dimensions
            )));
        }
    }
    Ok(())
}

fn check_finite(name: &str, values: &[f64], dimensions: usize) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "{name} {} has a coordinate that is not finite on axis {}",
            index / dimensions,
            index % dimensions
        )));
    }
    Ok(())
}

fn gpu_engine(spec: &Spec, context: Arc<GpuContext>) -> Result<GpuEngine> {
    let precision = spec.gpu_precision(context.supports_f64())?;
    let plan = match spec.kind {
        Kind::Type1 | Kind::Type2 => {
            let config = spec.type12_config(precision);
            let kind = spec.kind;
            GpuPlan::Type12(on_gpu(&context, || {
                let plan = if kind == Kind::Type1 {
                    NufftPlan::type1_gpu(&context.device, &context.queue, config)
                } else {
                    NufftPlan::type2_gpu(&context.device, &context.queue, config)
                };
                plan.map_err(Error::from)
            })?)
        }
        Kind::Type3 => {
            // Checks the parts of the configuration that do not depend on
            // the points now, as type-1 and type-2 plans do.
            let unit = vec![NufftInterval::new(-1.0, 1.0); spec.dimensions];
            spec.type3_config(unit.clone(), unit, precision)
                .validate()?;
            GpuPlan::Type3(None)
        }
    };
    Ok(GpuEngine {
        context,
        precision,
        plan,
        points: BufferSlot::default(),
        targets: BufferSlot::default(),
        values: BufferSlot::default(),
        outputs: OutputSlots::default(),
    })
}

fn cpu_engine(spec: &Spec) -> Result<CpuEngine> {
    let precision = spec.cpu_precision();
    Ok(match spec.kind {
        Kind::Type1 | Kind::Type2 => {
            let config = spec.type12_config(precision);
            let plan = if spec.kind == Kind::Type1 {
                CpuNufftPlan::type1(config)?
            } else {
                CpuNufftPlan::type2(config)?
            };
            let plan = match spec.threads() {
                Some(threads) => plan.with_threads(threads),
                None => plan,
            };
            CpuEngine::Type12 {
                plan: Box::new(plan),
                points: None,
            }
        }
        Kind::Type3 => {
            let unit = vec![NufftInterval::new(-1.0, 1.0); spec.dimensions];
            spec.type3_config(unit.clone(), unit, precision)
                .validate()?;
            CpuEngine::Type3 {
                plan: None,
                points: None,
            }
        }
    })
}

impl GpuEngine {
    fn set_points(
        &mut self,
        spec: &Spec,
        points: &[f64],
        targets: &[f64],
        point_count: usize,
        target_count: usize,
    ) -> Result<()> {
        let context = Arc::clone(&self.context);
        let precision = self.precision;
        match &mut self.plan {
            GpuPlan::Type12(plan) => {
                let bytes = plan.required_point_buffer_size_bytes(point_count)?;
                let words = gpu::device_words(points, precision);
                let slot = &mut self.points;
                gpu::guarded(&context, || {
                    let buffer = slot.upload(&context, &words, bytes, "wgpu_nufft.c.points")?;
                    let mut encoder = encoder(&context, "wgpu_nufft.c.set_points");
                    plan.set_points_gpu(&context.device, &mut encoder, point_count, &buffer)?;
                    context.queue.submit([encoder.finish()]);
                    Ok(())
                })
            }
            GpuPlan::Type3(state) => {
                let bounds = Bounds::of(points, targets, spec.dimensions);
                if !state
                    .as_ref()
                    .is_some_and(|(_, current)| current.contains(&bounds))
                {
                    *state = None;
                    let config = spec.type3_config(
                        bounds.sources.clone(),
                        bounds.targets.clone(),
                        precision,
                    );
                    let plan = on_gpu(&context, || {
                        NufftType3Plan::new_gpu(&context.device, &context.queue, config)
                            .map_err(Error::from)
                    })?;
                    *state = Some((plan, bounds));
                }
                let Some((plan, _)) = state.as_ref() else {
                    return Err(Error::Internal("type-3 plan missing".to_owned()));
                };
                let source_bytes = plan.required_source_point_buffer_size_bytes(point_count)?;
                let target_bytes = plan.required_target_point_buffer_size_bytes(target_count)?;
                let source_words = gpu::device_words(points, precision);
                let target_words = gpu::device_words(targets, precision);
                let (points_slot, targets_slot) = (&mut self.points, &mut self.targets);
                gpu::guarded(&context, || {
                    points_slot.upload(
                        &context,
                        &source_words,
                        source_bytes,
                        "wgpu_nufft.c.sources",
                    )?;
                    targets_slot.upload(
                        &context,
                        &target_words,
                        target_bytes,
                        "wgpu_nufft.c.targets",
                    )?;
                    Ok(())
                })
            }
        }
    }

    fn execute(
        &mut self,
        spec: &Spec,
        point_count: usize,
        target_count: usize,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        let context = Arc::clone(&self.context);
        let precision = self.precision;
        let words = gpu::device_words(input, precision);
        let device = &context.device;
        let (values, outputs) = (&mut self.values, &mut self.outputs);
        let read = match &self.plan {
            GpuPlan::Type12(plan) => {
                let (input_bytes, output_bytes) = match spec.kind {
                    Kind::Type1 => (
                        plan.required_type1_strength_buffer_size_bytes(point_count)?,
                        plan.required_type1_output_buffer_size_bytes()?,
                    ),
                    _ => (
                        plan.required_type2_coefficient_buffer_size_bytes()?,
                        plan.required_type2_output_buffer_size_bytes(point_count)?,
                    ),
                };
                // The plan remembers the points of `set_points`.
                gpu::guarded(&context, || {
                    let input =
                        values.upload(&context, &words, input_bytes, "wgpu_nufft.c.input")?;
                    let result = outputs.output(&context, output_bytes)?;
                    let mut encoder = encoder(&context, "wgpu_nufft.c.execute");
                    if spec.kind == Kind::Type1 {
                        plan.execute_type1_gpu(device, &mut encoder, &input, &result)?;
                    } else {
                        plan.execute_type2_gpu(device, &mut encoder, &input, &result)?;
                    }
                    outputs.submit_and_read(&context, encoder, &result, output_bytes)
                })?
            }
            GpuPlan::Type3(state) => {
                let Some((plan, _)) = state.as_ref() else {
                    return Err(invalid("set the points before executing the plan"));
                };
                let input_bytes = plan.required_strength_buffer_size_bytes(point_count)?;
                let output_bytes = plan.required_output_buffer_size_bytes(target_count)?;
                let sources = self.points.current()?;
                let targets = self.targets.current()?;
                gpu::guarded(&context, || {
                    let input =
                        values.upload(&context, &words, input_bytes, "wgpu_nufft.c.input")?;
                    let result = outputs.output(&context, output_bytes)?;
                    let mut encoder = encoder(&context, "wgpu_nufft.c.execute");
                    plan.encode_gpu(
                        device,
                        &mut encoder,
                        point_count,
                        &sources,
                        &input,
                        target_count,
                        &targets,
                        &result,
                    )?;
                    outputs.submit_and_read(&context, encoder, &result, output_bytes)
                })?
            }
        };
        let values = gpu::host_values(&read, precision);
        if values.len() != output.len() {
            return Err(Error::Internal(format!(
                "the GPU returned {} values for an output of {}",
                values.len(),
                output.len()
            )));
        }
        output.copy_from_slice(&values);
        Ok(())
    }
}

fn encoder(context: &GpuContext, label: &'static str) -> wgpu::CommandEncoder {
    context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

impl CpuEngine {
    fn set_points(&mut self, spec: &Spec, points: &[f64], targets: &[f64]) -> Result<()> {
        let single = spec.cpu_precision() == FftPrecision::F32;
        match self {
            Self::Type12 {
                plan,
                points: prepared,
            } => {
                *prepared = None;
                *prepared = Some(if single {
                    plan.prepare_points(&narrow(points))?
                } else {
                    plan.prepare_points_f64(points)?
                });
            }
            Self::Type3 {
                plan: state,
                points: prepared,
            } => {
                *prepared = None;
                let bounds = Bounds::of(points, targets, spec.dimensions);
                if !state.as_ref().is_some_and(|kept| kept.1.contains(&bounds)) {
                    *state = None;
                    let config = spec.type3_config(
                        bounds.sources.clone(),
                        bounds.targets.clone(),
                        spec.cpu_precision(),
                    );
                    let plan = CpuNufftType3Plan::new(config)?;
                    let plan = match spec.threads() {
                        Some(threads) => plan.with_threads(threads),
                        None => plan,
                    };
                    *state = Some(Box::new((plan, bounds)));
                }
                let Some(kept) = state.as_ref() else {
                    return Err(Error::Internal("type-3 plan missing".to_owned()));
                };
                let plan = &kept.0;
                *prepared = Some(if single {
                    plan.prepare_points(&narrow(points), &narrow(targets))?
                } else {
                    plan.prepare_points_f64(points, targets)?
                });
            }
        }
        Ok(())
    }

    fn execute(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        match self {
            Self::Type12 { plan, points } => {
                let points = points
                    .as_ref()
                    .ok_or_else(|| invalid("set the points before executing the plan"))?;
                if plan.config().precision() == FftPrecision::F32 {
                    let mut words = vec![0.0f32; output.len()];
                    plan.execute_prepared(points, &narrow(input), &mut words)?;
                    widen_into(&words, output);
                } else {
                    plan.execute_prepared_f64(points, input, output)?;
                }
            }
            Self::Type3 { plan, points } => {
                let (Some(kept), Some(points)) = (plan.as_ref(), points.as_ref()) else {
                    return Err(invalid("set the points before executing the plan"));
                };
                let plan = &kept.0;
                if plan.config().precision() == FftPrecision::F32 {
                    let mut words = vec![0.0f32; output.len()];
                    plan.execute_prepared(points, &narrow(input), &mut words)?;
                    widen_into(&words, output);
                } else {
                    plan.execute_prepared_f64(points, input, output)?;
                }
            }
        }
        Ok(())
    }
}

fn round_to_f32(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&value| f64::from(value as f32))
        .collect()
}

fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&value| value as f32).collect()
}

fn widen_into(words: &[f32], output: &mut [f64]) {
    for (target, &word) in output.iter_mut().zip(words) {
        *target = f64::from(word);
    }
}
