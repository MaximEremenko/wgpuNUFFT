//! C interface to `wgpu-nufft`; `include/wgpu_nufft.h` declares it.
//!
//! Plans run on one GPU device shared by the process, or on the CPU, and
//! take host arrays: points are copied into the plan at `setpts`, and every
//! execution uploads its input and reads back its output. The Fortran module
//! and the MATLAB MEX gateway build on these functions.

#![allow(clippy::missing_safety_doc)]

mod adapter;
mod cache;
mod error;
mod gpu;
mod plan;

use std::ffi::{c_char, CString};
use std::sync::{Mutex, OnceLock, PoisonError};

use adapter::AdapterSelection;
use error::{call, invalid, Result};
use plan::{Data, Kind, Plan, Spec};

pub use plan::{Opts, ADAPTER_NAME_SIZE, PCI_BUS_ID_SIZE};

/// A plan behind a C handle, for either array type.
pub struct Handle {
    plan: Plan,
}

/// `wgpu_nufft_adapter`: the strings are NUL-terminated `char` arrays in C.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adapter {
    pub name: [u8; ADAPTER_NAME_SIZE],
    pub backend: [u8; 16],
    pub device_type: [u8; 16],
    pub pci_bus_id: [u8; PCI_BUS_ID_SIZE],
    pub is_default: i32,
}

/// `text` in a NUL-terminated array, cut at a character boundary to fit.
fn c_text<const N: usize>(text: &str) -> [u8; N] {
    let mut end = text.len().min(N - 1);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut bytes = [0; N];
    bytes[..end].copy_from_slice(&text.as_bytes()[..end]);
    bytes
}

/// The element types of the two families of functions.
trait Real: Copy {
    const DATA: Data;
    fn widen(self) -> f64;
    fn narrow(value: f64) -> Self;
}

impl Real for f64 {
    const DATA: Data = Data::F64;
    fn widen(self) -> f64 {
        self
    }
    fn narrow(value: f64) -> Self {
        value
    }
}

impl Real for f32 {
    const DATA: Data = Data::F32;
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn narrow(value: f64) -> Self {
        value as f32
    }
}

fn count(value: i64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(format!("{name} must not be negative, not {value}")))
}

/// `len` values at `pointer`, which may be null only when `len` is zero.
unsafe fn values<'a, T>(pointer: *const T, len: usize, name: &str) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    Ok(std::slice::from_raw_parts(pointer, len))
}

unsafe fn values_mut<'a, T>(pointer: *mut T, len: usize, name: &str) -> Result<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    if pointer.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    Ok(std::slice::from_raw_parts_mut(pointer, len))
}

fn options(opts: *const Opts) -> Opts {
    if opts.is_null() {
        Opts::default()
    } else {
        // SAFETY: a non-null options pointer points to a wgpu_nufft_opts.
        unsafe { *opts }
    }
}

/// Point-major coordinates from one array per axis.
unsafe fn per_axis_points<T: Real>(
    len: usize,
    axes: &[*const T],
    names: &[&str],
) -> Result<Vec<f64>> {
    let columns = axes
        .iter()
        .zip(names)
        .map(|(&axis, name)| values(axis, len, name))
        .collect::<Result<Vec<_>>>()?;
    let mut points = Vec::with_capacity(len * axes.len());
    for index in 0..len {
        points.extend(columns.iter().map(|column| column[index].widen()));
    }
    Ok(points)
}

fn widened<T: Real>(values: &[T]) -> Vec<f64> {
    values.iter().map(|&value| value.widen()).collect()
}

unsafe fn handle<'a, T: Real>(plan: *mut Handle) -> Result<&'a mut Plan> {
    let handle = plan.as_mut().ok_or_else(|| invalid("the plan is null"))?;
    if handle.plan.spec().data != T::DATA {
        return Err(invalid(
            "the plan was made for the other array type; use the matching function family",
        ));
    }
    Ok(&mut handle.plan)
}

#[allow(clippy::too_many_arguments)]
unsafe fn makeplan<T: Real>(
    kind: i32,
    dim: i32,
    n_modes: *const i64,
    isign: i32,
    ntrans: i64,
    eps: f64,
    plan: *mut *mut Handle,
    opts: *const Opts,
) -> i32 {
    call(|| {
        let out = plan
            .as_mut()
            .ok_or_else(|| invalid("the plan pointer is null"))?;
        *out = std::ptr::null_mut();
        let kind = Kind::from_code(kind)?;
        let modes = if kind == Kind::Type3 {
            &[][..]
        } else {
            values(n_modes, usize::try_from(dim).unwrap_or(0), "n_modes")?
        };
        let spec = Spec::new(kind, dim, modes, isign, ntrans, eps, T::DATA, options(opts))?;
        *out = Box::into_raw(Box::new(Handle {
            plan: Plan::new(spec)?,
        }));
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn setpts<T: Real>(
    plan: *mut Handle,
    m: i64,
    x: *const T,
    y: *const T,
    z: *const T,
    n: i64,
    s: *const T,
    t: *const T,
    u: *const T,
) -> i32 {
    call(|| {
        let plan = handle::<T>(plan)?;
        let dimensions = plan.dimensions();
        if dimensions > 3 {
            return Err(invalid(
                "plans of more than three dimensions take their points through setpts_nd",
            ));
        }
        let points = per_axis_points(count(m, "M")?, &[x, y, z][..dimensions], &["x", "y", "z"])?;
        let targets = if plan.spec().kind == Kind::Type3 {
            per_axis_points(count(n, "N")?, &[s, t, u][..dimensions], &["s", "t", "u"])?
        } else {
            Vec::new()
        };
        plan.set_points(&points, &targets)
    })
}

unsafe fn setpts_nd<T: Real>(
    plan: *mut Handle,
    m: i64,
    points: *const T,
    n: i64,
    targets: *const T,
) -> i32 {
    call(|| {
        let plan = handle::<T>(plan)?;
        let dimensions = plan.dimensions();
        let words = |count: usize| {
            count
                .checked_mul(dimensions)
                .ok_or_else(|| invalid("the point array length overflows"))
        };
        let points = widened(values(points, words(count(m, "M")?)?, "points")?);
        let targets = if plan.spec().kind == Kind::Type3 {
            widened(values(targets, words(count(n, "N")?)?, "targets")?)
        } else {
            Vec::new()
        };
        plan.set_points(&points, &targets)
    })
}

unsafe fn execute<T: Real>(plan: *mut Handle, c: *mut T, f: *mut T) -> i32 {
    call(|| {
        let plan = handle::<T>(plan)?;
        let (input_len, output_len) = plan.execution_lengths()?;
        let (input, output, input_name, output_name) = if plan.spec().kind == Kind::Type2 {
            (f, c, "f", "c")
        } else {
            (c, f, "c", "f")
        };
        let input = widened(values(input.cast_const(), input_len, input_name)?);
        let mut result = vec![0.0f64; output_len];
        plan.execute(&input, &mut result)?;
        let output = values_mut(output, output_len, output_name)?;
        for (target, &value) in output.iter_mut().zip(&result) {
            *target = T::narrow(value);
        }
        Ok(())
    })
}

unsafe fn destroy(plan: *mut Handle) {
    if !plan.is_null() {
        // A panic while dropping GPU resources must not unwind into C.
        let _ =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(Box::from_raw(plan))));
    }
}

unsafe fn plan_query(plan: *mut Handle, query: fn(&Plan) -> i32) -> i32 {
    plan.as_ref().map_or(-1, |handle| query(&handle.plan))
}

/// One transform with a kept plan: `axes` and `targets` hold the coordinate
/// arrays of the points and of type-3 targets, `n_modes` the modes of types
/// 1 and 2.
#[allow(clippy::too_many_arguments)]
unsafe fn one_call<T: Real>(
    kind: Kind,
    m: i64,
    axes: &[*const T],
    isign: i32,
    eps: f64,
    n_modes: &[i64],
    n: i64,
    targets: &[*const T],
    input: *const T,
    output: *mut T,
    opts: *const Opts,
) -> i32 {
    call(|| {
        let dimensions = axes.len();
        let spec = Spec::new(
            kind,
            dimensions as i32,
            n_modes,
            isign,
            1,
            eps,
            T::DATA,
            options(opts),
        )?;
        let point_count = count(m, "M")?;
        let points = per_axis_points(point_count, axes, &["x", "y", "z"])?;
        let (target_points, target_count) = if kind == Kind::Type3 {
            let target_count = count(n, "N")?;
            (
                per_axis_points(target_count, targets, &["s", "t", "u"])?,
                target_count,
            )
        } else {
            (Vec::new(), 0)
        };
        let mode_count = n_modes
            .iter()
            .map(|&n| usize::try_from(n).unwrap_or(0))
            .product::<usize>();
        let (input_count, output_count, input_name, output_name) = match kind {
            Kind::Type1 => (point_count, mode_count, "c", "f"),
            Kind::Type2 => (mode_count, point_count, "f", "c"),
            Kind::Type3 => (point_count, target_count, "c", "f"),
        };
        let input = widened(values(input, 2 * input_count, input_name)?);
        let mut result = vec![0.0f64; 2 * output_count];
        cache::run(spec, &points, &target_points, &input, &mut result)?;
        let output = values_mut(output, 2 * output_count, output_name)?;
        for (target, &value) in output.iter_mut().zip(&result) {
            *target = T::narrow(value);
        }
        Ok(())
    })
}

// ------------------------------------------------------------------------
// Library-wide functions

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_default_opts(opts: *mut Opts) {
    if let Some(opts) = opts.as_mut() {
        *opts = Opts::default();
    }
}

#[no_mangle]
pub extern "C" fn wgpu_nufft_last_error() -> *const c_char {
    error::last_error()
}

#[no_mangle]
pub extern "C" fn wgpu_nufft_version() -> *const c_char {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).unwrap_or_default())
        .as_ptr()
}

#[no_mangle]
pub extern "C" fn wgpu_nufft_gpu_name() -> *const c_char {
    static NAME: Mutex<Option<CString>> = Mutex::new(None);
    let name = std::panic::catch_unwind(|| {
        gpu::context(&AdapterSelection::default()).map(|context| context.name.clone())
    })
    .ok()
    .and_then(|name| name.ok())
    .unwrap_or_default();
    let mut slot = NAME.lock().unwrap_or_else(PoisonError::into_inner);
    let name = slot.insert(CString::new(name.replace('\0', " ")).unwrap_or_default());
    name.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_list_adapters(
    adapters: *mut Adapter,
    capacity: i32,
    count: *mut i32,
) -> i32 {
    call(|| {
        let count = count
            .as_mut()
            .ok_or_else(|| invalid("the count pointer is null"))?;
        *count = 0;
        let capacity = usize::try_from(capacity)
            .map_err(|_| invalid(format!("capacity must not be negative, not {capacity}")))?;
        if capacity > 0 && adapters.is_null() {
            return Err(invalid("adapters is null"));
        }
        let listed = adapter::listed();
        for (position, (info, is_default)) in listed.iter().take(capacity).enumerate() {
            // SAFETY: the caller provides `capacity` entries, which may be
            // uninitialized, so they are written without being read.
            adapters.add(position).write(Adapter {
                name: c_text(&info.name),
                backend: c_text(&format!("{:?}", info.backend)),
                device_type: c_text(&format!("{:?}", info.device_type)),
                pci_bus_id: c_text(&info.device_pci_bus_id),
                is_default: i32::from(*is_default),
            });
        }
        *count = i32::try_from(listed.len()).unwrap_or(i32::MAX);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn wgpu_nufft_shutdown() {
    let _ = std::panic::catch_unwind(|| {
        cache::clear();
        gpu::release_context();
    });
}

// ------------------------------------------------------------------------
// Double precision

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_makeplan(
    kind: i32,
    dim: i32,
    n_modes: *const i64,
    isign: i32,
    ntrans: i64,
    eps: f64,
    plan: *mut *mut Handle,
    opts: *const Opts,
) -> i32 {
    makeplan::<f64>(kind, dim, n_modes, isign, ntrans, eps, plan, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_setpts(
    plan: *mut Handle,
    m: i64,
    x: *const f64,
    y: *const f64,
    z: *const f64,
    n: i64,
    s: *const f64,
    t: *const f64,
    u: *const f64,
) -> i32 {
    setpts(plan, m, x, y, z, n, s, t, u)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_setpts_nd(
    plan: *mut Handle,
    m: i64,
    points: *const f64,
    n: i64,
    targets: *const f64,
) -> i32 {
    setpts_nd(plan, m, points, n, targets)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_execute(plan: *mut Handle, c: *mut f64, f: *mut f64) -> i32 {
    execute(plan, c, f)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_destroy(plan: *mut Handle) {
    destroy(plan);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_plan_backend(plan: *mut Handle) -> i32 {
    plan_query(plan, Plan::backend)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft_plan_precision(plan: *mut Handle) -> i32 {
    plan_query(plan, Plan::precision)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft1d1(
    m: i64,
    x: *const f64,
    c: *const f64,
    isign: i32,
    eps: f64,
    ms: i64,
    f: *mut f64,
    opts: *const Opts,
) -> i32 {
    one_call(Kind::Type1, m, &[x], isign, eps, &[ms], 0, &[], c, f, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft1d2(
    m: i64,
    x: *const f64,
    c: *mut f64,
    isign: i32,
    eps: f64,
    ms: i64,
    f: *const f64,
    opts: *const Opts,
) -> i32 {
    one_call(Kind::Type2, m, &[x], isign, eps, &[ms], 0, &[], f, c, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft1d3(
    m: i64,
    x: *const f64,
    c: *const f64,
    isign: i32,
    eps: f64,
    n: i64,
    s: *const f64,
    f: *mut f64,
    opts: *const Opts,
) -> i32 {
    one_call(Kind::Type3, m, &[x], isign, eps, &[], n, &[s], c, f, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft2d1(
    m: i64,
    x: *const f64,
    y: *const f64,
    c: *const f64,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    f: *mut f64,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type1,
        m,
        &[x, y],
        isign,
        eps,
        &[ms, mt],
        0,
        &[],
        c,
        f,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft2d2(
    m: i64,
    x: *const f64,
    y: *const f64,
    c: *mut f64,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    f: *const f64,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type2,
        m,
        &[x, y],
        isign,
        eps,
        &[ms, mt],
        0,
        &[],
        f,
        c,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft2d3(
    m: i64,
    x: *const f64,
    y: *const f64,
    c: *const f64,
    isign: i32,
    eps: f64,
    n: i64,
    s: *const f64,
    t: *const f64,
    f: *mut f64,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type3,
        m,
        &[x, y],
        isign,
        eps,
        &[],
        n,
        &[s, t],
        c,
        f,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft3d1(
    m: i64,
    x: *const f64,
    y: *const f64,
    z: *const f64,
    c: *const f64,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    mu: i64,
    f: *mut f64,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type1,
        m,
        &[x, y, z],
        isign,
        eps,
        &[ms, mt, mu],
        0,
        &[],
        c,
        f,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft3d2(
    m: i64,
    x: *const f64,
    y: *const f64,
    z: *const f64,
    c: *mut f64,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    mu: i64,
    f: *const f64,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type2,
        m,
        &[x, y, z],
        isign,
        eps,
        &[ms, mt, mu],
        0,
        &[],
        f,
        c,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufft3d3(
    m: i64,
    x: *const f64,
    y: *const f64,
    z: *const f64,
    c: *const f64,
    isign: i32,
    eps: f64,
    n: i64,
    s: *const f64,
    t: *const f64,
    u: *const f64,
    f: *mut f64,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type3,
        m,
        &[x, y, z],
        isign,
        eps,
        &[],
        n,
        &[s, t, u],
        c,
        f,
        opts,
    )
}

// ------------------------------------------------------------------------
// Single precision

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_makeplan(
    kind: i32,
    dim: i32,
    n_modes: *const i64,
    isign: i32,
    ntrans: i64,
    eps: f64,
    plan: *mut *mut Handle,
    opts: *const Opts,
) -> i32 {
    makeplan::<f32>(kind, dim, n_modes, isign, ntrans, eps, plan, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_setpts(
    plan: *mut Handle,
    m: i64,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    n: i64,
    s: *const f32,
    t: *const f32,
    u: *const f32,
) -> i32 {
    setpts(plan, m, x, y, z, n, s, t, u)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_setpts_nd(
    plan: *mut Handle,
    m: i64,
    points: *const f32,
    n: i64,
    targets: *const f32,
) -> i32 {
    setpts_nd(plan, m, points, n, targets)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_execute(plan: *mut Handle, c: *mut f32, f: *mut f32) -> i32 {
    execute(plan, c, f)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_destroy(plan: *mut Handle) {
    destroy(plan);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_plan_backend(plan: *mut Handle) -> i32 {
    plan_query(plan, Plan::backend)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf_plan_precision(plan: *mut Handle) -> i32 {
    plan_query(plan, Plan::precision)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf1d1(
    m: i64,
    x: *const f32,
    c: *const f32,
    isign: i32,
    eps: f64,
    ms: i64,
    f: *mut f32,
    opts: *const Opts,
) -> i32 {
    one_call(Kind::Type1, m, &[x], isign, eps, &[ms], 0, &[], c, f, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf1d2(
    m: i64,
    x: *const f32,
    c: *mut f32,
    isign: i32,
    eps: f64,
    ms: i64,
    f: *const f32,
    opts: *const Opts,
) -> i32 {
    one_call(Kind::Type2, m, &[x], isign, eps, &[ms], 0, &[], f, c, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf1d3(
    m: i64,
    x: *const f32,
    c: *const f32,
    isign: i32,
    eps: f64,
    n: i64,
    s: *const f32,
    f: *mut f32,
    opts: *const Opts,
) -> i32 {
    one_call(Kind::Type3, m, &[x], isign, eps, &[], n, &[s], c, f, opts)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf2d1(
    m: i64,
    x: *const f32,
    y: *const f32,
    c: *const f32,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    f: *mut f32,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type1,
        m,
        &[x, y],
        isign,
        eps,
        &[ms, mt],
        0,
        &[],
        c,
        f,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf2d2(
    m: i64,
    x: *const f32,
    y: *const f32,
    c: *mut f32,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    f: *const f32,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type2,
        m,
        &[x, y],
        isign,
        eps,
        &[ms, mt],
        0,
        &[],
        f,
        c,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf2d3(
    m: i64,
    x: *const f32,
    y: *const f32,
    c: *const f32,
    isign: i32,
    eps: f64,
    n: i64,
    s: *const f32,
    t: *const f32,
    f: *mut f32,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type3,
        m,
        &[x, y],
        isign,
        eps,
        &[],
        n,
        &[s, t],
        c,
        f,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf3d1(
    m: i64,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    c: *const f32,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    mu: i64,
    f: *mut f32,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type1,
        m,
        &[x, y, z],
        isign,
        eps,
        &[ms, mt, mu],
        0,
        &[],
        c,
        f,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf3d2(
    m: i64,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    c: *mut f32,
    isign: i32,
    eps: f64,
    ms: i64,
    mt: i64,
    mu: i64,
    f: *const f32,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type2,
        m,
        &[x, y, z],
        isign,
        eps,
        &[ms, mt, mu],
        0,
        &[],
        f,
        c,
        opts,
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_nufftf3d3(
    m: i64,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    c: *const f32,
    isign: i32,
    eps: f64,
    n: i64,
    s: *const f32,
    t: *const f32,
    u: *const f32,
    f: *mut f32,
    opts: *const Opts,
) -> i32 {
    one_call(
        Kind::Type3,
        m,
        &[x, y, z],
        isign,
        eps,
        &[],
        n,
        &[s, t, u],
        c,
        f,
        opts,
    )
}
