//! The C interface against direct sums, on the CPU and, with
//! `WGPU_FFT_RUN_GPU_TESTS` set, on the GPU.

use std::f64::consts::PI;
use std::ffi::CStr;
use std::ptr;
use std::sync::{Mutex, MutexGuard, PoisonError};

use wgpu_nufft_c::*;

const BACKEND_AUTO: i32 = 0;
const BACKEND_GPU: i32 = 1;
const BACKEND_CPU: i32 = 2;
const PRECISION_F64: i32 = 1;
const PRECISION_DF64: i32 = 2;
const PRECISION_F32: i32 = 3;

fn opts(backend: i32, precision: i32) -> Opts {
    let mut opts = Opts::default();
    unsafe { wgpu_nufft_default_opts(&mut opts) };
    opts.backend = backend;
    opts.precision = precision;
    opts
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(wgpu_nufft_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn check(code: i32) {
    assert_eq!(code, 0, "call failed: {}", last_error());
}

/// Tests that create or select the process's GPU device take turns.
fn device_lock() -> MutexGuard<'static, ()> {
    static DEVICE: Mutex<()> = Mutex::new(());
    DEVICE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn gpu_tests_enabled(name: &str) -> bool {
    let enabled = std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_some();
    if !enabled {
        eprintln!("skipping {name}; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
    }
    enabled
}

fn set_text(field: &mut [u8], text: &str) {
    field.fill(0);
    field[..text.len()].copy_from_slice(text.as_bytes());
}

fn text(field: &[u8]) -> String {
    CStr::from_bytes_until_nul(field)
        .expect("a NUL-terminated field")
        .to_string_lossy()
        .into_owned()
}

fn gpu_name() -> String {
    unsafe { CStr::from_ptr(wgpu_nufft_gpu_name()) }
        .to_string_lossy()
        .into_owned()
}

fn list_adapters() -> Vec<Adapter> {
    let mut count = 0;
    check(unsafe { wgpu_nufft_list_adapters(ptr::null_mut(), 0, &mut count) });
    // Uninitialized entries, which the library only writes.
    let mut adapters = Vec::with_capacity(count as usize);
    let mut listed = 0;
    unsafe {
        check(wgpu_nufft_list_adapters(
            adapters.as_mut_ptr(),
            count,
            &mut listed,
        ));
        adapters.set_len(listed.min(count) as usize);
    }
    adapters
}

/// How `wgpu_nufft_gpu_name` names the device on `adapter`.
fn label(adapter: &Adapter) -> String {
    format!("{} ({})", text(&adapter.name), text(&adapter.backend))
}

/// A 1D type-1 transform with `options` against the direct sum, or the
/// error code.
fn type1_with(options: &Opts) -> Result<(), i32> {
    let mut rng = Rng(0xada);
    let x = rng.values(40, PI);
    let c = rng.values(2 * 40, 1.0);
    let mut f = vec![0.0; 2 * 16];
    let code = unsafe {
        wgpu_nufft1d1(
            40,
            x.as_ptr(),
            c.as_ptr(),
            1,
            1e-6,
            16,
            f.as_mut_ptr(),
            options,
        )
    };
    if code != 0 {
        return Err(code);
    }
    let error = relative_error(&f, &direct_type1(&[&x], &c, 1.0, &[16]));
    assert!(error < 1e-5, "1d1 with a selected adapter: {error:e}");
    Ok(())
}

/// The 1-based position of `adapters[position]` among the adapters for
/// which `same` holds.
fn index_among(adapters: &[Adapter], position: usize, same: impl Fn(&Adapter) -> bool) -> i32 {
    adapters[..=position].iter().filter(|&a| same(a)).count() as i32
}

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn values(&mut self, count: usize, scale: f64) -> Vec<f64> {
        (0..count)
            .map(|_| (2.0 * self.uniform() - 1.0) * scale)
            .collect()
    }
}

/// Centered frequencies of the column-major mode `index`.
fn frequencies(mut index: usize, n_modes: &[usize]) -> Vec<f64> {
    n_modes
        .iter()
        .map(|&n| {
            let k = index % n;
            index /= n;
            k as f64 - (n / 2) as f64
        })
        .collect()
}

/// exp(i * isign * k . x) as (re, im).
fn phase(isign: f64, k: &[f64], x: &[f64]) -> (f64, f64) {
    let angle = isign * k.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
    (angle.cos(), angle.sin())
}

/// Direct type 1: points per axis, interleaved strengths.
fn direct_type1(axes: &[&[f64]], c: &[f64], isign: f64, n_modes: &[usize]) -> Vec<f64> {
    let modes = n_modes.iter().product::<usize>();
    let mut f = vec![0.0; 2 * modes];
    for mode in 0..modes {
        let k = frequencies(mode, n_modes);
        for j in 0..axes[0].len() {
            let x: Vec<f64> = axes.iter().map(|axis| axis[j]).collect();
            let (re, im) = phase(isign, &k, &x);
            f[2 * mode] += c[2 * j] * re - c[2 * j + 1] * im;
            f[2 * mode + 1] += c[2 * j] * im + c[2 * j + 1] * re;
        }
    }
    f
}

fn direct_type2(axes: &[&[f64]], f: &[f64], isign: f64, n_modes: &[usize]) -> Vec<f64> {
    let modes = n_modes.iter().product::<usize>();
    let points = axes[0].len();
    let mut c = vec![0.0; 2 * points];
    for j in 0..points {
        let x: Vec<f64> = axes.iter().map(|axis| axis[j]).collect();
        for mode in 0..modes {
            let (re, im) = phase(isign, &frequencies(mode, n_modes), &x);
            c[2 * j] += f[2 * mode] * re - f[2 * mode + 1] * im;
            c[2 * j + 1] += f[2 * mode] * im + f[2 * mode + 1] * re;
        }
    }
    c
}

fn direct_type3(axes: &[&[f64]], c: &[f64], isign: f64, targets: &[&[f64]]) -> Vec<f64> {
    let count = targets[0].len();
    let mut f = vec![0.0; 2 * count];
    for k in 0..count {
        let s: Vec<f64> = targets.iter().map(|axis| axis[k]).collect();
        for j in 0..axes[0].len() {
            let x: Vec<f64> = axes.iter().map(|axis| axis[j]).collect();
            let (re, im) = phase(isign, &s, &x);
            f[2 * k] += c[2 * j] * re - c[2 * j + 1] * im;
            f[2 * k + 1] += c[2 * j] * im + c[2 * j + 1] * re;
        }
    }
    f
}

fn relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    let difference: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let norm: f64 = expected.iter().map(|b| b * b).sum();
    (difference / norm).sqrt()
}

/// One-call transforms of every type in one to three dimensions.
fn one_call_suite(backend: i32, precision: i32, eps: f64, tolerance: f64) {
    let mut rng = Rng(0x5eed + backend as u64 * 7 + precision as u64);
    let m = 60;
    let x = rng.values(m, PI);
    let y = rng.values(m, PI);
    let z = rng.values(m, PI);
    let c = rng.values(2 * m, 1.0);
    let options = opts(backend, precision);
    let isign = -1;
    unsafe {
        let mut f = vec![0.0; 2 * 12];
        check(wgpu_nufft1d1(
            m as i64,
            x.as_ptr(),
            c.as_ptr(),
            isign,
            eps,
            12,
            f.as_mut_ptr(),
            &options,
        ));
        let error = relative_error(&f, &direct_type1(&[&x], &c, -1.0, &[12]));
        assert!(error < tolerance, "1d1 backend {backend}: {error:e}");

        let modes = rng.values(2 * 8 * 6, 1.0);
        let mut values = vec![0.0; 2 * m];
        check(wgpu_nufft2d2(
            m as i64,
            x.as_ptr(),
            y.as_ptr(),
            values.as_mut_ptr(),
            isign,
            eps,
            8,
            6,
            modes.as_ptr(),
            &options,
        ));
        let error = relative_error(&values, &direct_type2(&[&x, &y], &modes, -1.0, &[8, 6]));
        assert!(error < tolerance, "2d2 backend {backend}: {error:e}");

        let mut f = vec![0.0; 2 * 6 * 5 * 4];
        check(wgpu_nufft3d1(
            m as i64,
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            c.as_ptr(),
            1,
            eps,
            6,
            5,
            4,
            f.as_mut_ptr(),
            &options,
        ));
        let error = relative_error(&f, &direct_type1(&[&x, &y, &z], &c, 1.0, &[6, 5, 4]));
        assert!(error < tolerance, "3d1 backend {backend}: {error:e}");

        let n = 40;
        let s = rng.values(n, 7.0);
        let t = rng.values(n, 5.0);
        let mut f = vec![0.0; 2 * n];
        check(wgpu_nufft2d3(
            m as i64,
            x.as_ptr(),
            y.as_ptr(),
            c.as_ptr(),
            1,
            eps,
            n as i64,
            s.as_ptr(),
            t.as_ptr(),
            f.as_mut_ptr(),
            &options,
        ));
        let error = relative_error(&f, &direct_type3(&[&x, &y], &c, 1.0, &[&s, &t]));
        assert!(error < tolerance, "2d3 backend {backend}: {error:e}");
    }
}

/// A reused plan: two point sets, a batch of three, and FFT mode order.
fn plan_suite(backend: i32, precision: i32, eps: f64, tolerance: f64) -> (i32, i32) {
    let mut rng = Rng(0xbeef + backend as u64);
    let n_modes = [10usize, 7];
    let modes = n_modes.iter().product::<usize>();
    let ntrans = 3;
    let mut options = opts(backend, precision);
    options.mode_order = 1;
    unsafe {
        let mut plan = ptr::null_mut();
        check(wgpu_nufft_makeplan(
            1,
            2,
            [10i64, 7].as_ptr(),
            1,
            ntrans as i64,
            eps,
            &mut plan,
            &options,
        ));
        let backend_used = wgpu_nufft_plan_backend(plan);
        let precision_used = wgpu_nufft_plan_precision(plan);
        for round in 0..2 {
            let m = 30 + 11 * round;
            let x = rng.values(m, 3.0 * PI);
            let y = rng.values(m, PI);
            let c = rng.values(2 * m * ntrans, 1.0);
            // Points through the per-axis call in one round and point-major
            // in the other.
            if round == 0 {
                check(wgpu_nufft_setpts(
                    plan,
                    m as i64,
                    x.as_ptr(),
                    y.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                ));
            } else {
                let points: Vec<f64> = x.iter().zip(&y).flat_map(|(&a, &b)| [a, b]).collect();
                check(wgpu_nufft_setpts_nd(
                    plan,
                    m as i64,
                    points.as_ptr(),
                    0,
                    ptr::null(),
                ));
            }
            let mut f = vec![0.0; 2 * modes * ntrans];
            check(wgpu_nufft_execute(
                plan,
                c.as_ptr().cast_mut(),
                f.as_mut_ptr(),
            ));
            for transform in 0..ntrans {
                let centered = direct_type1(
                    &[&x, &y],
                    &c[2 * m * transform..2 * m * (transform + 1)],
                    1.0,
                    &n_modes,
                );
                // FFT order along both axes: frequency k at index k mod n.
                let mut expected = vec![0.0; 2 * modes];
                for index in 0..modes {
                    let k = frequencies(index, &n_modes);
                    let fft_index =
                        (k[0].rem_euclid(10.0)) as usize + 10 * (k[1].rem_euclid(7.0)) as usize;
                    expected[2 * fft_index] = centered[2 * index];
                    expected[2 * fft_index + 1] = centered[2 * index + 1];
                }
                let actual = &f[2 * modes * transform..2 * modes * (transform + 1)];
                let error = relative_error(actual, &expected);
                assert!(
                    error < tolerance,
                    "plan round {round} transform {transform}: {error:e}"
                );
            }
        }
        wgpu_nufft_destroy(plan);
        (backend_used, precision_used)
    }
}

fn single_precision_suite(backend: i32) {
    let mut rng = Rng(0xf32);
    let m = 50;
    let x: Vec<f32> = rng.values(m, PI).iter().map(|&v| v as f32).collect();
    let c: Vec<f32> = rng.values(2 * m, 1.0).iter().map(|&v| v as f32).collect();
    let options = opts(backend, 0);
    let mut f = vec![0.0f32; 2 * 16];
    unsafe {
        check(wgpu_nufftf1d1(
            m as i64,
            x.as_ptr(),
            c.as_ptr(),
            1,
            1e-5,
            16,
            f.as_mut_ptr(),
            &options,
        ));
        let mut plan = ptr::null_mut();
        check(wgpu_nufftf_makeplan(
            2,
            1,
            [16i64].as_ptr(),
            1,
            1,
            1e-5,
            &mut plan,
            &options,
        ));
        assert_eq!(wgpu_nufftf_plan_precision(plan), PRECISION_F32);
        wgpu_nufftf_destroy(plan);
    }
    let wide_x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let wide_c: Vec<f64> = c.iter().map(|&v| f64::from(v)).collect();
    let wide_f: Vec<f64> = f.iter().map(|&v| f64::from(v)).collect();
    let error = relative_error(&wide_f, &direct_type1(&[&wide_x], &wide_c, 1.0, &[16]));
    assert!(error < 1e-4, "f32 1d1 backend {backend}: {error:e}");
}

#[test]
fn cpu_backend_matches_direct_sums() {
    one_call_suite(BACKEND_CPU, 0, 1e-12, 1e-10);
    let (backend, precision) = plan_suite(BACKEND_CPU, 0, 1e-12, 1e-10);
    assert_eq!((backend, precision), (BACKEND_CPU, PRECISION_F64));
    single_precision_suite(BACKEND_CPU);
    // f32 arithmetic on double arrays.
    one_call_suite(BACKEND_CPU, PRECISION_F32, 1e-5, 1e-4);
}

#[test]
fn errors_are_reported_with_messages() {
    let options = opts(BACKEND_CPU, 0);
    unsafe {
        let mut plan = ptr::null_mut();
        assert_eq!(
            wgpu_nufft_makeplan(4, 1, [8i64].as_ptr(), 1, 1, 1e-6, &mut plan, &options),
            1
        );
        assert!(last_error().contains("type"), "{}", last_error());
        assert!(plan.is_null());
        assert_eq!(
            wgpu_nufft_makeplan(1, 1, [0i64].as_ptr(), 1, 1, 1e-6, &mut plan, &options),
            1
        );
        assert_eq!(
            wgpu_nufft_makeplan(1, 1, [8i64].as_ptr(), 1, 1, 1e-20, &mut plan, &options),
            2
        );
        check(wgpu_nufft_makeplan(
            1,
            1,
            [8i64].as_ptr(),
            1,
            1,
            1e-6,
            &mut plan,
            &options,
        ));
        let mut f = vec![0.0; 16];
        let mut c = vec![0.0; 2];
        assert_eq!(wgpu_nufft_execute(plan, c.as_mut_ptr(), f.as_mut_ptr()), 1);
        assert!(last_error().contains("set the points"), "{}", last_error());
        let far = [10.0f64];
        assert_eq!(
            wgpu_nufft_setpts(
                plan,
                1,
                far.as_ptr(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null()
            ),
            1
        );
        assert!(last_error().contains("3*pi"), "{}", last_error());
        assert_eq!(
            wgpu_nufft_setpts(
                plan,
                1,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null()
            ),
            1
        );
        // The single-precision functions refuse a double plan.
        assert_eq!(
            wgpu_nufftf_execute(plan, ptr::null_mut(), ptr::null_mut()),
            1
        );
        assert!(last_error().contains("array type"), "{}", last_error());
        wgpu_nufft_destroy(plan);
        wgpu_nufft_destroy(ptr::null_mut());
        check(0);
        let version = CStr::from_ptr(wgpu_nufft_version()).to_string_lossy();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }
}

#[test]
fn options_default_to_no_adapter() {
    let mut options = Opts {
        adapter_index: 5,
        ..Opts::default()
    };
    set_text(&mut options.adapter_name, "NVIDIA GeForce RTX 4090");
    unsafe { wgpu_nufft_default_opts(&mut options) };
    assert_eq!(options, Opts::default());
    assert_eq!(options.adapter_index, 0);
    assert!(options.adapter_name.iter().all(|&byte| byte == 0));
    assert!(options.adapter_pci_bus_id.iter().all(|&byte| byte == 0));
    // The layouts the header, Fortran, MATLAB and Julia share.
    assert_eq!(std::mem::size_of::<Opts>(), 320);
    assert_eq!(std::mem::offset_of!(Opts, adapter_index), 24);
    assert_eq!(std::mem::offset_of!(Opts, adapter_name), 28);
    assert_eq!(std::mem::offset_of!(Opts, adapter_pci_bus_id), 284);
    assert_eq!(std::mem::size_of::<Adapter>(), 324);
}

#[test]
fn adapters_are_listed() {
    let adapters = list_adapters();
    for adapter in &adapters {
        eprintln!(
            "adapter: {} {} [{}]{}",
            label(adapter),
            text(&adapter.device_type),
            text(&adapter.pci_bus_id),
            if adapter.is_default == 1 {
                ", default"
            } else {
                ""
            }
        );
        assert!(!text(&adapter.name).is_empty());
        assert!(!text(&adapter.backend).is_empty());
        assert!(!text(&adapter.device_type).is_empty());
    }
    let defaults = adapters.iter().filter(|a| a.is_default == 1).count();
    assert_eq!(defaults, usize::from(!adapters.is_empty()));
    unsafe {
        // A short array takes the first adapters and the full count.
        if let Some(&first) = adapters.first() {
            let mut one = [first];
            one[0].name = [0; ADAPTER_NAME_SIZE];
            let mut count = 0;
            check(wgpu_nufft_list_adapters(one.as_mut_ptr(), 1, &mut count));
            assert_eq!((one[0], count as usize), (first, adapters.len()));
        }
        let mut count = 0;
        assert_eq!(
            wgpu_nufft_list_adapters(ptr::null_mut(), 0, ptr::null_mut()),
            1
        );
        assert_eq!(wgpu_nufft_list_adapters(ptr::null_mut(), 1, &mut count), 1);
        assert_eq!(wgpu_nufft_list_adapters(ptr::null_mut(), -1, &mut count), 1);
        assert!(last_error().contains("capacity"), "{}", last_error());
    }
}

#[test]
fn a_missing_adapter_is_an_error_without_a_cpu_fallback() {
    let _device = device_lock();
    let mut options = opts(BACKEND_AUTO, 0);
    set_text(&mut options.adapter_name, "No Such Adapter 7f3a");
    let makeplan = |options: &Opts| unsafe {
        let mut plan = ptr::null_mut();
        let code = wgpu_nufft_makeplan(1, 1, [8i64].as_ptr(), 1, 1, 1e-6, &mut plan, options);
        let backend = wgpu_nufft_plan_backend(plan);
        wgpu_nufft_destroy(plan);
        (code, backend)
    };
    assert_eq!(makeplan(&options), (3, -1));
    assert!(
        last_error().contains("no GPU adapter matches name \"No Such Adapter 7f3a\""),
        "{}",
        last_error()
    );
    // The one-call functions, and asking again.
    assert_eq!(type1_with(&options), Err(3));
    assert_eq!(makeplan(&options), (3, -1));
    assert!(last_error().contains("7f3a"), "{}", last_error());
    let mut beyond = opts(BACKEND_GPU, 0);
    beyond.adapter_index = 10_000;
    assert_eq!(makeplan(&beyond).0, 3);
    assert!(last_error().contains("index 10000"), "{}", last_error());

    // CPU plans ignore the selection, but not a malformed one.
    options.backend = BACKEND_CPU;
    assert_eq!(makeplan(&options), (0, BACKEND_CPU));
    let mut unterminated = options;
    unterminated.adapter_name = [b'a'; ADAPTER_NAME_SIZE];
    assert_eq!(makeplan(&unterminated).0, 1);
    assert!(last_error().contains("adapter_name"), "{}", last_error());
    let mut address = options;
    set_text(&mut address.adapter_pci_bus_id, "slot 3");
    assert_eq!(makeplan(&address).0, 1);
    assert!(last_error().contains("PCI address"), "{}", last_error());
    let mut negative = options;
    negative.adapter_index = -1;
    assert_eq!(makeplan(&negative).0, 1);
    assert!(last_error().contains("adapter_index"), "{}", last_error());
}

#[test]
fn adapters_are_selected_by_name_index_and_address() {
    if !gpu_tests_enabled("the GPU adapter-selection test") {
        return;
    }
    let _device = device_lock();
    wgpu_nufft_shutdown();
    let adapters = list_adapters();
    let default = adapters
        .iter()
        .position(|a| a.is_default == 1)
        .expect("no GPU adapter");
    let chosen = &adapters[default];
    let name = text(&chosen.name);

    // Without a selection, the device is on the adapter the list marks.
    assert_eq!(gpu_name(), label(chosen));
    type1_with(&opts(BACKEND_GPU, 0)).unwrap();
    wgpu_nufft_shutdown();

    // By name, regardless of case; an index tells identical cards apart.
    let mut by_name = opts(BACKEND_AUTO, 0);
    set_text(&mut by_name.adapter_name, &name.to_uppercase());
    by_name.adapter_index = index_among(&adapters, default, |a| text(&a.name) == name);
    type1_with(&by_name).unwrap();
    assert_eq!(gpu_name(), label(chosen));
    // The device serves plans without a selection, and other selections of
    // its adapter: its position alone, and its address.
    type1_with(&opts(BACKEND_GPU, 0)).unwrap();
    let mut by_index = opts(BACKEND_GPU, 0);
    by_index.adapter_index = default as i32 + 1;
    type1_with(&by_index).unwrap();
    let address = text(&chosen.pci_bus_id);
    if !address.is_empty() {
        let mut by_address = opts(BACKEND_GPU, 0);
        set_text(&mut by_address.adapter_pci_bus_id, &address);
        by_address.adapter_index =
            index_among(&adapters, default, |a| text(&a.pci_bus_id) == address);
        type1_with(&by_address).unwrap();
    }

    // Another adapter waits for the device to go.
    if let Some(other) = (0..adapters.len()).find(|&position| position != default) {
        let mut by_other = opts(BACKEND_GPU, 0);
        by_other.adapter_index = other as i32 + 1;
        assert_eq!(type1_with(&by_other), Err(1));
        assert!(
            last_error().contains("wgpu_nufft_shutdown"),
            "{}",
            last_error()
        );
        wgpu_nufft_shutdown();
        match type1_with(&by_other) {
            Ok(()) => assert_eq!(gpu_name(), label(&adapters[other])),
            Err(code) => {
                assert_ne!(code, 1, "{}", last_error());
                eprintln!(
                    "{} serves no plan: {}",
                    label(&adapters[other]),
                    last_error()
                );
            }
        }
    }
    wgpu_nufft_shutdown();
}

#[test]
fn gpu_backend_matches_direct_sums() {
    if !gpu_tests_enabled("the GPU C-interface test") {
        return;
    }
    let _device = device_lock();
    let name = gpu_name();
    assert!(!name.is_empty(), "no GPU");
    eprintln!("GPU: {name}");
    // The best double arithmetic of this device, and Df64 in any case.
    one_call_suite(BACKEND_GPU, 0, 1e-9, 1e-8);
    let (backend, precision) = plan_suite(BACKEND_GPU, 0, 1e-9, 1e-8);
    assert_eq!(backend, BACKEND_GPU);
    assert!(precision == PRECISION_F64 || precision == PRECISION_DF64);
    one_call_suite(BACKEND_GPU, PRECISION_DF64, 1e-9, 1e-8);
    single_precision_suite(BACKEND_GPU);
    one_call_suite(BACKEND_GPU, PRECISION_F32, 1e-5, 1e-4);
    // Repeated one-call transforms reuse their plans.
    let start = std::time::Instant::now();
    one_call_suite(BACKEND_GPU, 0, 1e-9, 1e-8);
    eprintln!("repeated one-call suite: {:?}", start.elapsed());
    wgpu_nufft_shutdown();
}
