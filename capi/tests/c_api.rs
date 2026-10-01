//! The C interface against direct sums, on the CPU and, with
//! `WGPU_FFT_RUN_GPU_TESTS` set, on the GPU.

use std::f64::consts::PI;
use std::ffi::CStr;
use std::ptr;

use wgpu_nufft_c::*;

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
fn gpu_backend_matches_direct_sums() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping the GPU C-interface test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    let name = unsafe { CStr::from_ptr(wgpu_nufft_gpu_name()) }
        .to_string_lossy()
        .into_owned();
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
