//! Wall-clock timings of the CPU plans.
//!
//! ```text
//! cargo bench --bench nufft_cpu_bench -- [--threads N] [--runs N] [--f64]
//! ```

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use wgpu_nufft::{CpuNufftPlan, FftPrecision, NufftConfig};

struct Options {
    threads: Option<usize>,
    runs: usize,
    precision: FftPrecision,
}

fn main() {
    let options = parse_options();
    let cases: [(&[usize], usize); 6] = [
        (&[1_000_000], 10_000_000),
        (&[10_000], 1_000_000),
        (&[1024, 1024], 10_000_000),
        (&[256, 256], 1_000_000),
        (&[128, 128, 128], 10_000_000),
        (&[32, 32, 32], 1_000_000),
    ];
    for (shape, point_count) in cases {
        for kind in ["type1", "type2"] {
            let config = NufftConfig::new(shape, 1.0e-6).with_precision(options.precision);
            let plan = match kind {
                "type1" => CpuNufftPlan::type1(config),
                _ => CpuNufftPlan::type2(config),
            }
            .expect("plan");
            let plan = match options.threads {
                Some(threads) => plan.with_threads(threads),
                None => plan,
            };
            let median = time_plan(&plan, point_count, options.runs);
            println!(
                "CPU_NUFFT {kind} modes={shape:?} points={point_count} {:?} threads={} median={:.1} ms ({:.1} Mpts/s)",
                options.precision,
                plan.threads(),
                median.as_secs_f64() * 1.0e3,
                point_count as f64 / median.as_secs_f64() / 1.0e6,
            );
        }
    }
}

fn time_plan(plan: &CpuNufftPlan, point_count: usize, runs: usize) -> Duration {
    let dimensions = plan.config().dimensions();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let coordinates = (0..point_count * dimensions)
        .map(|_| (2.0 * next() - 1.0) * PI)
        .collect::<Vec<_>>();
    let input_len = plan.required_input_len(point_count).expect("input length");
    let input = (0..input_len).map(|_| next() - 0.5).collect::<Vec<_>>();
    let output_len = plan
        .required_output_len(point_count)
        .expect("output length");
    let mut times = Vec::with_capacity(runs);
    match plan.config().precision() {
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            for _ in 0..runs {
                let start = Instant::now();
                plan.execute_f64(&coordinates, &input, &mut output)
                    .expect("execute");
                times.push(start.elapsed());
            }
        }
        _ => {
            let points = coordinates.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let input = input.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            for _ in 0..runs {
                let start = Instant::now();
                plan.execute(&points, &input, &mut output).expect("execute");
                times.push(start.elapsed());
            }
        }
    }
    times.sort();
    times[times.len() / 2]
}

fn parse_options() -> Options {
    let mut options = Options {
        threads: None,
        runs: 3,
        precision: FftPrecision::F32,
    };
    // Cargo passes `--bench` to harness-free benchmarks.
    let mut arguments = std::env::args().skip(1).filter(|arg| arg != "--bench");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--threads" => options.threads = arguments.next().and_then(|v| v.parse().ok()),
            "--runs" => {
                options.runs = arguments
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|&runs| runs > 0)
                    .expect("--runs takes a positive count");
            }
            "--f64" => options.precision = FftPrecision::F64,
            "--help" | "-h" => {
                println!(
                    "usage: cargo bench --bench nufft_cpu_bench -- [--threads N] [--runs N] [--f64]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    options
}
