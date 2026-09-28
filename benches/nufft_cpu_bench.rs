//! Wall-clock timings of the CPU plans: one-shot executions, point
//! preparation, and executions on prepared points.
//!
//! ```text
//! cargo bench --bench nufft_cpu_bench -- [--threads N] [--runs N] [--f64] [--only TEXT]
//! ```

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use wgpu_nufft::{
    CpuNufftPlan, CpuNufftType3Plan, FftPrecision, NufftConfig, NufftInterval, NufftType3Config,
};

struct Options {
    threads: Option<usize>,
    runs: usize,
    precision: FftPrecision,
    /// Runs only the cases whose label contains this text.
    only: Option<String>,
}

impl Options {
    fn runs_case(&self, label: &str) -> bool {
        self.only
            .as_ref()
            .is_none_or(|text| label.contains(text.as_str()))
    }
}

/// Median times of one case.
struct Timings {
    execute: Duration,
    prepare: Duration,
    prepared: Duration,
}

impl Timings {
    fn report(&self, point_count: usize) -> String {
        let ms = |time: Duration| time.as_secs_f64() * 1.0e3;
        format!(
            "execute={:.1} ms ({:.1} Mpts/s) prepare={:.1} ms prepared={:.1} ms",
            ms(self.execute),
            point_count as f64 / self.execute.as_secs_f64() / 1.0e6,
            ms(self.prepare),
            ms(self.prepared),
        )
    }
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
            if !options.runs_case(&format!("{kind} modes={shape:?}")) {
                continue;
            }
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
            let timings = time_plan(&plan, point_count, options.runs);
            println!(
                "CPU_NUFFT {kind} modes={shape:?} points={point_count} {:?} threads={} {}",
                options.precision,
                plan.threads(),
                timings.report(point_count),
            );
        }
    }
    // Type 3 from [-pi, pi] per axis to frequencies within `half_band`.
    let type3_cases: [(usize, f64, usize); 3] = [
        (1, 500_000.0, 10_000_000),
        (2, 512.0, 10_000_000),
        (3, 32.0, 10_000_000),
    ];
    for (dimensions, half_band, point_count) in type3_cases {
        if !options.runs_case(&format!("type3 dims={dimensions}")) {
            continue;
        }
        let config = NufftType3Config::new(
            vec![NufftInterval::new(-PI, PI); dimensions],
            vec![NufftInterval::new(-half_band, half_band); dimensions],
            1.0e-6,
        )
        .with_precision(options.precision);
        let plan = CpuNufftType3Plan::new(config).expect("type-3 plan");
        let plan = match options.threads {
            Some(threads) => plan.with_threads(threads),
            None => plan,
        };
        let timings = time_type3(&plan, point_count, half_band, options.runs);
        println!(
            "CPU_NUFFT type3 dims={dimensions} band={half_band} sources=targets={point_count} outer={:?} {:?} threads={} {}",
            plan.outer_grid_shape(),
            options.precision,
            plan.threads(),
            timings.report(point_count),
        );
    }
}

/// Median of `runs` timings of `work`.
fn median(runs: usize, mut work: impl FnMut()) -> Duration {
    let mut times = (0..runs)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed()
        })
        .collect::<Vec<_>>();
    times.sort();
    times[times.len() / 2]
}

fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

fn time_type3(plan: &CpuNufftType3Plan, count: usize, half_band: f64, runs: usize) -> Timings {
    let dimensions = plan.config().dimensions();
    let mut next = uniform(0x2545_f491_4f6c_dd1d);
    let sources = (0..count * dimensions)
        .map(|_| (2.0 * next() - 1.0) * PI)
        .collect::<Vec<_>>();
    let targets = (0..count * dimensions)
        .map(|_| (2.0 * next() - 1.0) * half_band)
        .collect::<Vec<_>>();
    let strengths_len = plan.required_strengths_len(count).expect("strength length");
    let strengths = (0..strengths_len).map(|_| next() - 0.5).collect::<Vec<_>>();
    let output_len = plan.required_output_len(count).expect("output length");
    match plan.config().precision() {
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            let points = plan
                .prepare_points_f64(&sources, &targets)
                .expect("prepare");
            Timings {
                execute: median(runs, || {
                    plan.execute_f64(&sources, &strengths, &targets, &mut output)
                        .expect("execute")
                }),
                prepare: median(runs, || {
                    plan.prepare_points_f64(&sources, &targets)
                        .expect("prepare");
                }),
                prepared: median(runs, || {
                    plan.execute_prepared_f64(&points, &strengths, &mut output)
                        .expect("execute prepared")
                }),
            }
        }
        _ => {
            let (sources, strengths, targets) =
                (narrow(&sources), narrow(&strengths), narrow(&targets));
            let mut output = vec![0.0f32; output_len];
            let points = plan.prepare_points(&sources, &targets).expect("prepare");
            Timings {
                execute: median(runs, || {
                    plan.execute(&sources, &strengths, &targets, &mut output)
                        .expect("execute")
                }),
                prepare: median(runs, || {
                    plan.prepare_points(&sources, &targets).expect("prepare");
                }),
                prepared: median(runs, || {
                    plan.execute_prepared(&points, &strengths, &mut output)
                        .expect("execute prepared")
                }),
            }
        }
    }
}

fn time_plan(plan: &CpuNufftPlan, point_count: usize, runs: usize) -> Timings {
    let dimensions = plan.config().dimensions();
    let mut next = uniform(0x9e37_79b9_7f4a_7c15);
    let coordinates = (0..point_count * dimensions)
        .map(|_| (2.0 * next() - 1.0) * PI)
        .collect::<Vec<_>>();
    let input_len = plan.required_input_len(point_count).expect("input length");
    let input = (0..input_len).map(|_| next() - 0.5).collect::<Vec<_>>();
    let output_len = plan
        .required_output_len(point_count)
        .expect("output length");
    match plan.config().precision() {
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            let points = plan.prepare_points_f64(&coordinates).expect("prepare");
            Timings {
                execute: median(runs, || {
                    plan.execute_f64(&coordinates, &input, &mut output)
                        .expect("execute")
                }),
                prepare: median(runs, || {
                    plan.prepare_points_f64(&coordinates).expect("prepare");
                }),
                prepared: median(runs, || {
                    plan.execute_prepared_f64(&points, &input, &mut output)
                        .expect("execute prepared")
                }),
            }
        }
        _ => {
            let (points, input) = (narrow(&coordinates), narrow(&input));
            let mut output = vec![0.0f32; output_len];
            let prepared = plan.prepare_points(&points).expect("prepare");
            Timings {
                execute: median(runs, || {
                    plan.execute(&points, &input, &mut output).expect("execute")
                }),
                prepare: median(runs, || {
                    plan.prepare_points(&points).expect("prepare");
                }),
                prepared: median(runs, || {
                    plan.execute_prepared(&prepared, &input, &mut output)
                        .expect("execute prepared")
                }),
            }
        }
    }
}

fn parse_options() -> Options {
    let mut options = Options {
        threads: None,
        runs: 3,
        precision: FftPrecision::F32,
        only: None,
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
            "--only" => options.only = arguments.next(),
            "--help" | "-h" => {
                println!(
                    "usage: cargo bench --bench nufft_cpu_bench -- [--threads N] [--runs N] [--f64] [--only TEXT]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    options
}
