#![cfg(feature = "cpu")]

//! CPU type-3 plans against the direct `f64` reference, bitwise
//! repeatability across thread counts, and the interval contract.

use wgpu_nufft::{
    reference_type3_f64, Complex64, CpuNufftType3Plan, CpuNufftType3Points, FftPrecision,
    NufftError, NufftInterval, NufftType3Config,
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Points inside `bounds`, point-major.
    fn points(&mut self, count: usize, bounds: &[NufftInterval]) -> Vec<f64> {
        (0..count)
            .flat_map(|_| bounds.to_vec())
            .map(|interval| interval.lower() + self.next() * (interval.upper() - interval.lower()))
            .collect()
    }
}

fn split(value: f64) -> [f32; 2] {
    let hi = value as f32;
    [hi, (value - f64::from(hi)) as f32]
}

/// Rounds values to what a plan of `precision` stores.
fn stored(precision: FftPrecision, values: &[f64]) -> Vec<f64> {
    match precision {
        FftPrecision::F32 => values.iter().map(|&v| f64::from(v as f32)).collect(),
        FftPrecision::F64 => values.to_vec(),
        FftPrecision::Df64 => values
            .iter()
            .map(|&v| {
                let [hi, lo] = split(v);
                f64::from(hi) + f64::from(lo)
            })
            .collect(),
    }
}

fn execute(
    plan: &CpuNufftType3Plan,
    sources: &[f64],
    strengths: &[f64],
    targets: &[f64],
) -> Result<Vec<Complex64>, NufftError> {
    let target_count = targets.len() / plan.config().dimensions();
    let output_len = plan.required_output_len(target_count)?;
    let words: Vec<f64> = match plan.config().precision() {
        FftPrecision::F32 => {
            let narrow = |values: &[f64]| values.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            plan.execute(
                &narrow(sources),
                &narrow(strengths),
                &narrow(targets),
                &mut output,
            )?;
            output.into_iter().map(f64::from).collect()
        }
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            plan.execute_f64(sources, strengths, targets, &mut output)?;
            output
        }
        FftPrecision::Df64 => {
            let pairs = |values: &[f64]| values.iter().flat_map(|&v| split(v)).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            plan.execute(
                &pairs(sources),
                &pairs(strengths),
                &pairs(targets),
                &mut output,
            )?;
            output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[hi, lo]| f64::from(hi) + f64::from(lo))
                .collect()
        }
    };
    Ok(words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect())
}

fn pairs(values: &[f64]) -> Vec<f32> {
    values.iter().flat_map(|&v| split(v)).collect()
}

fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

/// Prepares `f64` sources and targets with `plan`, in its storage precision.
fn prepare(
    plan: &CpuNufftType3Plan,
    sources: &[f64],
    targets: &[f64],
) -> Result<CpuNufftType3Points, NufftError> {
    match plan.config().precision() {
        FftPrecision::F32 => plan.prepare_points(&narrow(sources), &narrow(targets)),
        FftPrecision::F64 => plan.prepare_points_f64(sources, targets),
        FftPrecision::Df64 => plan.prepare_points(&pairs(sources), &pairs(targets)),
    }
}

/// Runs `plan` on prepared points, returning its output words as `f64`.
fn execute_prepared(
    plan: &CpuNufftType3Plan,
    points: &CpuNufftType3Points,
    strengths: &[f64],
) -> Result<Vec<f64>, NufftError> {
    let output_len = plan.required_output_len(points.target_count())?;
    match plan.config().precision() {
        FftPrecision::F32 => {
            let mut output = vec![0.0f32; output_len];
            plan.execute_prepared(points, &narrow(strengths), &mut output)?;
            Ok(output.into_iter().map(f64::from).collect())
        }
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            plan.execute_prepared_f64(points, strengths, &mut output)?;
            Ok(output)
        }
        FftPrecision::Df64 => {
            let mut output = vec![0.0f32; output_len];
            plan.execute_prepared(points, &pairs(strengths), &mut output)?;
            Ok(output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[hi, lo]| f64::from(hi) + f64::from(lo))
                .collect())
        }
    }
}

fn relative_l2(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut norm) = (0.0, 0.0);
    for (a, e) in actual.iter().zip(expected) {
        error += (a.re - e.re).powi(2) + (a.im - e.im).powi(2);
        norm += e.re.powi(2) + e.im.powi(2);
    }
    (error / norm).sqrt()
}

fn check(config: &NufftType3Config, counts: (usize, usize), tolerance: f64, seed: u64) {
    let precision = config.precision();
    let mut rng = Lcg(seed);
    let sources = stored(precision, &rng.points(counts.0, config.source_bounds()));
    let targets = stored(precision, &rng.points(counts.1, config.target_bounds()));
    let strengths = stored(
        precision,
        &(0..2 * counts.0 * config.batch())
            .map(|_| rng.next() - 0.5)
            .collect::<Vec<_>>(),
    );
    let complex = strengths
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect::<Vec<_>>();
    let plan = CpuNufftType3Plan::new(config.clone()).unwrap();
    let actual = execute(&plan, &sources, &strengths, &targets).unwrap();
    let expected = reference_type3_f64(config, &sources, &targets, &complex).unwrap();
    let error = relative_l2(&actual, &expected);
    assert!(
        error <= tolerance,
        "{} dimensions {precision:?} isign={} batch={}: error {error:.3e} > {tolerance:.1e}",
        config.dimensions(),
        config.isign(),
        config.batch()
    );
}

/// Source and target intervals with off-center, unequal widths per axis.
/// Four-dimensional plans get narrower intervals, since the outer grid grows
/// with the product of every axis's space-bandwidth product.
fn bounds(dimensions: usize) -> (Vec<NufftInterval>, Vec<NufftInterval>) {
    let (sources, targets) = if dimensions < 4 {
        (
            [(-1.5, 2.5), (0.5, 3.0), (-4.0, -1.0), (0.0, 0.0)],
            [(-30.0, 50.0), (-10.0, 25.0), (5.0, 45.0), (0.0, 0.0)],
        )
    } else {
        (
            [(-0.5, 0.7), (0.2, 1.0), (-1.1, -0.4), (-0.3, 0.9)],
            [(-6.0, 4.0), (-3.0, 5.0), (1.0, 7.0), (-5.0, 2.0)],
        )
    };
    (
        sources[..dimensions]
            .iter()
            .map(|&(lower, upper)| NufftInterval::new(lower, upper))
            .collect(),
        targets[..dimensions]
            .iter()
            .map(|&(lower, upper)| NufftInterval::new(lower, upper))
            .collect(),
    )
}

#[test]
fn cpu_type3_matches_the_reference_in_one_to_four_dimensions() {
    let mut seed = 1;
    for (dimensions, counts) in [
        (1, (300, 250)),
        (2, (200, 150)),
        (3, (150, 120)),
        (4, (60, 50)),
    ] {
        let (source_bounds, target_bounds) = bounds(dimensions);
        for (precision, eps, factor) in [
            (FftPrecision::F32, 1.0e-5, 20.0),
            (FftPrecision::F64, 1.0e-10, 20.0),
            (FftPrecision::Df64, 1.0e-10, 20.0),
        ] {
            for isign in [1, -1] {
                for batch in [1, 2] {
                    let config =
                        NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), eps)
                            .with_isign(isign)
                            .with_batch(batch)
                            .with_precision(precision);
                    check(&config, counts, factor * eps, seed);
                    seed += 1;
                }
            }
        }
    }
}

#[test]
fn cpu_type3_handles_points_on_the_interval_ends_and_degenerate_intervals() {
    // Zero-width source and target intervals, and points on every end.
    let config = NufftType3Config::new(
        vec![
            NufftInterval::new(0.75, 0.75),
            NufftInterval::new(-2.0, 2.0),
        ],
        vec![
            NufftInterval::new(-20.0, 20.0),
            NufftInterval::new(3.0, 3.0),
        ],
        1.0e-9,
    )
    .with_precision(FftPrecision::F64);
    let sources = [0.75, -2.0, 0.75, 2.0, 0.75, 0.3];
    let targets = [-20.0, 3.0, 20.0, 3.0, 1.5, 3.0];
    let strengths = [1.0, 0.5, -0.25, 0.75, 0.5, -1.0];
    let complex = strengths
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect::<Vec<_>>();
    let plan = CpuNufftType3Plan::new(config.clone()).unwrap();
    let actual = execute(&plan, &sources, &strengths, &targets).unwrap();
    let expected = reference_type3_f64(&config, &sources, &targets, &complex).unwrap();
    let error = relative_l2(&actual, &expected);
    assert!(error <= 1.0e-8, "edge points: {error:.3e}");
}

#[test]
fn cpu_type3_results_do_not_depend_on_the_thread_count() {
    for dimensions in [1, 2, 3] {
        let (source_bounds, target_bounds) = bounds(dimensions);
        let config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-6)
            .with_batch(2);
        let mut rng = Lcg(9);
        // Enough sources for the dense spreading path in 1D.
        let (source_count, target_count) = (40_000, 5_000);
        let sources = rng.points(source_count, &source_bounds);
        let targets = rng.points(target_count, &target_bounds);
        let strengths = (0..4 * source_count)
            .map(|_| rng.next() - 0.5)
            .collect::<Vec<_>>();
        let run = |threads| {
            let plan = CpuNufftType3Plan::new(config.clone())
                .unwrap()
                .with_threads(threads);
            execute(&plan, &sources, &strengths, &targets).unwrap()
        };
        let single = run(1);
        for threads in [2, 7] {
            let parallel = run(threads);
            assert!(
                single.iter().zip(&parallel).all(|(a, b)| {
                    a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                }),
                "{dimensions}D: {threads} threads changed the result"
            );
        }
    }
}

#[test]
fn cpu_type3_rejects_points_outside_the_intervals() {
    let config = NufftType3Config::new(
        vec![NufftInterval::new(-1.0, 1.0)],
        vec![NufftInterval::new(-5.0, 5.0)],
        1.0e-6,
    );
    let plan = CpuNufftType3Plan::new(config).unwrap();
    let mut output = vec![0.0f32; plan.required_output_len(1).unwrap()];
    assert!(matches!(
        plan.execute(&[0.5, 1.5], &[1.0; 4], &[0.0], &mut output),
        Err(NufftError::Type3CoordinateOutOfBounds { point: 1, .. })
    ));
    assert!(matches!(
        plan.execute(&[0.5], &[1.0; 2], &[f32::NAN], &mut output),
        Err(NufftError::Type3NonFiniteCoordinate { .. })
    ));
    assert!(matches!(
        plan.execute(&[0.5], &[1.0; 4], &[0.0], &mut output),
        Err(NufftError::InputLength { .. })
    ));
    assert!(matches!(
        plan.execute_f64(&[0.5], &[1.0; 2], &[0.0], &mut [0.0; 2]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));
    // No sources give zeros; no targets give nothing.
    let mut zeros = vec![1.0f32; 2];
    plan.execute(&[], &[], &[2.0], &mut zeros).unwrap();
    assert_eq!(zeros, [0.0, 0.0]);
    plan.execute(&[0.5], &[1.0, 0.0], &[], &mut []).unwrap();
}

#[test]
fn cpu_type3_prepared_points_repeat_one_shot_results_bitwise() {
    for dimensions in [1, 2, 3] {
        let (source_bounds, target_bounds) = bounds(dimensions);
        let mut rng = Lcg(17);
        let (source_count, target_count) = (6_000, 4_000);
        let sources = rng.points(source_count, &source_bounds);
        let targets = rng.points(target_count, &target_bounds);
        for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
            let config =
                NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-7)
                    .with_isign(-1)
                    .with_precision(precision);
            let (sources, targets) = (stored(precision, &sources), stored(precision, &targets));
            // Prepared by a single-transform plan, used by a batched one.
            let preparer = CpuNufftType3Plan::new(config.clone())
                .unwrap()
                .with_threads(1);
            let points = prepare(&preparer, &sources, &targets).unwrap();
            assert_eq!(
                (
                    points.source_count(),
                    points.target_count(),
                    points.dimensions()
                ),
                (source_count, target_count, dimensions)
            );
            let plan = CpuNufftType3Plan::new(config.with_batch(2))
                .unwrap()
                .with_threads(5);
            let strengths = stored(
                precision,
                &(0..4 * source_count)
                    .map(|_| rng.next() - 0.5)
                    .collect::<Vec<_>>(),
            );
            let expected = execute(&plan, &sources, &strengths, &targets)
                .unwrap()
                .iter()
                .flat_map(|value| [value.re, value.im])
                .collect::<Vec<_>>();
            for _ in 0..2 {
                let actual = execute_prepared(&plan, &points, &strengths).unwrap();
                assert!(
                    actual.len() == expected.len()
                        && actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{dimensions}D {precision:?}: prepared points changed the result"
                );
            }
        }
    }
}

#[test]
fn cpu_type3_prepared_points_fit_only_matching_plans() {
    let config = |half: f64| {
        NufftType3Config::new(
            vec![NufftInterval::new(-1.0, 1.0)],
            vec![NufftInterval::new(-half, half)],
            1.0e-6,
        )
    };
    let plan = CpuNufftType3Plan::new(config(5.0)).unwrap();
    let points = plan.prepare_points(&[0.5, -0.25], &[1.0]).unwrap();
    let mut output = vec![0.0f32; plan.required_output_len(1).unwrap()];
    plan.execute_prepared(&points, &[1.0, 0.0, 0.5, 0.5], &mut output)
        .unwrap();
    assert!(matches!(
        plan.execute_prepared(&points, &[1.0, 0.0], &mut output),
        Err(NufftError::InputLength { .. })
    ));
    for other in [config(6.0), config(5.0).with_isign(-1)] {
        let other = CpuNufftType3Plan::new(other).unwrap();
        assert!(matches!(
            other.execute_prepared(&points, &[1.0, 0.0, 0.5, 0.5], &mut output),
            Err(NufftError::PointSetMismatch { .. })
        ));
    }
    assert!(matches!(
        plan.prepare_points(&[1.5], &[0.0]),
        Err(NufftError::Type3CoordinateOutOfBounds { .. })
    ));
    assert!(matches!(
        plan.prepare_points_f64(&[0.5], &[0.0]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));
}

#[test]
fn cpu_plans_can_be_shared_between_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<wgpu_nufft::CpuNufftPlan>();
    shareable::<CpuNufftType3Plan>();
    shareable::<CpuNufftType3Points>();
}
