#![cfg(feature = "cpu")]

//! CPU plans against the direct `f64` reference transforms, bitwise
//! repeatability across thread counts, and the input contract.

use std::f64::consts::PI;

use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, Complex64, CpuNufftPlan, CpuNufftPoints,
    FftPrecision, ModeOrder, NufftConfig, NufftError, NufftKind,
};

/// Deterministic uniform values in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Coordinates over the whole `[-3*pi, 3*pi]` contract.
    fn coordinates(&mut self, count: usize) -> Vec<f64> {
        (0..count)
            .map(|_| (2.0 * self.next() - 1.0) * 3.0 * PI)
            .collect()
    }

    fn values(&mut self, count: usize) -> Vec<Complex64> {
        (0..count)
            .map(|_| Complex64::new(self.next() - 0.5, self.next() - 0.5))
            .collect()
    }
}

fn split(value: f64) -> [f32; 2] {
    let hi = value as f32;
    [hi, (value - f64::from(hi)) as f32]
}

/// Rounds data to what a plan of `precision` stores, so that the reference
/// sees exactly the plan's inputs.
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

fn interleave(values: &[Complex64]) -> Vec<f64> {
    values.iter().flat_map(|v| [v.re, v.im]).collect()
}

fn deinterleave(words: &[f64]) -> Vec<Complex64> {
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect()
}

/// Runs `plan` on `f64` data converted to its storage precision.
fn execute(
    plan: &CpuNufftPlan,
    coordinates: &[f64],
    input: &[f64],
) -> Result<Vec<f64>, NufftError> {
    let point_count = coordinates.len() / plan.config().dimensions();
    let output_len = plan.required_output_len(point_count)?;
    match plan.config().precision() {
        FftPrecision::F32 => {
            let points = coordinates.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let input = input.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            plan.execute(&points, &input, &mut output)?;
            Ok(output.into_iter().map(f64::from).collect())
        }
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            plan.execute_f64(coordinates, input, &mut output)?;
            Ok(output)
        }
        FftPrecision::Df64 => {
            let points = coordinates
                .iter()
                .flat_map(|&v| split(v))
                .collect::<Vec<_>>();
            let input = input.iter().flat_map(|&v| split(v)).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            plan.execute(&points, &input, &mut output)?;
            Ok(output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[hi, lo]| f64::from(hi) + f64::from(lo))
                .collect())
        }
    }
}

/// Prepares `f64` coordinates with `plan`, converted to its storage
/// precision.
fn prepare(plan: &CpuNufftPlan, coordinates: &[f64]) -> Result<CpuNufftPoints, NufftError> {
    match plan.config().precision() {
        FftPrecision::F32 => {
            let points = coordinates.iter().map(|&v| v as f32).collect::<Vec<_>>();
            plan.prepare_points(&points)
        }
        FftPrecision::F64 => plan.prepare_points_f64(coordinates),
        FftPrecision::Df64 => {
            let points = coordinates
                .iter()
                .flat_map(|&v| split(v))
                .collect::<Vec<_>>();
            plan.prepare_points(&points)
        }
    }
}

/// Runs `plan` on prepared `points` with `f64` data converted to its storage
/// precision.
fn execute_prepared(
    plan: &CpuNufftPlan,
    points: &CpuNufftPoints,
    input: &[f64],
) -> Result<Vec<f64>, NufftError> {
    let output_len = plan.required_output_len(points.len())?;
    match plan.config().precision() {
        FftPrecision::F32 => {
            let input = input.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            plan.execute_prepared(points, &input, &mut output)?;
            Ok(output.into_iter().map(f64::from).collect())
        }
        FftPrecision::F64 => {
            let mut output = vec![0.0f64; output_len];
            plan.execute_prepared_f64(points, input, &mut output)?;
            Ok(output)
        }
        FftPrecision::Df64 => {
            let input = input.iter().flat_map(|&v| split(v)).collect::<Vec<_>>();
            let mut output = vec![0.0f32; output_len];
            plan.execute_prepared(points, &input, &mut output)?;
            Ok(output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[hi, lo]| f64::from(hi) + f64::from(lo))
                .collect())
        }
    }
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
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

/// Compares both transform types of `config` with the reference.
fn check_against_reference(config: &NufftConfig, point_count: usize, tolerance: f64, seed: u64) {
    let precision = config.precision();
    let mut rng = Lcg(seed);
    let coordinates = stored(
        precision,
        &rng.coordinates(point_count * config.dimensions()),
    );
    let label = format!(
        "{:?} {:?} isign={} {:?} batch={}",
        config.n_modes(),
        precision,
        config.isign(),
        config.mode_order(),
        config.batch()
    );

    let strengths = stored(
        precision,
        &interleave(&rng.values(point_count * config.batch())),
    );
    let type1 = CpuNufftPlan::type1(config.clone()).unwrap();
    let modes = deinterleave(&execute(&type1, &coordinates, &strengths).unwrap());
    let expected = reference_type1_f64(config, &coordinates, &deinterleave(&strengths)).unwrap();
    let error = relative_l2(&modes, &expected);
    assert!(
        error <= tolerance,
        "type 1 {label}: error {error:.3e} > {tolerance:.1e}"
    );

    let mode_count = config.n_modes().iter().product::<usize>();
    let coefficients = stored(
        precision,
        &interleave(&rng.values(mode_count * config.batch())),
    );
    let type2 = CpuNufftPlan::type2(config.clone()).unwrap();
    let values = deinterleave(&execute(&type2, &coordinates, &coefficients).unwrap());
    let expected = reference_type2_f64(config, &coordinates, &deinterleave(&coefficients)).unwrap();
    let error = relative_l2(&values, &expected);
    assert!(
        error <= tolerance,
        "type 2 {label}: error {error:.3e} > {tolerance:.1e}"
    );
}

#[test]
fn cpu_plans_match_the_reference_in_one_to_five_dimensions() {
    let shapes: [(&[usize], usize); 6] = [
        (&[37], 150),
        (&[64], 200),
        (&[12, 9], 150),
        (&[8, 6, 5], 120),
        (&[4, 5, 3, 4], 60),
        (&[3, 4, 2, 3, 2], 40),
    ];
    let mut seed = 1;
    for (shape, point_count) in shapes {
        for (precision, eps, factor) in [
            (FftPrecision::F32, 1.0e-5, 10.0),
            (FftPrecision::F64, 1.0e-10, 10.0),
            (FftPrecision::Df64, 1.0e-10, 10.0),
        ] {
            for isign in [1, -1] {
                for mode_order in [ModeOrder::Centered, ModeOrder::Fft] {
                    for batch in [1, 2] {
                        let config = NufftConfig::new(shape, eps)
                            .with_isign(isign)
                            .with_mode_order(mode_order)
                            .with_batch(batch)
                            .with_precision(precision);
                        check_against_reference(&config, point_count, factor * eps, seed);
                        seed += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_plans_reach_their_tolerance_across_eps() {
    for eps in [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-9, 1.0e-12] {
        let precision = if eps >= 1.0e-6 {
            FftPrecision::F32
        } else {
            FftPrecision::F64
        };
        let factor = if precision == FftPrecision::F32 && eps < 1.0e-5 {
            // f32 rounding dominates below about 1e-6.
            30.0
        } else {
            10.0
        };
        for shape in [&[48][..], &[10, 14], &[6, 7, 8]] {
            let config = NufftConfig::new(shape, eps).with_precision(precision);
            check_against_reference(&config, 100, factor * eps, 77);
        }
    }
}

#[test]
fn cpu_plans_handle_clustered_and_boundary_points() {
    // Every point within a hundredth of a cell of the others, plus points on
    // the contract's ends and on fold boundaries.
    for shape in [&[40][..], &[16, 20], &[10, 8, 12]] {
        let dimensions = shape.len();
        let config = NufftConfig::new(shape, 1.0e-9).with_precision(FftPrecision::F64);
        let mut rng = Lcg(5);
        let mut coordinates = (0..400 * dimensions)
            .map(|_| 1.234 + 1.0e-4 * rng.next())
            .collect::<Vec<_>>();
        for edge in [-3.0 * PI, 3.0 * PI, -PI, PI, 0.0, -1.0e-300, 2.0 * PI] {
            coordinates.extend(std::iter::repeat_n(edge, dimensions));
        }
        let point_count = coordinates.len() / dimensions;
        let strengths = interleave(&rng.values(point_count));
        let type1 = CpuNufftPlan::type1(config.clone()).unwrap();
        let modes = deinterleave(&execute(&type1, &coordinates, &strengths).unwrap());
        let expected =
            reference_type1_f64(&config, &coordinates, &deinterleave(&strengths)).unwrap();
        let error = relative_l2(&modes, &expected);
        assert!(error <= 1.0e-8, "{shape:?} clustered type 1: {error:.3e}");

        let mode_count = shape.iter().product::<usize>();
        let coefficients = interleave(&rng.values(mode_count));
        let type2 = CpuNufftPlan::type2(config.clone()).unwrap();
        let values = deinterleave(&execute(&type2, &coordinates, &coefficients).unwrap());
        let expected =
            reference_type2_f64(&config, &coordinates, &deinterleave(&coefficients)).unwrap();
        let error = relative_l2(&values, &expected);
        assert!(error <= 1.0e-8, "{shape:?} clustered type 2: {error:.3e}");
    }
}

#[test]
fn cpu_results_do_not_depend_on_the_thread_count() {
    for (shape, point_count) in [
        (&[4096][..], 30_000),
        (&[96, 80], 20_000),
        (&[24, 20, 28], 12_000),
        (&[8, 6, 10, 6], 3_000),
    ] {
        let dimensions = shape.len();
        let mut rng = Lcg(11);
        // Half the points uniform, half in a tight cluster.
        let mut coordinates = rng.coordinates(point_count / 2 * dimensions);
        coordinates.extend((0..point_count / 2 * dimensions).map(|_| -0.7 + 0.01 * rng.next()));
        for precision in [FftPrecision::F32, FftPrecision::F64] {
            let config = NufftConfig::new(shape, 1.0e-6)
                .with_precision(precision)
                .with_batch(2);
            let strengths = interleave(&rng.values(point_count * 2));
            let mode_count = shape.iter().product::<usize>();
            let coefficients = interleave(&rng.values(mode_count * 2));
            for (kind, input) in [
                (NufftKind::Type1, &strengths),
                (NufftKind::Type2, &coefficients),
            ] {
                let plan = |threads| {
                    let plan = match kind {
                        NufftKind::Type1 => CpuNufftPlan::type1(config.clone()),
                        _ => CpuNufftPlan::type2(config.clone()),
                    };
                    plan.unwrap().with_threads(threads)
                };
                let single = execute(&plan(1), &coordinates, input).unwrap();
                for threads in [2, 3, 8] {
                    let parallel = execute(&plan(threads), &coordinates, input).unwrap();
                    assert!(
                        single
                            .iter()
                            .zip(&parallel)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{shape:?} {precision:?} {kind:?}: {threads} threads changed the result"
                    );
                }
                let again = execute(&plan(8), &coordinates, input).unwrap();
                let first = execute(&plan(8), &coordinates, input).unwrap();
                assert!(again
                    .iter()
                    .zip(&first)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
}

#[test]
fn cpu_plans_accept_no_points() {
    for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
        let config = NufftConfig::new([6, 4], 1.0e-6).with_precision(precision);
        let type1 = CpuNufftPlan::type1(config.clone()).unwrap();
        let modes = execute(&type1, &[], &[]).unwrap();
        assert_eq!(modes.len(), 2 * 24);
        assert!(modes.iter().all(|&value| value == 0.0));
        let type2 = CpuNufftPlan::type2(config).unwrap();
        let coefficients = vec![1.0; 2 * 24];
        assert!(execute(&type2, &[], &coefficients).unwrap().is_empty());
    }
}

#[test]
fn cpu_plans_reject_invalid_inputs() {
    let config = NufftConfig::new([8, 8], 1.0e-6);
    let plan = CpuNufftPlan::type1(config.clone()).unwrap();
    let mut modes = vec![0.0f32; plan.required_output_len(2).unwrap()];

    let nan = [0.0, 0.1, f32::NAN, 0.2];
    assert!(matches!(
        plan.execute(&nan, &[0.0; 4], &mut modes),
        Err(NufftError::NonFiniteCoordinate {
            point: 1,
            axis: 0,
            ..
        })
    ));
    let far = [0.0, 0.1, 0.2, 10.0];
    assert!(matches!(
        plan.execute(&far, &[0.0; 4], &mut modes),
        Err(NufftError::PointOutOfRange {
            point: 1,
            axis: 1,
            ..
        })
    ));
    assert!(matches!(
        plan.execute(&[0.0; 3], &[0.0; 4], &mut modes),
        Err(NufftError::CoordinateLength { .. })
    ));
    assert!(matches!(
        plan.execute(&[0.0; 4], &[0.0; 6], &mut modes),
        Err(NufftError::InputLength {
            expected: 4,
            actual: 6,
            ..
        })
    ));
    assert!(matches!(
        plan.execute(&[0.0; 4], &[0.0; 4], &mut modes[1..]),
        Err(NufftError::InputLength { .. })
    ));
    assert!(matches!(
        plan.execute_f64(&[0.0; 4], &[0.0; 4], &mut [0.0; 128]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));
    let f64_plan = CpuNufftPlan::type2(config.with_precision(FftPrecision::F64)).unwrap();
    assert!(matches!(
        f64_plan.execute(&[0.0; 4], &[0.0; 128], &mut [0.0; 4]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));
}

#[test]
fn cpu_dense_grids_match_the_reference_and_repeat() {
    // Small grids holding many points spread through private grid copies.
    for shape in [&[1000][..], &[16, 16], &[8, 8, 8]] {
        let dimensions = shape.len();
        let config = NufftConfig::new(shape, 1.0e-9).with_precision(FftPrecision::F64);
        let mut rng = Lcg(21);
        let point_count = 40_000;
        let coordinates = rng.coordinates(point_count * dimensions);
        let strengths = interleave(&rng.values(point_count));
        let plan = CpuNufftPlan::type1(config.clone()).unwrap();
        let modes = execute(&plan, &coordinates, &strengths).unwrap();
        let expected =
            reference_type1_f64(&config, &coordinates, &deinterleave(&strengths)).unwrap();
        let error = relative_l2(&deinterleave(&modes), &expected);
        assert!(error <= 1.0e-8, "{shape:?} dense type 1: {error:.3e}");
        for threads in [1, 5] {
            let plan = CpuNufftPlan::type1(config.clone())
                .unwrap()
                .with_threads(threads);
            let repeat = execute(&plan, &coordinates, &strengths).unwrap();
            assert!(
                repeat
                    .iter()
                    .zip(&modes)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{shape:?}: {threads} threads changed the dense result"
            );
        }
    }
}

#[test]
fn cpu_plans_reuse_their_buffers_across_point_counts() {
    // One plan run on shrinking and growing point sets must match a fresh
    // plan bit for bit, whatever its kept buffers hold.
    for shape in [&[300][..], &[40, 36], &[12, 10, 14]] {
        let dimensions = shape.len();
        let mode_count = shape.iter().product::<usize>();
        let config = NufftConfig::new(shape, 1.0e-6).with_batch(2);
        for kind in [NufftKind::Type1, NufftKind::Type2] {
            let make = || match kind {
                NufftKind::Type1 => CpuNufftPlan::type1(config.clone()).unwrap(),
                _ => CpuNufftPlan::type2(config.clone()).unwrap(),
            };
            let reused = make();
            let mut rng = Lcg(31);
            for point_count in [5_000, 700, 0, 9_000, 1] {
                let coordinates = rng.coordinates(point_count * dimensions);
                let input_count = match kind {
                    NufftKind::Type1 => point_count,
                    _ => mode_count,
                };
                let input = interleave(&rng.values(input_count * 2));
                let expected = execute(&make(), &coordinates, &input).unwrap();
                let actual = execute(&reused, &coordinates, &input).unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{shape:?} {kind:?} with {point_count} points differs after reuse"
                );
            }
        }
    }
}

#[test]
fn cpu_prepared_points_repeat_one_shot_results_bitwise() {
    // One point set, prepared by a single-threaded type-1 plan, serves the
    // type-1 and type-2 plans of its configuration at any thread count. The
    // long 1D grid interpolates prepared points in sorted order, the short
    // one in input order.
    for (shape, point_count) in [
        (&[20_000][..], 12_000),
        (&[64][..], 5_000),
        (&[96, 80], 12_000),
        (&[24, 20, 28], 6_000),
        (&[8, 6, 10, 6], 2_000),
    ] {
        let dimensions = shape.len();
        let mode_count = shape.iter().product::<usize>();
        let mut rng = Lcg(41);
        let mut coordinates = rng.coordinates(point_count / 2 * dimensions);
        coordinates.extend((0..point_count / 2 * dimensions).map(|_| 0.3 + 0.01 * rng.next()));
        for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
            let config = NufftConfig::new(shape, 1.0e-7)
                .with_precision(precision)
                .with_batch(2);
            let coordinates = stored(precision, &coordinates);
            let strengths = stored(precision, &interleave(&rng.values(point_count * 2)));
            let coefficients = stored(precision, &interleave(&rng.values(mode_count * 2)));
            let preparer = CpuNufftPlan::type1(config.clone()).unwrap().with_threads(1);
            let points = prepare(&preparer, &coordinates).unwrap();
            assert_eq!(points.len(), point_count);
            assert_eq!(points.dimensions(), dimensions);
            for threads in [1, 4] {
                let type1 = CpuNufftPlan::type1(config.clone())
                    .unwrap()
                    .with_threads(threads);
                let type2 = CpuNufftPlan::type2(config.clone())
                    .unwrap()
                    .with_threads(threads);
                let label = format!("{shape:?} {precision:?} with {threads} threads");
                let modes = execute(&type1, &coordinates, &strengths).unwrap();
                let values = execute(&type2, &coordinates, &coefficients).unwrap();
                for _ in 0..2 {
                    let prepared = execute_prepared(&type1, &points, &strengths).unwrap();
                    assert!(same_bits(&prepared, &modes), "type 1 {label}");
                    let prepared = execute_prepared(&type2, &points, &coefficients).unwrap();
                    assert!(same_bits(&prepared, &values), "type 2 {label}");
                }
            }
        }
    }
}

#[test]
fn cpu_prepared_points_fit_only_matching_plans() {
    let config = NufftConfig::new([16, 12], 1.0e-6);
    let plan = CpuNufftPlan::type1(config.clone()).unwrap();
    let points = plan.prepare_points(&[0.5, -1.0, 2.0, 0.25]).unwrap();
    let strengths = [1.0f32, 0.0, 0.5, -0.5];
    let mut modes = vec![0.0f32; plan.required_output_len(2).unwrap()];
    plan.execute_prepared(&points, &strengths, &mut modes)
        .unwrap();

    let other_grid = CpuNufftPlan::type1(NufftConfig::new([16, 14], 1.0e-6)).unwrap();
    let other_width = CpuNufftPlan::type1(NufftConfig::new([16, 12], 1.0e-3)).unwrap();
    for other in [&other_grid, &other_width] {
        let mut modes = vec![0.0f32; other.required_output_len(2).unwrap()];
        assert!(matches!(
            other.execute_prepared(&points, &strengths, &mut modes),
            Err(NufftError::PointSetMismatch { .. })
        ));
    }
    let df64 = CpuNufftPlan::type1(config.clone().with_precision(FftPrecision::Df64)).unwrap();
    let mut words = vec![0.0f32; df64.required_output_len(2).unwrap()];
    assert!(matches!(
        df64.execute_prepared(&points, &[0.0; 8], &mut words),
        Err(NufftError::PointSetMismatch { .. })
    ));

    // Lengths, coordinates, and precisions are checked as in `execute`.
    assert!(matches!(
        plan.execute_prepared(&points, &strengths[..2], &mut modes),
        Err(NufftError::InputLength { .. })
    ));
    assert!(matches!(
        plan.prepare_points(&[0.0, 0.1, f32::NAN, 0.2]),
        Err(NufftError::NonFiniteCoordinate {
            point: 1,
            axis: 0,
            ..
        })
    ));
    assert!(matches!(
        plan.prepare_points(&[0.0, 0.1, 0.2]),
        Err(NufftError::CoordinateLength { .. })
    ));
    assert!(matches!(
        plan.prepare_points_f64(&[0.0, 0.1]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));
    let f64_plan = CpuNufftPlan::type2(config.with_precision(FftPrecision::F64)).unwrap();
    assert!(matches!(
        f64_plan.prepare_points(&[0.0, 0.1]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));
    let f64_points = f64_plan.prepare_points_f64(&[0.0, 0.1]).unwrap();
    assert!(matches!(
        f64_plan.execute_prepared(&f64_points, &[0.0; 384], &mut [0.0; 2]),
        Err(NufftError::PrecisionUnsupported { .. })
    ));

    // An empty set gives zero modes and no values.
    let empty = plan.prepare_points(&[]).unwrap();
    assert!(empty.is_empty());
    let mut zeros = vec![1.0f32; modes.len()];
    plan.execute_prepared(&empty, &[], &mut zeros).unwrap();
    assert!(zeros.iter().all(|&value| value == 0.0));
    let type2 = CpuNufftPlan::type2(NufftConfig::new([16, 12], 1.0e-6)).unwrap();
    type2
        .execute_prepared(&empty, &vec![1.0; modes.len()], &mut [])
        .unwrap();
}

#[test]
fn cpu_point_sets_can_be_shared_between_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<CpuNufftPoints>();
}
