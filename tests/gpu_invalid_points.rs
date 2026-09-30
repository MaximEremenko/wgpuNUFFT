#![cfg(not(target_arch = "wasm32"))]

//! Opt-in GPU checks of coordinates outside the `[-3*pi, 3*pi]`
//! contract, non-finite ones included. Every spreader clamps their bins, so
//! they leave the other points' type-2 outputs bit for bit, later executions
//! of the plan intact, and, when they are finite but far away, type-1 modes
//! exactly as without them. The binned F32 paths also give non-finite and
//! out-of-reach points zero weight: type-1 modes then equal those of the
//! remaining points exactly, and type-2 interpolates zero at them. The
//! rank-generic spreader gives them zero weight too, but splits the points of
//! a crowded block into parts by count, so zero-weight points can move the
//! part boundaries and the modes match up to rounding.

use std::f32::consts::PI;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{FftPrecision, NufftConfig, NufftKind, NufftPlan};

const EPS: f64 = 1.0e-6;
const VALID_POINTS: usize = 300;
/// One invalid point follows every this many valid ones.
const INVALID_EVERY: usize = 25;
const INVALID: [f32; 7] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1.0e30,
    -1.0e30,
    -40.0,
    5.0 * PI,
];
/// Finite coordinates whose kernel support misses the grid on every path.
const FAR: [f32; 3] = [1.0e30, -1.0e30, 1.0e20];

#[test]
fn gpu_invalid_points_leave_other_points_intact() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run());
}

async fn run() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let device = &context.device;
    let queue = &context.queue;
    // F32 type-1 plans of these shapes take the binned spreaders.
    for shape in [vec![64], vec![24, 20], vec![16, 16, 12]] {
        let invalid = Points::new(shape.len(), &INVALID);
        let case = Case {
            device,
            queue,
            shape: &shape,
            precision: FftPrecision::F32,
        };
        case.check_others_kept(&invalid);
        case.check_type1_ignores(&Points::new(shape.len(), &FAR));
        case.check_type1_ignores(&invalid);
        case.check_type2_zero_at(&invalid);
        eprintln!("NUFFT_INVALID_POINTS shape={shape:?} binned F32 ok");
    }
    // These are too small for the binned spreaders, so their type-1 plans
    // spread through the rank-generic path, like every Df64 one.
    for shape in [vec![4], vec![6, 5], vec![12, 10, 8]] {
        let invalid = Points::new(shape.len(), &INVALID);
        let far = Points::new(shape.len(), &FAR);
        for precision in [FftPrecision::F32, FftPrecision::Df64] {
            let case = Case {
                device,
                queue,
                shape: &shape,
                precision,
            };
            case.check_others_kept(&invalid);
            case.check_type1_nearly_ignores(&far);
            eprintln!("NUFFT_INVALID_POINTS shape={shape:?} small {precision:?} ok");
        }
    }
    // The rank-generic paths bin and prepare every point, in every precision.
    for shape in [vec![8, 6, 6, 6], vec![4, 4, 3, 3, 3, 3]] {
        let invalid = Points::new(shape.len(), &INVALID);
        let far = Points::new(shape.len(), &FAR);
        for precision in [FftPrecision::F32, FftPrecision::Df64] {
            let case = Case {
                device,
                queue,
                shape: &shape,
                precision,
            };
            case.check_others_kept(&invalid);
            case.check_type1_nearly_ignores(&far);
            case.check_type1_nearly_ignores(&invalid);
            case.check_type2_zero_at(&invalid);
            eprintln!("NUFFT_INVALID_POINTS shape={shape:?} rank-generic {precision:?} ok");
        }
    }
    // Mirror the other GPU tests, which leak their devices to avoid a
    // teardown stall on Windows.
    std::mem::forget(context);
}

/// Point-major coordinates with and without the invalid points.
struct Points {
    valid: Vec<f32>,
    mixed: Vec<f32>,
    /// For every mixed point, its index among the valid points, if valid.
    valid_index: Vec<Option<usize>>,
}

impl Points {
    /// Inserts a point after every [`INVALID_EVERY`] valid ones whose axes
    /// are valid but one, which cycles through the axes and `invalid`.
    fn new(dimensions: usize, invalid: &[f32]) -> Self {
        let mut state = 0x2545_f491u32 ^ dimensions as u32;
        let mut coordinate = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 - 0.5) * 2.0 * PI
        };
        let mut valid = Vec::new();
        let mut mixed = Vec::new();
        let mut valid_index = Vec::new();
        for index in 0..VALID_POINTS {
            let point: Vec<f32> = (0..dimensions).map(|_| coordinate()).collect();
            valid.extend_from_slice(&point);
            mixed.extend_from_slice(&point);
            valid_index.push(Some(index));
            if index % INVALID_EVERY == INVALID_EVERY - 1 {
                let count = index / INVALID_EVERY;
                for axis in 0..dimensions {
                    mixed.push(if axis == count % dimensions {
                        invalid[count % invalid.len()]
                    } else {
                        coordinate()
                    });
                }
                valid_index.push(None);
            }
        }
        Self {
            valid,
            mixed,
            valid_index,
        }
    }

    fn mixed_count(&self) -> usize {
        self.valid_index.len()
    }

    /// Mixed-point `values` of the valid points only, `words` per point.
    fn valid_part(&self, values: &[f32], words: usize) -> Vec<f32> {
        self.valid_index
            .iter()
            .enumerate()
            .filter(|(_, valid)| valid.is_some())
            .flat_map(|(index, _)| values[words * index..words * (index + 1)].to_vec())
            .collect()
    }
}

struct Case<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    shape: &'a [usize],
    precision: FftPrecision,
}

impl Case<'_> {
    fn config(&self) -> NufftConfig {
        NufftConfig::new(self.shape.to_vec(), EPS).with_precision(self.precision)
    }

    /// Words per stored scalar.
    fn scalar_words(&self) -> usize {
        match self.precision {
            FftPrecision::Df64 => 2,
            _ => 1,
        }
    }

    /// `values` in the case's storage: Df64 gets zero low words.
    fn stored(&self, values: &[f32]) -> Vec<f32> {
        match self.precision {
            FftPrecision::Df64 => values.iter().flat_map(|&value| [value, 0.0]).collect(),
            _ => values.to_vec(),
        }
    }

    fn execute(
        &self,
        plan: &NufftPlan,
        point_count: usize,
        points: &[f32],
        input: &[f32],
    ) -> Vec<f32> {
        execute(
            self.device,
            self.queue,
            plan,
            point_count,
            &self.stored(points),
            &self.stored(input),
        )
    }

    /// Other points' type-2 outputs stay bit for bit, and the type-1 plan
    /// computes the same modes before and after a set with invalid points.
    fn check_others_kept(&self, points: &Points) {
        let shape = self.shape;
        let precision = self.precision;
        let type2 = NufftPlan::type2_gpu(self.device, self.queue, self.config()).unwrap();
        let coefficients = values(shape.iter().product(), 0.5);
        let valid = self.execute(&type2, VALID_POINTS, &points.valid, &coefficients);
        let mixed = self.execute(&type2, points.mixed_count(), &points.mixed, &coefficients);
        let words = 2 * self.scalar_words();
        assert!(
            bits(&points.valid_part(&mixed, words)) == bits(&valid),
            "{shape:?} {precision:?}: invalid points changed other type-2 outputs"
        );

        let type1 = NufftPlan::type1_gpu(self.device, self.queue, self.config()).unwrap();
        let strengths = values(points.mixed_count(), 1.5);
        let valid_strengths = points.valid_part(&strengths, 2);
        let before = self.execute(&type1, VALID_POINTS, &points.valid, &valid_strengths);
        self.execute(&type1, points.mixed_count(), &points.mixed, &strengths);
        let after = self.execute(&type1, VALID_POINTS, &points.valid, &valid_strengths);
        assert!(
            bits(&after) == bits(&before),
            "{shape:?} {precision:?}: invalid points changed a later type-1 execution"
        );
    }

    /// Type-1 modes equal those of the valid points alone.
    fn check_type1_ignores(&self, points: &Points) {
        let shape = self.shape;
        let precision = self.precision;
        let type1 = NufftPlan::type1_gpu(self.device, self.queue, self.config()).unwrap();
        let strengths = values(points.mixed_count(), 1.5);
        let valid_strengths = points.valid_part(&strengths, 2);
        let valid = self.execute(&type1, VALID_POINTS, &points.valid, &valid_strengths);
        let mixed = self.execute(&type1, points.mixed_count(), &points.mixed, &strengths);
        // Zero weights add signed zeros, so compare values rather than bits.
        assert!(
            mixed == valid,
            "{shape:?} {precision:?}: invalid points changed type-1 modes"
        );
    }

    /// Type-1 modes equal those of the valid points alone up to the rounding
    /// of another summation grouping, as the rank-generic spreader gives.
    fn check_type1_nearly_ignores(&self, points: &Points) {
        let shape = self.shape;
        let precision = self.precision;
        let type1 = NufftPlan::type1_gpu(self.device, self.queue, self.config()).unwrap();
        let strengths = values(points.mixed_count(), 1.5);
        let valid_strengths = points.valid_part(&strengths, 2);
        let valid = self.execute(&type1, VALID_POINTS, &points.valid, &valid_strengths);
        let mixed = self.execute(&type1, points.mixed_count(), &points.mixed, &strengths);
        let error = relative_l2(&self.wide(&mixed), &self.wide(&valid));
        let tolerance = match precision {
            FftPrecision::F32 => 1.0e-5,
            _ => 1.0e-11,
        };
        assert!(
            error <= tolerance,
            "{shape:?} {precision:?}: invalid points changed type-1 modes by {error:e}"
        );
    }

    /// Stored words as values; `Df64` adds each low word to its high word.
    fn wide(&self, words: &[f32]) -> Vec<f64> {
        match self.precision {
            FftPrecision::Df64 => words
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[hi, lo]| f64::from(hi) + f64::from(lo))
                .collect(),
            _ => words.iter().map(|&word| f64::from(word)).collect(),
        }
    }

    /// Type-2 interpolates exactly zero at every invalid point.
    fn check_type2_zero_at(&self, points: &Points) {
        let shape = self.shape;
        let type2 = NufftPlan::type2_gpu(self.device, self.queue, self.config()).unwrap();
        let coefficients = values(shape.iter().product(), 0.5);
        let mixed = self.execute(&type2, points.mixed_count(), &points.mixed, &coefficients);
        let words = 2 * self.scalar_words();
        for (index, valid) in points.valid_index.iter().enumerate() {
            if valid.is_none() {
                assert!(
                    mixed[words * index..words * (index + 1)]
                        .iter()
                        .all(|&word| word == 0.0),
                    "{shape:?}: type-2 output at invalid point {index} is not zero"
                );
            }
        }
    }
}

/// Deterministic interleaved complex values.
fn values(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .flat_map(|index| {
            let x = index as f32 * 0.37 + seed;
            [x.sin() * 0.7 + 0.1, (x * 0.61).cos() * 0.5 - 0.2]
        })
        .collect()
}

fn relative_l2(actual: &[f64], reference: &[f64]) -> f64 {
    let error = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>();
    let norm = reference.iter().map(|b| b * b).sum::<f64>();
    (error / norm).sqrt()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn execute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    point_count: usize,
    points: &[f32],
    input: &[f32],
) -> Vec<f32> {
    let storage = |label, contents: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(contents),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let point_buffer = storage("wgpu_nufft.test.points", points);
    let input_buffer = storage("wgpu_nufft.test.input", input);
    let output_bytes = match plan.kind() {
        NufftKind::Type1 => plan.required_type1_output_buffer_size_bytes(),
        _ => plan.required_type2_output_buffer_size_bytes(point_count),
    }
    .unwrap();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    match plan.kind() {
        NufftKind::Type1 => plan.encode_type1_gpu(
            device,
            &mut encoder,
            point_count,
            &point_buffer,
            &input_buffer,
            &output,
        ),
        _ => plan.encode_type2_gpu(
            device,
            &mut encoder,
            point_count,
            &point_buffer,
            &input_buffer,
            &output,
        ),
    }
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().expect("mapped readback range");
    let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    result
}
