#![cfg(not(target_arch = "wasm32"))]

//! Opt-in GPU checks of tightly clustered type-1 inputs: bins far above the
//! 64-point ranking limit, sorted by the parallel large-bin sort, and gather
//! tiles and segments whose reach exceeds the heavy-path thresholds.

use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{reference_type1_f64, Complex64, NufftConfig, NufftPlan};

const EPS: f64 = 1.0e-6;
/// The float regression factor of the 2D tests.
const TOLERANCE: f64 = 20.0 * EPS;
const BATCH: usize = 2;

#[test]
fn gpu_clustered_type1_is_accurate_and_deterministic() {
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
    // Three merge stages; every segment near the cluster takes the heavy path.
    check(device, queue, &[64], &clustered(300_000, &[0.3], 1));
    // Two merge stages and the copy back; heavy tiles.
    check(
        device,
        queue,
        &[32, 40],
        &clustered(20_000, &[0.3, -1.1], 2),
    );
    // Two merge stages through the 3D block spreader.
    check(
        device,
        queue,
        &[16, 20, 12],
        &clustered(10_000, &[0.3, -1.1, 2.0], 3),
    );
    // 300,000 points pass the 3D heavy-block limit, while either half stays
    // below it, so the heavy passes must match the light ones by linearity.
    check_heavy_blocks_by_linearity(
        device,
        queue,
        &[16, 20, 12],
        &clustered(300_000, &[0.3, -1.1, 2.0], 4),
    );
    // Mirror the other GPU tests, which leak their devices to avoid a
    // teardown stall on Windows.
    std::mem::forget(context);
}

/// `count` points within 1e-3 of `center`, then 257 points spread over the
/// whole domain, point-major.
fn clustered(count: usize, center: &[f32], seed: u32) -> Vec<f32> {
    let mut state = 0x9e37_79b9u32 ^ seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    let mut points = Vec::new();
    for _ in 0..count {
        for &axis_center in center {
            points.push(axis_center + (next() - 0.5) * 2.0e-3);
        }
    }
    for _ in 0..257 {
        for _ in center {
            points.push((next() - 0.5) * 2.0 * std::f32::consts::PI);
        }
    }
    points
}

fn check(device: &wgpu::Device, queue: &wgpu::Queue, shape: &[usize], points: &[f32]) {
    let dimensions = shape.len();
    let point_count = points.len() / dimensions;
    let config = NufftConfig::new(shape.to_vec(), EPS).with_batch(BATCH);
    let plan = NufftPlan::type1_gpu(device, queue, config).unwrap();
    let strengths: Vec<f32> = (0..BATCH * point_count)
        .flat_map(|index| {
            let x = index as f32 * 0.37;
            [x.sin() * 0.7 + 0.1, (x * 0.61).cos() * 0.5 - 0.2]
        })
        .collect();

    let encoded = execute(device, queue, &plan, points, &strengths, false);
    let repeated = execute(device, queue, &plan, points, &strengths, false);
    assert!(
        bits(&encoded) == bits(&repeated),
        "{shape:?}: repeated execution changed bits"
    );
    let reused = execute(device, queue, &plan, points, &strengths, true);
    assert!(
        bits(&encoded) == bits(&reused),
        "{shape:?}: set_points + execute differs from encode"
    );

    let single = NufftConfig::new(shape.to_vec(), EPS);
    let points_f64: Vec<f64> = points.iter().map(|&x| f64::from(x)).collect();
    let mode_count: usize = shape.iter().product();
    for vector in 0..BATCH {
        let vector_strengths: Vec<Complex64> = strengths
            [2 * vector * point_count..2 * (vector + 1) * point_count]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect();
        let reference = reference_type1_f64(&single, &points_f64, &vector_strengths).unwrap();
        let actual: Vec<Complex64> = encoded
            [2 * vector * mode_count..2 * (vector + 1) * mode_count]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect();
        let error = relative_l2(&actual, &reference);
        eprintln!(
            "NUFFT_CLUSTERED shape={shape:?} M={point_count} vector={vector} relative_l2={error:.3e}"
        );
        assert!(error <= TOLERANCE, "{shape:?} vector {vector}: {error}");
    }
}

/// Compares the transform of all points with the sum of the transforms of
/// two halves that each stay below the heavy-block limit.
fn check_heavy_blocks_by_linearity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shape: &[usize],
    points: &[f32],
) {
    let dimensions = shape.len();
    let point_count = points.len() / dimensions;
    let half = point_count / 2;
    let config = NufftConfig::new(shape.to_vec(), EPS).with_batch(BATCH);
    let plan = NufftPlan::type1_gpu(device, queue, config).unwrap();
    let strengths: Vec<f32> = (0..BATCH * point_count)
        .flat_map(|index| {
            let x = index as f32 * 0.29;
            [x.cos() * 0.6 - 0.1, (x * 0.43).sin() * 0.8 + 0.2]
        })
        .collect();
    let vector_part = |range: std::ops::Range<usize>| -> Vec<f32> {
        (0..BATCH)
            .flat_map(|vector| {
                let base = vector * point_count;
                strengths[2 * (base + range.start)..2 * (base + range.end)].to_vec()
            })
            .collect()
    };

    let whole = execute(device, queue, &plan, points, &strengths, false);
    let repeated = execute(device, queue, &plan, points, &strengths, true);
    assert!(
        bits(&whole) == bits(&repeated),
        "{shape:?}: heavy blocks are not deterministic"
    );
    let first = execute(
        device,
        queue,
        &plan,
        &points[..half * dimensions],
        &vector_part(0..half),
        false,
    );
    let second = execute(
        device,
        queue,
        &plan,
        &points[half * dimensions..],
        &vector_part(half..point_count),
        false,
    );
    let difference = whole
        .iter()
        .zip(first.iter().zip(&second))
        .map(|(&whole, (&first, &second))| (f64::from(whole) - f64::from(first + second)).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = whole
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    let relative = difference / norm;
    eprintln!("NUFFT_CLUSTERED_HEAVY shape={shape:?} M={point_count} linearity={relative:.3e}");
    assert!(
        relative <= 1.0e-5,
        "{shape:?}: heavy blocks differ from light by {relative}"
    );
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn execute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f32],
    strengths: &[f32],
    set_points_first: bool,
) -> Vec<f32> {
    let point_count = points.len() / plan.config().dimensions();
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let strength_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.strengths"),
        contents: bytemuck::cast_slice(strengths),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = plan
        .required_type1_output_buffer_size_bytes_for_batch(BATCH)
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
    if set_points_first {
        plan.set_points_gpu(device, &mut encoder, point_count, &point_buffer)
            .unwrap();
        plan.execute_type1_gpu_batch(device, &mut encoder, BATCH, &strength_buffer, &output)
            .unwrap();
    } else {
        plan.encode_type1_gpu_batch(
            device,
            &mut encoder,
            BATCH,
            point_count,
            &point_buffer,
            &strength_buffer,
            &output,
        )
        .unwrap();
    }
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

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let error = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| {
            (actual.re - reference.re).powi(2) + (actual.im - reference.im).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let norm = reference
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>()
        .sqrt();
    error / norm.max(f64::MIN_POSITIVE)
}
