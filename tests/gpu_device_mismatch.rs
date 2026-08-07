#![cfg(not(target_arch = "wasm32"))]

//! Opt-in check that GPU plans reject a device they were not created with,
//! instead of letting wgpu panic on a cross-device binding.

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    NufftConfig, NufftError, NufftInterval, NufftPlan, NufftType3Config, NufftType3Plan,
};

#[test]
fn gpu_plans_reject_a_foreign_device() {
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
    let (foreign, _foreign_queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.test.foreign_device"),
            required_limits: context.adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("a second device from the same adapter");
    let device = &context.device;
    let queue = &context.queue;

    let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.points"),
        contents: bytemuck::cast_slice(&[0.25_f32, -1.0, 2.0, 3.0]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let complex = |count: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.test.complex"),
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let values = complex(4);
    let modes = complex(16);
    let mismatch = |kind| NufftError::GpuDeviceMismatch { kind };
    // Encoders from both devices: a mismatch must be reported before any
    // command reaches either one.
    let mut encoder = foreign.create_command_encoder(&Default::default());

    let type1 = NufftPlan::type1_gpu(device, queue, NufftConfig::new([16], 1.0e-6)).unwrap();
    assert_eq!(
        type1.encode_type1_gpu(&foreign, &mut encoder, 4, &points, &values, &modes),
        Err(mismatch("type-1"))
    );
    assert_eq!(
        type1.set_points_gpu(&foreign, &mut encoder, 4, &points),
        Err(mismatch("type-1"))
    );

    let type2 = NufftPlan::type2_gpu(device, queue, NufftConfig::new([16], 1.0e-6)).unwrap();
    assert_eq!(
        type2.encode_type2_gpu(&foreign, &mut encoder, 4, &points, &modes, &values),
        Err(mismatch("type-2"))
    );
    assert_eq!(
        type2.set_points_gpu(&foreign, &mut encoder, 4, &points),
        Err(mismatch("type-2"))
    );

    let type3 = NufftType3Plan::new_gpu(
        device,
        queue,
        NufftType3Config::new(
            vec![NufftInterval::new(-3.5, 3.5)],
            vec![NufftInterval::new(-4.0, 4.0)],
            1.0e-6,
        ),
    )
    .unwrap();
    assert_eq!(
        type3.encode_gpu(
            &foreign,
            &mut encoder,
            4,
            &points,
            &values,
            4,
            &points,
            &values
        ),
        Err(mismatch("type-3"))
    );

    // The plans stay usable with their own device.
    let mut own = device.create_command_encoder(&Default::default());
    type1
        .encode_type1_gpu(device, &mut own, 4, &points, &values, &modes)
        .unwrap();
    queue.submit([own.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    drop(encoder);
    // Mirror the other GPU tests, which leak their devices to avoid a
    // teardown stall on Windows.
    std::mem::forget(foreign);
    std::mem::forget(context);
}
