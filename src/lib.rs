//! Portable type-1, type-2, and type-3 nonuniform FFT plans built on
//! `wgpu-fft`.
//!
//! The crate provides reusable planning metadata, direct
//! `f64` reference transforms, and exponential-of-semicircle kernel tables for
//! all three transform types. GPU execution supports rank up to
//! [`MAX_GPU_NUFFT_DIMENSIONS`]: type 1 spreads through one rank-generic
//! spreader in every rank, type 2 interpolates `f32` through dedicated 1D,
//! 2D, and 3D paths and every other precision and rank through a
//! rank-generic path, and the CPU reference transforms reach
//! [`MAX_NUFFT_DIMENSIONS`] too.
//! Plans support `f32`, native `f64`, and portable double-float precision,
//! and encode spreading or interpolation around a fine-grid C2C transform
//! using caller-owned input and output buffers. With the default `cpu` feature,
//! [`CpuNufftPlan`] and [`CpuNufftType3Plan`] run all three transform types
//! in host memory, with the same configurations and layouts, and prepare
//! point sets ([`CpuNufftPoints`], [`CpuNufftType3Points`]) once for
//! repeated transforms.

#![forbid(unsafe_code)]

mod config;
#[cfg(feature = "cpu")]
mod cpu;
mod direct;
mod error;
mod gpu;
mod gpu_2d;
mod gpu_3d;
mod gpu_bin_sort;
mod gpu_dense_spread;
mod gpu_dispatch;
mod gpu_nd;
mod gpu_nd_bins;
mod gpu_nd_spread;
mod gpu_nd_wgsl;
mod gpu_point_bins;
#[cfg(feature = "gpu-profiling")]
mod gpu_profile;
mod gpu_recorder;
mod gpu_scan;
mod gpu_type1_2d;
mod gpu_type1_3d;
mod gpu_type1_nd;
mod gpu_type1_scratch;
mod gpu_type3;
mod kernel;
mod plan;
mod type3;
#[cfg(test)]
mod wgsl_validation;

pub use config::{
    ModeOrder, NufftConfig, NufftSign, DEFAULT_EPS, DEFAULT_SIGMA, MAX_GPU_NUFFT_DIMENSIONS,
    MAX_NUFFT_DIMENSIONS,
};
#[cfg(feature = "cpu")]
pub use cpu::{CpuNufftPlan, CpuNufftPoints, CpuNufftType3Plan, CpuNufftType3Points};
pub use direct::{reference_type1_f64, reference_type2_f64};
pub use error::{NufftError, Result};
#[cfg(feature = "gpu-profiling")]
pub use gpu_profile::{NufftGpuProfileLayout, NufftGpuStage, NufftGpuStageQuery};
pub use kernel::{select_fine_grid_size, EsKernel, MAX_ES_KERNEL_WIDTH, MIN_ES_KERNEL_WIDTH};
pub use plan::{NufftKind, NufftPlan};
pub use type3::{
    es_kernel_fourier_transform, reference_type3_f64, NufftInterval, NufftType3AxisMetadata,
    NufftType3Config, NufftType3Plan,
};
/// The `wgpu` version whose devices, queues, and buffers the plans take; using
/// this re-export avoids a mismatched second copy.
pub use wgpu;
/// The `wgpu-fft` version behind the fine-grid transforms.
pub use wgpu_fft;
pub use wgpu_fft::math::{Complex64, ComplexDoubleFloat, DoubleFloat};
pub use wgpu_fft::FftPrecision;

/// Compiles the README's Rust examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
