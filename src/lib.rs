//! Portable type-1, type-2, and type-3 nonuniform FFT plans built on
//! `wgpu-fft`.
//!
//! The crate provides reusable planning metadata, direct
//! `f64` reference transforms, and exponential-of-semicircle kernel tables for
//! all three transform types. GPU execution supports arbitrary rank up to
//! [`MAX_NUFFT_DIMENSIONS`]: dedicated 1D, 2D, and 3D paths are complemented by
//! a rank-generic path for every supported dimension `d >= 4`. Plans support
//! `f32`, native `f64`, and portable double-float precision, and encode
//! spreading or interpolation around a fine-grid C2C transform using
//! caller-owned input and output buffers.

#![forbid(unsafe_code)]

mod config;
mod direct;
mod error;
mod gpu;
mod gpu_2d;
mod gpu_3d;
mod gpu_bin_sort;
mod gpu_dense_spread;
mod gpu_dispatch;
mod gpu_nd;
mod gpu_point_bins;
#[cfg(feature = "gpu-profiling")]
mod gpu_profile;
mod gpu_recorder;
mod gpu_scan;
mod gpu_type1;
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
    ModeOrder, NufftConfig, NufftSign, DEFAULT_EPS, DEFAULT_SIGMA, MAX_NUFFT_DIMENSIONS,
};
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
pub use wgpu_fft::math::{Complex64, ComplexDoubleFloat, DoubleFloat};
pub use wgpu_fft::FftPrecision;
