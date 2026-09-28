//! Scalar types of the CPU plans.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

use wgpu_fft::{CpuFftPlan, FftError};

/// Arithmetic of a plan: `f32` for `F32` plans, `f64` for `F64` and `Df64`.
pub(crate) trait Real:
    bytemuck::Pod
    + Debug
    + PartialEq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f64(value: f64) -> Self;

    fn to_f64(self) -> f64;

    fn is_finite(self) -> bool;

    /// Transforms `input` into `output` with `plan`, whose precision
    /// computes in `Self`.
    fn fft(plan: &CpuFftPlan, input: &[Self], output: &mut [Self]) -> Result<(), FftError>;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }

    fn fft(plan: &CpuFftPlan, input: &[Self], output: &mut [Self]) -> Result<(), FftError> {
        plan.execute(input, output)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    fn fft(plan: &CpuFftPlan, input: &[Self], output: &mut [Self]) -> Result<(), FftError> {
        plan.execute_f64(input, output)
    }
}

/// A stored point coordinate: `f32` for `F32` plans, `f64` otherwise.
pub(crate) trait Coordinate: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

impl Coordinate for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Coordinate for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
