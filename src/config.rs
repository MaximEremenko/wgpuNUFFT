use crate::error::{NufftError, Result};
use crate::kernel::EsKernel;
use wgpu_fft::FftPrecision;

pub const DEFAULT_EPS: f64 = 1.0e-6;
pub const DEFAULT_SIGMA: f64 = 2.0;
/// Maximum transform dimensionality of a plan and of the CPU reference
/// transforms.
pub const MAX_NUFFT_DIMENSIONS: usize = 8;
/// Maximum dimensionality of a GPU plan. Ranks 1-3 use hand-tuned GPU paths;
/// ranks 4 to 8 use the rank-generic implementation.
pub const MAX_GPU_NUFFT_DIMENSIONS: usize = 8;

/// Sign in the NUFFT exponential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NufftSign {
    #[default]
    Positive,
    Negative,
}

impl NufftSign {
    /// Treats every nonnegative `isign` as positive and every negative
    /// value as negative.
    pub const fn from_isign(isign: i32) -> Self {
        if isign >= 0 {
            Self::Positive
        } else {
            Self::Negative
        }
    }

    pub const fn isign(self) -> i32 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }

    pub(crate) const fn multiplier(self) -> f64 {
        self.isign() as f64
    }
}

/// Storage order for uniform Fourier modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModeOrder {
    /// Increasing centered indices: `-N/2, ..., N/2-1` for even `N`.
    #[default]
    Centered,
    /// FFT order: nonnegative modes first, followed by negative modes.
    Fft,
}

impl ModeOrder {
    pub const fn modeord(self) -> i32 {
        match self {
            Self::Centered => 0,
            Self::Fft => 1,
        }
    }
}

/// Mathematical configuration shared by type-1 and type-2 plans.
#[derive(Debug, Clone, PartialEq)]
pub struct NufftConfig {
    n_modes: Vec<usize>,
    batch: usize,
    eps: f64,
    sign: NufftSign,
    mode_order: ModeOrder,
    sigma: f64,
    precision: FftPrecision,
}

impl NufftConfig {
    /// Creates a configuration. Dimensionality is derived from `n_modes`.
    pub fn new(n_modes: impl Into<Vec<usize>>, eps: f64) -> Self {
        Self {
            n_modes: n_modes.into(),
            batch: 1,
            eps,
            sign: NufftSign::Positive,
            mode_order: ModeOrder::Centered,
            sigma: DEFAULT_SIGMA,
            precision: FftPrecision::F32,
        }
    }

    pub fn with_isign(mut self, isign: i32) -> Self {
        self.sign = NufftSign::from_isign(isign);
        self
    }

    pub fn with_sign(mut self, sign: NufftSign) -> Self {
        self.sign = sign;
        self
    }

    pub fn with_mode_order(mut self, mode_order: ModeOrder) -> Self {
        self.mode_order = mode_order;
        self
    }

    /// Typed builder alias named after the `modeord` option.
    pub fn with_modeord(self, mode_order: ModeOrder) -> Self {
        self.with_mode_order(mode_order)
    }

    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Sets the maximum number of transforms encoded together by this plan.
    ///
    /// All transforms share the same nonuniform points. Complex input and
    /// output vectors are stored transform-major: the complete first vector,
    /// followed by the complete second vector, and so on.
    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    /// Selects the scalar precision used by GPU coordinates, complex values,
    /// kernel evaluation, and the embedded fine-grid FFT.
    pub fn with_precision(mut self, precision: FftPrecision) -> Self {
        self.precision = precision;
        self
    }

    pub fn dimensions(&self) -> usize {
        self.n_modes.len()
    }

    pub fn n_modes(&self) -> &[usize] {
        &self.n_modes
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn sign(&self) -> NufftSign {
        self.sign
    }

    pub fn isign(&self) -> i32 {
        self.sign.isign()
    }

    pub fn mode_order(&self) -> ModeOrder {
        self.mode_order
    }

    pub fn modeord(&self) -> i32 {
        self.mode_order.modeord()
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn precision(&self) -> FftPrecision {
        self.precision
    }

    pub fn mode_count(&self) -> Result<usize> {
        self.n_modes.iter().try_fold(1usize, |count, &length| {
            count.checked_mul(length).ok_or(NufftError::LengthOverflow {
                context: "total Fourier mode count",
            })
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(NufftError::InvalidBatch {
                actual: self.batch,
                maximum: usize::MAX,
            });
        }
        let dimensions = self.dimensions();
        if !(1..=MAX_NUFFT_DIMENSIONS).contains(&dimensions) {
            return Err(NufftError::InvalidDimensions {
                actual: dimensions,
                minimum: 1,
                maximum: MAX_NUFFT_DIMENSIONS,
            });
        }
        for (axis, &length) in self.n_modes.iter().enumerate() {
            if length == 0 {
                return Err(NufftError::ZeroMode { axis });
            }
        }
        let mode_count = self.mode_count()?;
        mode_count
            .checked_mul(self.batch)
            .ok_or(NufftError::LengthOverflow {
                context: "batched Fourier mode count",
            })?;
        // A singleton NUFFT axis is mathematically meaningful, but wgpu-fft
        // rejects selecting a length-one FFT axis in a nontrivial transform.
        // Use only nontrivial axes here to reuse its checked u32 index-space
        // validation without narrowing the NUFFT shape contract.
        let mut fft_shape = self
            .n_modes
            .iter()
            .copied()
            .filter(|&length| length > 1)
            .collect::<Vec<_>>();
        if fft_shape.is_empty() {
            fft_shape.push(1);
        }
        wgpu_fft::FftConfig::new_nd(fft_shape)
            .with_batch(self.batch)
            .with_precision(self.precision)
            .validate()
            .map_err(|source| NufftError::FftShapeUnsupported {
                stage: "requested NUFFT mode shape",
                source,
            })?;
        EsKernel::for_tolerance(self.eps, self.sigma)?;
        Ok(())
    }
}

impl Default for NufftConfig {
    fn default() -> Self {
        Self::new([1], DEFAULT_EPS)
    }
}

/// Estimated error beyond which an `F32` plan misses its tolerance by far:
/// ten times the tolerance, and never less than ordinary `f32` rounding.
const F32_ROUNDING_TOLERANCE_FACTOR: f64 = 10.0;
const F32_ROUNDING_FLOOR: f64 = 1.0e-5;

/// Rejects an `F32` plan whose rounding, amplified by the deconvolution,
/// would miss the tolerance by far. `coefficients` holds, per axis, the
/// centered kernel Fourier coefficients by `|k|`.
///
/// Rounding in the fine grid reaches mode `k` scaled by
/// `phi_hat(0) / phi_hat(k)` along every axis. The estimate is the `f32`
/// unit roundoff times, per axis, the RMS of that gain over the modes; it
/// matched measured errors within a factor of ten for `sigma = 1.25` in two
/// to four dimensions, and stays near the plain roundoff for `sigma = 2`.
pub(crate) fn validate_f32_rounding(config: &NufftConfig, coefficients: &[Vec<f64>]) -> Result<()> {
    if config.precision() != FftPrecision::F32 {
        return Ok(());
    }
    let mut estimate = f64::from(f32::EPSILON) / 2.0;
    for (&modes, axis) in config.n_modes().iter().zip(coefficients) {
        let Some(&center) = axis.first() else {
            continue;
        };
        let half = modes / 2;
        let mean = (0..modes)
            .map(|index| {
                let magnitude = index.abs_diff(half);
                let value = axis.get(magnitude).copied().unwrap_or(0.0).abs();
                if value > 0.0 {
                    (center / value).powi(2)
                } else {
                    f64::INFINITY
                }
            })
            .sum::<f64>()
            / modes as f64;
        estimate *= mean.sqrt();
    }
    if estimate > F32_ROUNDING_TOLERANCE_FACTOR * config.eps() && estimate > F32_ROUNDING_FLOOR {
        return Err(NufftError::F32ToleranceUnreachable {
            eps: config.eps(),
            sigma: config.sigma(),
            estimated_error: estimate,
        });
    }
    Ok(())
}

/// Rejects ranks above [`MAX_GPU_NUFFT_DIMENSIONS`] for a GPU plan of `kind`.
pub(crate) fn validate_gpu_dimensions(kind: &'static str, dimensions: usize) -> Result<()> {
    if dimensions > MAX_GPU_NUFFT_DIMENSIONS {
        return Err(NufftError::GpuDimensionsUnsupported {
            kind,
            actual: dimensions,
            supported: MAX_GPU_NUFFT_DIMENSIONS,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rounding_check(
        n_modes: &[usize],
        eps: f64,
        sigma: f64,
        precision: FftPrecision,
    ) -> Result<()> {
        let config = NufftConfig::new(n_modes.to_vec(), eps)
            .with_sigma(sigma)
            .with_precision(precision);
        let kernel = EsKernel::for_tolerance(eps, sigma).unwrap();
        let coefficients = n_modes
            .iter()
            .map(|&modes| {
                let fine = crate::kernel::select_fine_grid_size(modes, sigma, kernel.width())?;
                kernel.centered_fourier_coefficients(fine)
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();
        validate_f32_rounding(&config, &coefficients)
    }

    #[test]
    fn f32_plans_reject_tolerances_their_rounding_misses_by_far() {
        // sigma = 1.25 at 1e-6 measured 2e-5 in 2D, 3e-4 in 3D and 1e-2 in 4D.
        for n_modes in [vec![256, 256], vec![64; 3], vec![16; 4]] {
            assert!(matches!(
                rounding_check(&n_modes, 1.0e-6, 1.25, FftPrecision::F32),
                Err(NufftError::F32ToleranceUnreachable { .. })
            ));
            for precision in [FftPrecision::F64, FftPrecision::Df64] {
                assert!(rounding_check(&n_modes, 1.0e-6, 1.25, precision).is_ok());
            }
        }
        // Reachable: a looser tolerance, or sigma = 2 at any tolerance.
        assert!(rounding_check(&[16; 4], 1.0e-4, 1.25, FftPrecision::F32).is_ok());
        assert!(rounding_check(&[6, 5, 4, 4], 1.0e-3, 1.25, FftPrecision::F32).is_ok());
        for n_modes in [vec![1 << 20], vec![1024, 1024], vec![128; 3], vec![16; 6]] {
            assert!(rounding_check(&n_modes, 1.0e-9, 2.0, FftPrecision::F32).is_ok());
        }
    }

    #[test]
    fn batch_defaults_to_one_and_rejects_zero() {
        let default = NufftConfig::new([8], DEFAULT_EPS);
        assert_eq!(default.batch(), 1);
        assert_eq!(default.precision(), FftPrecision::F32);
        assert_eq!(default.clone().with_batch(4).batch(), 4);
        assert_eq!(
            default
                .clone()
                .with_precision(FftPrecision::F64)
                .precision(),
            FftPrecision::F64
        );
        assert_eq!(
            default.with_batch(0).validate(),
            Err(NufftError::InvalidBatch {
                actual: 0,
                maximum: usize::MAX,
            })
        );
    }

    #[test]
    fn batched_mode_count_overflow_is_structured() {
        assert_eq!(
            NufftConfig::new([2], DEFAULT_EPS)
                .with_batch(usize::MAX)
                .validate(),
            Err(NufftError::LengthOverflow {
                context: "batched Fourier mode count",
            })
        );
    }
}
