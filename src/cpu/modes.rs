//! Deconvolution between the Fourier modes and the fine grid.

use super::filled;
use super::grid::FineGrid;
use super::real::Real;
use super::workers::for_each_chunk_group;
use crate::config::ModeOrder;
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};

/// Words per thread below which the passes stay on one thread.
const MIN_WORDS_PER_THREAD: usize = 1 << 15;
/// Marks a fine-grid index that holds no mode.
const UNMAPPED: u32 = u32::MAX;

/// Where the modes of one axis sit on the fine grid, and their
/// deconvolution factors.
pub(crate) struct AxisModes<T> {
    /// Fine-grid index of every mode, in storage order.
    fine_index: Vec<usize>,
    /// `1 / phi_hat(k)` of every mode, in storage order.
    amplitude: Vec<T>,
    /// Storage index of the mode at every fine-grid index, or [`UNMAPPED`].
    mode_at: Vec<u32>,
}

impl<T: Real> AxisModes<T> {
    /// `coefficients` are the axis's centered kernel Fourier coefficients
    /// `phi_hat(|k|)`.
    pub(crate) fn new(
        order: ModeOrder,
        modes: usize,
        fine_length: usize,
        coefficients: &[f64],
    ) -> Result<Self> {
        let mut fine_index = filled("CPU mode fine-grid indices", modes, 0usize)?;
        let mut amplitude = filled("CPU deconvolution amplitudes", modes, T::ZERO)?;
        let mut mode_at = filled("CPU fine-grid mode indices", fine_length, UNMAPPED)?;
        for index in 0..modes {
            let mode = mode_for_storage_index(order, index, modes);
            let coefficient = coefficients[mode.unsigned_abs() as usize];
            let value = T::from_f64(1.0 / coefficient);
            if !coefficient.is_finite() || coefficient == 0.0 || !value.is_finite() {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
            let cell = if mode >= 0 {
                mode as usize
            } else {
                fine_length - mode.unsigned_abs() as usize
            };
            fine_index[index] = cell;
            amplitude[index] = value;
            mode_at[cell] = index as u32;
        }
        Ok(Self {
            fine_index,
            amplitude,
            mode_at,
        })
    }
}

/// Type 1: deconvolves and truncates the transformed fine grids in `fine`
/// into transform-major, interleaved `modes`.
pub(crate) fn deconvolve<T: Real, const D: usize>(
    axes: &[AxisModes<T>],
    grid: &FineGrid<D>,
    n_modes: &[usize],
    fine: &[T],
    modes: &mut [T],
    threads: usize,
) {
    let row_len = n_modes[0];
    let rows_per_transform = n_modes[1..].iter().product::<usize>();
    let cells = grid.cells();
    let min_rows = MIN_WORDS_PER_THREAD.div_ceil(2 * row_len);
    for_each_chunk_group(threads, modes, 2 * row_len, min_rows, |first, rows| {
        for (offset, row) in rows.chunks_exact_mut(2 * row_len).enumerate() {
            let row_index = first + offset;
            let mut rest = row_index % rows_per_transform;
            let mut base = row_index / rows_per_transform * cells;
            let mut amplitude = T::ONE;
            for axis in 1..D {
                let index = rest % n_modes[axis];
                rest /= n_modes[axis];
                base += axes[axis].fine_index[index] * grid.strides[axis];
                amplitude = amplitude * axes[axis].amplitude[index];
            }
            let axis = &axes[0];
            for (pair, (&cell, &factor)) in row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(axis.fine_index.iter().zip(&axis.amplitude))
            {
                let source = 2 * (base + cell);
                let scale = amplitude * factor;
                pair[0] = fine[source] * scale;
                pair[1] = fine[source + 1] * scale;
            }
        }
    });
}

/// Type 2: writes the deconvolved, zero-padded transform-major
/// `coefficients` onto the fine grids in `fine`.
pub(crate) fn predeconvolve<T: Real, const D: usize>(
    axes: &[AxisModes<T>],
    grid: &FineGrid<D>,
    n_modes: &[usize],
    coefficients: &[T],
    fine: &mut [T],
    threads: usize,
) {
    let row_len = grid.shape[0];
    let rows_per_transform = grid.cells() / row_len;
    let mode_count = n_modes.iter().product::<usize>();
    let min_rows = MIN_WORDS_PER_THREAD.div_ceil(2 * row_len);
    for_each_chunk_group(threads, fine, 2 * row_len, min_rows, |first, rows| {
        'rows: for (offset, row) in rows.chunks_exact_mut(2 * row_len).enumerate() {
            row.fill(T::ZERO);
            let row_index = first + offset;
            let mut rest = row_index % rows_per_transform;
            let mut source = row_index / rows_per_transform * mode_count;
            let mut amplitude = T::ONE;
            let mut mode_stride = n_modes[0];
            for axis in 1..D {
                let cell = rest % grid.shape[axis];
                rest /= grid.shape[axis];
                let mode = axes[axis].mode_at[cell];
                if mode == UNMAPPED {
                    continue 'rows;
                }
                source += mode as usize * mode_stride;
                amplitude = amplitude * axes[axis].amplitude[mode as usize];
                mode_stride *= n_modes[axis];
            }
            let axis = &axes[0];
            for (mode, (&cell, &factor)) in axis.fine_index.iter().zip(&axis.amplitude).enumerate()
            {
                let value = 2 * (source + mode);
                let scale = amplitude * factor;
                row[2 * cell] = coefficients[value] * scale;
                row[2 * cell + 1] = coefficients[value + 1] * scale;
            }
        }
    });
}
