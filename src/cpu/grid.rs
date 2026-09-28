//! Fine-grid geometry and kernel weights of the CPU plans.

use std::f64::consts::TAU;

use super::real::Real;
use crate::kernel::{EsHornerTable, EsKernel, MAX_ES_KERNEL_WIDTH};

pub(crate) const MAX_WIDTH: usize = MAX_ES_KERNEL_WIDTH;

/// Shape of a fine grid and the mapping of coordinates onto it. Axis zero is
/// contiguous.
pub(crate) struct FineGrid<const D: usize> {
    pub(crate) shape: [usize; D],
    /// Cells between consecutive indices of each axis.
    pub(crate) strides: [usize; D],
    /// Cells per radian of each axis.
    scale: [f64; D],
    width: usize,
    half_width: f64,
}

impl<const D: usize> FineGrid<D> {
    pub(crate) fn new(shape: &[usize], kernel: EsKernel) -> Self {
        let shape: [usize; D] = shape
            .try_into()
            .expect("the fine-grid shape has the plan's dimension count");
        let mut strides = [1; D];
        for axis in 1..D {
            strides[axis] = strides[axis - 1] * shape[axis - 1];
        }
        Self {
            shape,
            strides,
            scale: shape.map(|length| length as f64 / TAU),
            width: kernel.width(),
            half_width: kernel.half_width(),
        }
    }

    /// Cells of one transform's grid.
    pub(crate) fn cells(&self) -> usize {
        self.strides[D - 1] * self.shape[D - 1]
    }

    /// First support cell of `coordinate` on `axis`, wrapped into the grid,
    /// and the local panel coordinate `t` in `[-1, 1)` that the kernel
    /// weights of all `width` support cells share.
    ///
    /// As on the GPU, coordinate zero lies on the middle cell and positions
    /// fold periodically into the grid; coordinates must lie within
    /// `[-3*pi, 3*pi]`.
    #[inline]
    pub(crate) fn support(&self, axis: usize, coordinate: f64) -> (usize, f64) {
        let length = self.shape[axis];
        let length_f64 = length as f64;
        let mut position = coordinate * self.scale[axis] + (length / 2) as f64;
        if position < 0.0 {
            position += length_f64;
        }
        if position < 0.0 {
            position += length_f64;
        }
        if position >= length_f64 {
            position -= length_f64;
        }
        if position >= length_f64 {
            position -= length_f64;
        }
        let start = (position - self.half_width).ceil();
        let t = 2.0 * (start - position) + (self.width - 1) as f64;
        // `start` lies in [-width/2, length), and the grid is at least two
        // kernel widths long.
        let start = start as i64;
        let start = if start < 0 {
            start + length as i64
        } else {
            start
        };
        (start as usize, t)
    }
}

/// The ES kernel as the piecewise Horner polynomials of the GPU `F64` and
/// `Df64` paths, in the plan's arithmetic.
pub(crate) struct Weights<T> {
    width: usize,
    /// Horner rows, highest degree first. Row `c` holds coefficient `c` of
    /// every panel, padded with zeros to [`MAX_WIDTH`] panels.
    rows: Vec<[T; MAX_WIDTH]>,
}

impl<T: Real> Weights<T> {
    pub(crate) fn new(table: &EsHornerTable) -> Self {
        let width = table.width();
        let rows = table
            .coefficients()
            .chunks_exact(width)
            .map(|row| {
                let mut padded = [T::ZERO; MAX_WIDTH];
                for (padded, &coefficient) in padded.iter_mut().zip(row) {
                    *padded = T::from_f64(coefficient);
                }
                padded
            })
            .collect();
        Self { width, rows }
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Weights of the `width` support cells whose panels share the local
    /// coordinate `t`; panel `j` is the kernel on support cell `j`. The
    /// padding makes every row a fixed-length loop that vectorizes.
    #[inline]
    pub(crate) fn evaluate(&self, t: T) -> [T; MAX_WIDTH] {
        let (first, rest) = self
            .rows
            .split_first()
            .expect("a Horner table has at least one row");
        let mut weights = *first;
        for row in rest {
            for (weight, &coefficient) in weights.iter_mut().zip(row) {
                *weight = *weight * t + coefficient;
            }
        }
        weights
    }
}
