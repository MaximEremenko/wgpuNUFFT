//! Type-1 spreading onto the fine grid and type-2 interpolation from it.
//!
//! Both work on runs of `RUN` complex cells along axis zero, where `RUN` is
//! the kernel width rounded up to 4, 8, or 16 and the extra weights are
//! zero. Fixed-length runs vectorize and keep interpolation sums in
//! registers; a run that would wrap around the end of a row takes a slower
//! cell-by-cell path.

use super::grid::{FineGrid, Weights, MAX_WIDTH};
use super::points::{Slabs, SortedPoints};
use super::real::{Coordinate, Real};
use super::workers::{for_each_chunk_group, run_tasks};
use super::zeroed;
use crate::error::Result;

/// Points gathered together before their supports are computed, so that
/// their scattered loads overlap.
const BLOCK: usize = 64;
/// Points per thread below which interpolation stays on one thread.
const MIN_POINTS_PER_THREAD: usize = 1 << 11;
/// Grids with fewer slabs than this may spread through private copies.
const DENSE_MAX_SLABS: usize = 32;
/// Points per private copy of a dense grid.
const DENSE_POINTS_PER_GROUP: usize = 1 << 14;
const DENSE_MAX_GROUPS: usize = 64;
/// Cells that all private copies may hold together.
const DENSE_MAX_CELLS: usize = 1 << 24;
/// Cells per thread below which the copies are summed on one thread.
const MIN_CELLS_PER_THREAD: usize = 1 << 15;

/// A complex value, real part first.
type Cell<T> = [T; 2];

/// The points of one execution and the fine grid they meet.
pub(crate) struct Points<'a, T, P, const D: usize> {
    pub(crate) grid: &'a FineGrid<D>,
    pub(crate) weights: &'a Weights<T>,
    /// Point-major coordinates.
    pub(crate) coordinates: &'a [P],
    pub(crate) count: usize,
    pub(crate) batch: usize,
}

/// First support cell and kernel weights of one point on every axis; the
/// weights past the kernel width are zero.
struct Support<T, const D: usize> {
    start: [usize; D],
    weights: [[T; MAX_WIDTH]; D],
}

/// A thread's scratch space: one block of gathered points.
struct Block<T> {
    coordinates: Vec<f64>,
    /// Strengths of the block's points, one per transform.
    strengths: Vec<Cell<T>>,
}

impl<T: Real, P: Coordinate, const D: usize> Points<'_, T, P, D> {
    fn width(&self) -> usize {
        self.weights.width()
    }

    fn block(&self) -> Block<T> {
        Block {
            coordinates: vec![0.0; BLOCK * D],
            strengths: vec![[T::ZERO; 2]; BLOCK * self.batch],
        }
    }

    /// Copies the coordinates of `points` into `block`, `D` per point.
    fn gather(&self, points: &[u32], block: &mut Block<T>) {
        for (&point, target) in points.iter().zip(block.coordinates.chunks_exact_mut(D)) {
            let source = &self.coordinates[point as usize * D..(point as usize + 1) * D];
            for (target, &coordinate) in target.iter_mut().zip(source) {
                *target = coordinate.to_f64();
            }
        }
    }

    #[inline]
    fn support(&self, coordinates: &[f64]) -> Support<T, D> {
        let mut support = Support {
            start: [0; D],
            weights: [[T::ZERO; MAX_WIDTH]; D],
        };
        for (axis, &coordinate) in coordinates.iter().enumerate() {
            let (start, t) = self.grid.support(axis, coordinate);
            support.start[axis] = start;
            support.weights[axis] = self.weights.evaluate(T::from_f64(t));
        }
        support
    }
}

/// Spreads the transform-major `strengths` onto `fine`, which holds every
/// transform's grid in turn.
///
/// Each slab of the fine grid is one task: it zeroes its cells, then adds
/// the points of the previous slab that reach into it and then its own
/// points, each in bin order. Every cell therefore sums its points in the
/// same order, whichever thread runs the task.
pub(crate) fn spread<T: Real, P: Coordinate, const D: usize, const RUN: usize>(
    points: &Points<'_, T, P, D>,
    strengths: &[Cell<T>],
    slabs: &Slabs,
    sorted: &SortedPoints,
    fine: &mut [Cell<T>],
    threads: usize,
) {
    let grid = points.grid;
    let slab_count = slabs.count();
    let mut transforms = fine
        .chunks_mut(grid.cells())
        .map(|transform| slabs.split(transform, grid.strides[D - 1]).into_iter())
        .collect::<Vec<_>>();
    let tasks = (0..slab_count)
        .map(|slab| {
            let grids = transforms
                .iter_mut()
                .map(|parts| parts.next().expect("every transform has every slab"))
                .collect::<Vec<_>>();
            (slab, grids)
        })
        .collect::<Vec<_>>();
    run_tasks(
        threads,
        tasks,
        || points.block(),
        |block, (slab, mut grids)| {
            for grid in &mut grids {
                grid.fill([T::ZERO; 2]);
            }
            let (low, high) = slabs.range(slab);
            let previous = (slab + slab_count - 1) % slab_count;
            let incoming = if previous == slab {
                &[][..]
            } else {
                sorted.crossing(previous)
            };
            for list in [incoming, sorted.slab(slab)] {
                for chunk in list.chunks(BLOCK) {
                    points.gather(chunk, block);
                    gather_strengths(points, strengths, chunk, block);
                    for index in 0..chunk.len() {
                        spread_point::<T, P, D, RUN>(points, block, index, low, high, &mut grids);
                    }
                }
            }
        },
    );
}

/// Adds point `index` of `block` to the cells of the slab `low..high` along
/// the last axis. `grids` holds that slab of every transform's grid.
#[inline]
fn spread_point<T: Real, P: Coordinate, const D: usize, const RUN: usize>(
    points: &Points<'_, T, P, D>,
    block: &Block<T>,
    index: usize,
    low: usize,
    high: usize,
    grids: &mut [&mut [Cell<T>]],
) {
    let grid = points.grid;
    let width = points.width();
    let support = points.support(&block.coordinates[index * D..(index + 1) * D]);
    let strengths = &block.strengths[index * points.batch..(index + 1) * points.batch];
    let x_weights = run_weights::<T, RUN>(&support.weights[0]);
    let x_start = support.start[0];

    if D == 1 {
        for (grid, &strength) in grids.iter_mut().zip(strengths) {
            let values = x_weights.map(|weight| [weight * strength[0], weight * strength[1]]);
            if low <= x_start && x_start + RUN <= high {
                add_cells(
                    &mut grid[x_start - low..x_start - low + RUN],
                    &values,
                    T::ONE,
                );
            } else {
                for (offset, value) in values[..width].iter().enumerate() {
                    let mut cell = x_start + offset;
                    if cell >= points.grid.shape[0] {
                        cell -= points.grid.shape[0];
                    }
                    if cell >= low && cell < high {
                        let target = &mut grid[cell - low];
                        target[0] += value[0];
                        target[1] += value[1];
                    }
                }
            }
        }
        return;
    }

    let last = D - 1;
    let length = grid.shape[last];
    let row_len = grid.shape[0];
    let plane = grid.strides[last];
    for (grid_slab, &strength) in grids.iter_mut().zip(strengths) {
        let values = x_weights.map(|weight| [weight * strength[0], weight * strength[1]]);
        for (offset, &weight) in support.weights[last][..width].iter().enumerate() {
            let mut cell = support.start[last] + offset;
            if cell >= length {
                cell -= length;
            }
            if cell < low || cell >= high {
                continue;
            }
            let plane_base = (cell - low) * plane;
            for_each_row(grid, &support, width, weight, |row, scale| {
                let row = plane_base + row;
                if x_start + RUN <= row_len {
                    add_cells(
                        &mut grid_slab[row + x_start..row + x_start + RUN],
                        &values,
                        scale,
                    );
                } else {
                    for (offset, value) in values[..width].iter().enumerate() {
                        let mut x = x_start + offset;
                        if x >= row_len {
                            x -= row_len;
                        }
                        let target = &mut grid_slab[row + x];
                        target[0] += value[0] * scale;
                        target[1] += value[1] * scale;
                    }
                }
            });
        }
    }
}

/// The number of private grid copies to spread through, if the grid has too
/// few slabs to keep threads busy, at least as many points as cells, and
/// room for more copies than it has slabs.
///
/// It depends on the problem alone, never on the thread count, because it
/// fixes the order in which every cell sums its points.
pub(crate) fn dense_groups(slab_count: usize, point_count: usize, cells: usize) -> Option<usize> {
    if slab_count >= DENSE_MAX_SLABS || point_count < cells {
        return None;
    }
    let groups = (point_count / DENSE_POINTS_PER_GROUP)
        .min(DENSE_MAX_GROUPS)
        .min(DENSE_MAX_CELLS / cells.max(1));
    (groups >= 2 && groups > slab_count).then_some(groups)
}

/// Spreads like [`spread`] onto a small grid that holds many points: the
/// points, in sorted order, are cut into `groups` fixed runs, each spread
/// onto its own zeroed copy of the grid, and the copies are added in group
/// order.
pub(crate) fn spread_dense<T: Real, P: Coordinate, const D: usize, const RUN: usize>(
    points: &Points<'_, T, P, D>,
    strengths: &[Cell<T>],
    sorted: &SortedPoints,
    groups: usize,
    fine: &mut [Cell<T>],
    threads: usize,
) -> Result<()> {
    let grid = points.grid;
    let copy_cells = grid.cells() * points.batch;
    let mut copies = zeroed::<Cell<T>>("CPU private fine-grid copies", copy_cells * groups)?;
    let order = &sorted.order;
    let per_group = order.len().div_ceil(groups);
    let tasks = copies
        .chunks_mut(copy_cells)
        .enumerate()
        .collect::<Vec<_>>();
    let length = grid.shape[D - 1];
    run_tasks(
        threads,
        tasks,
        || points.block(),
        |block, (group, copy)| {
            let mut grids = copy.chunks_mut(grid.cells()).collect::<Vec<_>>();
            let first = (group * per_group).min(order.len());
            let last = ((group + 1) * per_group).min(order.len());
            for chunk in order[first..last].chunks(BLOCK) {
                points.gather(chunk, block);
                gather_strengths(points, strengths, chunk, block);
                for index in 0..chunk.len() {
                    spread_point::<T, P, D, RUN>(points, block, index, 0, length, &mut grids);
                }
            }
        },
    );
    let copies = &copies;
    for_each_chunk_group(threads, fine, 1, MIN_CELLS_PER_THREAD, |first, cells| {
        let len = cells.len();
        cells.copy_from_slice(&copies[first..first + len]);
        for group in 1..groups {
            let start = group * copy_cells + first;
            for (cell, copy) in cells.iter_mut().zip(&copies[start..start + len]) {
                cell[0] += copy[0];
                cell[1] += copy[1];
            }
        }
    });
    Ok(())
}

/// Copies the strengths of `chunk`'s points for every transform into
/// `block`.
fn gather_strengths<T: Real, P: Coordinate, const D: usize>(
    points: &Points<'_, T, P, D>,
    strengths: &[Cell<T>],
    chunk: &[u32],
    block: &mut Block<T>,
) {
    for (&point, targets) in chunk
        .iter()
        .zip(block.strengths.chunks_exact_mut(points.batch))
    {
        for (transform, target) in targets.iter_mut().enumerate() {
            *target = strengths[transform * points.count + point as usize];
        }
    }
}

/// Interpolates every transform's grid in `fine` at the points, in sorted
/// order, into point-major `values`: one value per transform and point.
/// Each value is one sum in a fixed order.
pub(crate) fn interpolate<T: Real, P: Coordinate, const D: usize, const RUN: usize>(
    points: &Points<'_, T, P, D>,
    fine: &[Cell<T>],
    sorted: &SortedPoints,
    values: &mut [Cell<T>],
    threads: usize,
) {
    let batch = points.batch;
    for_each_chunk_group(
        threads,
        values,
        batch,
        MIN_POINTS_PER_THREAD,
        |first, values| {
            let mut block = points.block();
            let order = &sorted.order[first..first + values.len() / batch];
            for (chunk, outputs) in order.chunks(BLOCK).zip(values.chunks_mut(BLOCK * batch)) {
                points.gather(chunk, &mut block);
                for (index, output) in outputs.chunks_exact_mut(batch).enumerate() {
                    interpolate_point::<T, P, D, RUN>(points, fine, &block, index, output);
                }
            }
        },
    );
}

#[inline]
fn interpolate_point<T: Real, P: Coordinate, const D: usize, const RUN: usize>(
    points: &Points<'_, T, P, D>,
    fine: &[Cell<T>],
    block: &Block<T>,
    index: usize,
    output: &mut [Cell<T>],
) {
    let grid = points.grid;
    let width = points.width();
    let support = points.support(&block.coordinates[index * D..(index + 1) * D]);
    let x_weights = run_weights::<T, RUN>(&support.weights[0]);
    let x_start = support.start[0];
    let cells = grid.cells();
    let row_len = grid.shape[0];

    for (transform, value) in output.iter_mut().enumerate() {
        let grid_cells = &fine[transform * cells..(transform + 1) * cells];
        let mut sums = [[T::ZERO; 2]; RUN];
        if D == 1 {
            gather_cells(grid_cells, 0, row_len, x_start, width, &mut sums, T::ONE);
        } else {
            let last = D - 1;
            let length = grid.shape[last];
            let plane = grid.strides[last];
            for (offset, &weight) in support.weights[last][..width].iter().enumerate() {
                let mut cell = support.start[last] + offset;
                if cell >= length {
                    cell -= length;
                }
                let plane_base = cell * plane;
                for_each_row(grid, &support, width, weight, |row, scale| {
                    gather_cells(
                        grid_cells,
                        plane_base + row,
                        row_len,
                        x_start,
                        width,
                        &mut sums,
                        scale,
                    );
                });
            }
        }
        let (mut re, mut im) = (T::ZERO, T::ZERO);
        for (sum, &weight) in sums.iter().zip(&x_weights) {
            re += sum[0] * weight;
            im += sum[1] * weight;
        }
        *value = [re, im];
    }
}

/// The first `RUN` axis-zero weights; those past the kernel width are zero.
#[inline]
fn run_weights<T: Real, const RUN: usize>(weights: &[T; MAX_WIDTH]) -> [T; RUN] {
    let mut run = [T::ZERO; RUN];
    run.copy_from_slice(&weights[..RUN]);
    run
}

/// Adds `values * scale` to `RUN` consecutive cells.
#[inline]
fn add_cells<T: Real, const RUN: usize>(cells: &mut [Cell<T>], values: &[Cell<T>; RUN], scale: T) {
    let cells: &mut [Cell<T>; RUN] = cells.try_into().expect("a run of RUN cells");
    for (cell, value) in cells.iter_mut().zip(values) {
        cell[0] += value[0] * scale;
        cell[1] += value[1] * scale;
    }
}

/// Adds the support cells from `start` of the row starting at cell `row`,
/// times `scale`, to `sums`, wrapping around the end of the row.
#[inline]
fn gather_cells<T: Real, const RUN: usize>(
    grid: &[Cell<T>],
    row: usize,
    row_len: usize,
    start: usize,
    width: usize,
    sums: &mut [Cell<T>; RUN],
    scale: T,
) {
    if start + RUN <= row_len {
        let cells: &[Cell<T>; RUN] = grid[row + start..row + start + RUN]
            .try_into()
            .expect("a run of RUN cells");
        for (sum, cell) in sums.iter_mut().zip(cells) {
            sum[0] += cell[0] * scale;
            sum[1] += cell[1] * scale;
        }
    } else {
        for (offset, sum) in sums[..width].iter_mut().enumerate() {
            let mut x = start + offset;
            if x >= row_len {
                x -= row_len;
            }
            let cell = grid[row + x];
            sum[0] += cell[0] * scale;
            sum[1] += cell[1] * scale;
        }
    }
}

/// Calls `visit` with the first cell and weight of every support row: the
/// cells along axis zero at one support index of each axis from 1 to
/// `D - 2`. `scale` is the weight of the caller's last-axis index.
#[inline]
fn for_each_row<T: Real, const D: usize>(
    grid: &FineGrid<D>,
    support: &Support<T, D>,
    width: usize,
    scale: T,
    mut visit: impl FnMut(usize, T),
) {
    if D <= 2 {
        visit(0, scale);
        return;
    }
    if D == 3 {
        let axis = 1;
        let (start, length, stride) = (support.start[axis], grid.shape[axis], grid.strides[axis]);
        for (offset, &weight) in support.weights[axis][..width].iter().enumerate() {
            let mut index = start + offset;
            if index >= length {
                index -= length;
            }
            visit(index * stride, scale * weight);
        }
        return;
    }
    let mut offsets = [0usize; D];
    loop {
        let mut row = 0;
        let mut weight = scale;
        for (axis, &offset) in offsets.iter().enumerate().take(D - 1).skip(1) {
            let mut index = support.start[axis] + offset;
            if index >= grid.shape[axis] {
                index -= grid.shape[axis];
            }
            row += index * grid.strides[axis];
            weight = weight * support.weights[axis][offset];
        }
        visit(row, weight);
        let mut axis = 1;
        loop {
            offsets[axis] += 1;
            if offsets[axis] < width {
                break;
            }
            offsets[axis] = 0;
            axis += 1;
            if axis == D - 1 {
                return;
            }
        }
    }
}
