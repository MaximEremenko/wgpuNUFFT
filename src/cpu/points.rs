//! Slabs of the fine grid and points sorted into coarse bins.

use std::sync::atomic::{AtomicU32, Ordering};

use super::grid::FineGrid;
use super::real::Coordinate;
use super::workers::{map_parts, map_ranges, ranges, split_ranges};
use super::{ensure, filled, MAX_COORDINATE_MAGNITUDE};
use crate::error::{NufftError, Result};

/// Bin edge along axis zero, in cells.
const AXIS0_BIN: usize = 32;
/// Bin edge along the other axes before the last, in cells.
const BIN: usize = 16;
/// Points per thread below which sorting stays on one thread.
const MIN_POINTS_PER_THREAD: usize = 1 << 14;

/// Slabs of the fine grid along its last axis. Each is at least
/// `width - 1` cells thick, so a kernel support reaches at most two
/// consecutive slabs.
pub(crate) struct Slabs {
    /// Slab `s` covers last-axis cells `bounds[s]..bounds[s + 1]`.
    bounds: Vec<usize>,
    /// Slab of every last-axis cell.
    of_cell: Vec<u32>,
    width: usize,
}

impl Slabs {
    /// Splits `length` cells into slabs about `thickness` cells thick.
    pub(crate) fn new(length: usize, width: usize, thickness: usize) -> Result<Self> {
        let thickness = thickness.max(width.saturating_sub(1)).max(1);
        let count = (length / thickness).max(1);
        let bounds = (0..=count)
            .map(|slab| (slab as u64 * length as u64 / count as u64) as usize)
            .collect::<Vec<_>>();
        let mut of_cell = filled("CPU slab of every last-axis cell", length, 0u32)?;
        for (slab, pair) in bounds.windows(2).enumerate() {
            of_cell[pair[0]..pair[1]].fill(slab as u32);
        }
        Ok(Self {
            bounds,
            of_cell,
            width,
        })
    }

    pub(crate) fn count(&self) -> usize {
        self.bounds.len() - 1
    }

    /// Last-axis cells of `slab`.
    pub(crate) fn range(&self, slab: usize) -> (usize, usize) {
        (self.bounds[slab], self.bounds[slab + 1])
    }

    /// Slab of a support starting at last-axis cell `start`, and whether the
    /// support reaches into the next slab.
    fn locate(&self, start: usize) -> (usize, bool) {
        let slab = self.of_cell[start] as usize;
        let crosses = self.count() > 1 && start + self.width > self.bounds[slab + 1];
        (slab, crosses)
    }

    /// Splits one transform's grid into its slabs, `unit` words per
    /// last-axis cell.
    pub(crate) fn split<'a, T>(&self, mut data: &'a mut [T], unit: usize) -> Vec<&'a mut [T]> {
        let mut slabs = Vec::with_capacity(self.count());
        for pair in self.bounds.windows(2) {
            let (slab, rest) = std::mem::take(&mut data).split_at_mut((pair[1] - pair[0]) * unit);
            slabs.push(slab);
            data = rest;
        }
        slabs
    }
}

/// Point indices sorted into coarse bins. Bins run slab by slab along the
/// last axis. Within a slab, the points whose support stays in the slab come
/// first, then those that reach into the next slab; each group runs over
/// tiles of the other axes with axis zero fastest. Points keep their input
/// order within a bin.
///
/// The indices are atomics only so that threads can scatter them without
/// `unsafe`; every slot is written once, and relaxed loads are plain loads.
pub(crate) struct SortedPoints {
    buffers: SortBuffers,
    count: usize,
    /// Slab `s` holds `order[starts[s]..starts[s + 1]]`.
    starts: Vec<usize>,
    /// The points of slab `s` that reach into the next slab start at
    /// `crossing[s]`.
    crossing: Vec<usize>,
}

/// The allocations of a sort, kept between executions so that they are
/// made, and their pages first touched, once.
#[derive(Default)]
pub(crate) struct SortBuffers {
    keys: Vec<u32>,
    order: Vec<AtomicU32>,
    ranks: Vec<AtomicU32>,
}

impl SortedPoints {
    /// Validates the coordinates and sorts the points by the bins of their
    /// first support cells, in `buffers`.
    pub(crate) fn new<P: Coordinate, const D: usize>(
        grid: &FineGrid<D>,
        slabs: &Slabs,
        coordinates: &[P],
        threads: usize,
        mut buffers: SortBuffers,
    ) -> Result<Self> {
        let count = coordinates.len() / D;
        ensure(&mut buffers.keys, count, "CPU point bin keys")?;
        ensure(&mut buffers.order, count, "CPU sorted point order")?;
        let mut tiles = [1usize; D];
        for (axis, tile_count) in tiles.iter_mut().enumerate().take(D - 1) {
            *tile_count = grid.shape[axis].div_ceil(bin_edge(axis));
        }
        let tiles_per_group = tiles[..D - 1].iter().product::<usize>();
        let bin_count = slabs.count() * 2 * tiles_per_group;
        if u32::try_from(bin_count).is_err() {
            return Err(NufftError::LengthOverflow {
                context: "CPU point bin count",
            });
        }
        let key = |point: &[P]| -> u32 {
            let mut tile = 0;
            for axis in (0..D - 1).rev() {
                let (start, _) = grid.support(axis, point[axis].to_f64());
                tile = tile * tiles[axis] + start / bin_edge(axis);
            }
            let (start, _) = grid.support(D - 1, point[D - 1].to_f64());
            let (slab, crosses) = slabs.locate(start);
            ((2 * slab + usize::from(crosses)) * tiles_per_group + tile) as u32
        };

        // Each range computes its keys and histogram, and reports its first
        // invalid coordinate.
        let point_ranges = ranges(count, threads, MIN_POINTS_PER_THREAD);
        let key_parts = split_ranges(&mut buffers.keys[..count], &point_ranges);
        let histograms = map_parts(
            key_parts
                .into_iter()
                .zip(point_ranges.iter().cloned())
                .collect(),
            |(keys, range)| {
                let mut histogram = vec![0u32; bin_count];
                for (point, slot) in range.zip(keys) {
                    let point_coordinates = &coordinates[point * D..(point + 1) * D];
                    if let Some(error) = invalid_coordinate(point, point_coordinates) {
                        return Err(error);
                    }
                    let key = key(point_coordinates);
                    histogram[key as usize] += 1;
                    *slot = key;
                }
                Ok(histogram)
            },
        )
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

        // Turn the histograms into each range's first slot in every bin, so
        // that the scatter keeps input order within a bin.
        let mut bin_starts = filled("CPU point bin offsets", bin_count + 1, 0usize)?;
        let mut cursors = histograms;
        let mut next = 0usize;
        for bin in 0..bin_count {
            bin_starts[bin] = next;
            for cursor in &mut cursors {
                let size = cursor[bin] as usize;
                cursor[bin] = next as u32;
                next += size;
            }
        }
        bin_starts[bin_count] = next;

        let (keys, order) = (&buffers.keys[..count], &buffers.order[..count]);
        map_ranges(&point_ranges, |group, range| {
            let mut cursor = cursors[group].clone();
            for point in range {
                let slot = &mut cursor[keys[point] as usize];
                order[*slot as usize].store(point as u32, Ordering::Relaxed);
                *slot += 1;
            }
        });

        let starts = (0..=slabs.count())
            .map(|slab| bin_starts[2 * slab * tiles_per_group])
            .collect();
        let crossing = (0..slabs.count())
            .map(|slab| bin_starts[(2 * slab + 1) * tiles_per_group])
            .collect();
        Ok(Self {
            buffers,
            count,
            starts,
            crossing,
        })
    }

    /// Returns the allocations for the next sort.
    pub(crate) fn into_buffers(self) -> SortBuffers {
        self.buffers
    }

    /// Every point, in bin order.
    pub(crate) fn all(&self) -> &[AtomicU32] {
        &self.buffers.order[..self.count]
    }

    /// Points whose support starts in `slab`, in bin order.
    pub(crate) fn slab(&self, slab: usize) -> &[AtomicU32] {
        &self.buffers.order[self.starts[slab]..self.starts[slab + 1]]
    }

    /// Points of `slab` whose support reaches into the next slab, in bin
    /// order.
    pub(crate) fn crossing(&self, slab: usize) -> &[AtomicU32] {
        &self.buffers.order[self.crossing[slab]..self.starts[slab + 1]]
    }

    /// Position of every point in the sorted order.
    pub(crate) fn ranks(&mut self, threads: usize) -> Result<&[AtomicU32]> {
        let count = self.count;
        ensure(&mut self.buffers.ranks, count, "CPU point ranks")?;
        let (order, ranks) = (&self.buffers.order[..count], &self.buffers.ranks[..count]);
        map_ranges(
            &ranges(count, threads, MIN_POINTS_PER_THREAD),
            |_, range| {
                for rank in range {
                    let point = order[rank].load(Ordering::Relaxed) as usize;
                    ranks[point].store(rank as u32, Ordering::Relaxed);
                }
            },
        );
        Ok(ranks)
    }
}

/// Checks every coordinate against the type-1 and type-2 contract, in
/// parallel, reporting the first failure.
pub(crate) fn validate_points<P: Coordinate, const D: usize>(
    coordinates: &[P],
    threads: usize,
) -> Result<()> {
    let count = coordinates.len() / D;
    map_ranges(
        &ranges(count, threads, MIN_POINTS_PER_THREAD),
        |_, range| {
            for point in range {
                if let Some(error) =
                    invalid_coordinate(point, &coordinates[point * D..(point + 1) * D])
                {
                    return Err(error);
                }
            }
            Ok(())
        },
    )
    .into_iter()
    .collect::<Result<Vec<()>>>()?;
    Ok(())
}

/// The error for the first coordinate of `point` outside the type-1 and
/// type-2 contract, if any.
fn invalid_coordinate<P: Coordinate>(point: usize, coordinates: &[P]) -> Option<NufftError> {
    coordinates
        .iter()
        .enumerate()
        .find_map(|(axis, &coordinate)| {
            let value = coordinate.to_f64();
            if !value.is_finite() {
                Some(NufftError::NonFiniteCoordinate { point, axis, value })
            } else if value.abs() > MAX_COORDINATE_MAGNITUDE {
                Some(NufftError::PointOutOfRange {
                    point,
                    axis,
                    value,
                    maximum_magnitude: MAX_COORDINATE_MAGNITUDE,
                })
            } else {
                None
            }
        })
}

fn bin_edge(axis: usize) -> usize {
    if axis == 0 {
        AXIS0_BIN
    } else {
        BIN
    }
}
