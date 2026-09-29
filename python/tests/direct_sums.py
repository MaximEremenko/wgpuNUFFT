"""Direct O(M N) sums that the NUFFT tests compare against."""

import numpy as np


def point_matrix(points):
    points = np.asarray(points, dtype=np.float64)
    return points[:, None] if points.ndim == 1 else points


def mode_axes(n_modes, mode_order):
    axes = []
    for length in n_modes:
        if mode_order == "centered":
            axes.append(np.arange(-(length // 2), (length - 1) // 2 + 1))
        else:
            axes.append(np.rint(np.fft.fftfreq(length) * length).astype(np.int64))
    return axes


def direct_type1(points, strengths, n_modes, isign, mode_order):
    points = point_matrix(points)
    axes = mode_axes(n_modes, mode_order)
    output = np.empty(n_modes, dtype=np.complex128)
    for index in np.ndindex(*n_modes):
        mode = np.array([axes[axis][position] for axis, position in enumerate(index)])
        output[index] = np.sum(
            np.asarray(strengths, dtype=np.complex128)
            * np.exp(1j * isign * (points @ mode))
        )
    return output


def direct_type2(points, coefficients, n_modes, isign, mode_order):
    points = point_matrix(points)
    axes = mode_axes(n_modes, mode_order)
    output = np.zeros(points.shape[0], dtype=np.complex128)
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    for index in np.ndindex(*n_modes):
        mode = np.array([axes[axis][position] for axis, position in enumerate(index)])
        output += coefficients[index] * np.exp(1j * isign * (points @ mode))
    return output


def direct_type3(source_points, strengths, target_points, isign):
    source_points = point_matrix(source_points)
    target_points = point_matrix(target_points)
    phase = target_points @ source_points.T
    return np.exp(1j * isign * phase) @ np.asarray(strengths, dtype=np.complex128)
