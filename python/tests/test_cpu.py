"""The CPU plans; they need no GPU adapter."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import wgpu_nufft
from direct_sums import direct_type1, direct_type2, direct_type3

# Tolerances and matching point types of each dtype.
DTYPES = {
    "complex64": (1e-5, 1e-3, np.float32),
    "complex128": (1e-10, 1e-8, np.float64),
}


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _points(rng, count, dimensions, dtype):
    shape = (count,) if dimensions == 1 else (count, dimensions)
    return rng.uniform(-np.pi, np.pi, shape).astype(dtype)


def _values(rng, shape, dtype):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_modes", [(11,), (6, 5), (4, 3, 5)])
@pytest.mark.parametrize("isign", [-1, 1])
@pytest.mark.parametrize("mode_order", ["centered", "fft"])
def test_cpu_plans_match_direct_sums(dtype, n_modes, isign, mode_order):
    eps, tolerance, point_type = DTYPES[dtype]
    rng = np.random.default_rng([len(n_modes), isign + 1, len(mode_order)])
    points = _points(rng, 60, len(n_modes), point_type)
    strengths = _values(rng, 60, dtype)
    coefficients = _values(rng, n_modes, dtype)
    options = dict(eps=eps, isign=isign, mode_order=mode_order, dtype=dtype)

    type1 = wgpu_nufft.CpuType1Plan(n_modes, **options)
    modes = type1.execute(points, strengths)
    assert modes.dtype == np.dtype(dtype)
    assert modes.shape == n_modes
    assert modes.flags.c_contiguous
    expected = direct_type1(points, strengths, n_modes, isign, mode_order)
    assert _relative_error(modes, expected) < tolerance

    type2 = wgpu_nufft.CpuType2Plan(n_modes, **options)
    values = type2.execute(points, coefficients)
    assert values.dtype == np.dtype(dtype)
    assert values.shape == (60,)
    expected = direct_type2(points, coefficients, n_modes, isign, mode_order)
    assert _relative_error(values, expected) < tolerance


@pytest.mark.parametrize("dtype", DTYPES)
def test_cpu_type3_matches_direct_sums(dtype):
    eps, tolerance, point_type = DTYPES[dtype]
    rng = np.random.default_rng(3)
    source_bounds = np.array([[-1.0, 2.0], [0.5, 1.5]])
    target_bounds = np.array([[-20.0, 10.0], [-8.0, 8.0]])
    sources = rng.uniform(source_bounds[:, 0], source_bounds[:, 1], (50, 2)).astype(point_type)
    targets = rng.uniform(target_bounds[:, 0], target_bounds[:, 1], (40, 2)).astype(point_type)
    strengths = _values(rng, (2, 50), dtype)
    plan = wgpu_nufft.CpuType3Plan(
        source_bounds, target_bounds, eps=eps, isign=-1, batch=2, dtype=dtype
    )
    actual = plan.execute(sources, strengths, targets)
    assert actual.shape == (2, 40)
    assert actual.dtype == np.dtype(dtype)
    for transform in range(2):
        expected = direct_type3(sources, strengths[transform], targets, -1)
        assert _relative_error(actual[transform], expected) < tolerance

    prepared = plan.prepare_points(sources, targets)
    assert (prepared.source_count, prepared.target_count) == (50, 40)
    assert prepared.dimensions == 2
    np.testing.assert_array_equal(plan.execute(prepared, strengths), actual)
    np.testing.assert_array_equal(
        plan.execute(prepared, strengths[1]), plan.execute(sources, strengths[1], targets)
    )


@pytest.mark.parametrize("n_modes", [(5000,), (40, 36), (12, 10, 14)])
def test_cpu_prepared_points_serve_both_types_bitwise(n_modes):
    rng = np.random.default_rng(7)
    points = _points(rng, 4000, len(n_modes), np.float64)
    strengths = _values(rng, (3, 4000), np.complex128)
    coefficients = _values(rng, (3, *n_modes), np.complex128)
    type1 = wgpu_nufft.CpuType1Plan(n_modes, eps=1e-9, batch=3)
    type2 = wgpu_nufft.CpuType2Plan(n_modes, eps=1e-9, batch=3, threads=3)
    prepared = type1.prepare_points(points)
    assert len(prepared) == 4000
    assert prepared.dimensions == len(n_modes)
    assert prepared.dtype == "complex128"
    for _ in range(2):
        np.testing.assert_array_equal(
            type1.execute(prepared, strengths), type1.execute(points, strengths)
        )
        np.testing.assert_array_equal(
            type2.execute(prepared, coefficients), type2.execute(points, coefficients)
        )
    # Smaller batches reuse the same prepared points.
    np.testing.assert_array_equal(
        type2.execute(prepared, coefficients[0]), type2.execute(points, coefficients[0])
    )


def test_cpu_batches_keep_their_axis():
    rng = np.random.default_rng(11)
    points = _points(rng, 30, 2, np.float32)
    strengths = _values(rng, (3, 30), np.complex64)
    plan = wgpu_nufft.CpuType1Plan((8, 6), eps=1e-5, batch=3, dtype=np.complex64)
    assert plan.dtype == "complex64"
    assert plan.batch_capacity == 3
    batched = plan.execute(points, strengths)
    assert batched.shape == (3, 8, 6)
    for transform in range(3):
        np.testing.assert_array_equal(
            batched[transform], plan.execute(points, strengths[transform])
        )
    assert plan.execute(points, strengths[:1]).shape == (1, 8, 6)
    assert plan.execute(points, strengths[0]).shape == (8, 6)
    with pytest.raises(ValueError, match="plan capacity"):
        plan.execute(points, np.concatenate([strengths, strengths]))


def test_cpu_results_do_not_depend_on_threads_or_layout():
    rng = np.random.default_rng(5)
    points = _points(rng, 5000, 3, np.float64)
    coefficients = _values(rng, (10, 12, 8), np.complex128)
    single = wgpu_nufft.CpuType2Plan((10, 12, 8), threads=1)
    many = wgpu_nufft.CpuType2Plan((10, 12, 8), threads=6)
    assert (single.threads, many.threads) == (1, 6)
    expected = single.execute(points, coefficients)
    np.testing.assert_array_equal(many.execute(points, coefficients), expected)
    # Fortran-ordered and strided inputs are copied into C order.
    np.testing.assert_array_equal(
        many.execute(np.asfortranarray(points), np.asfortranarray(coefficients)), expected
    )
    doubled = np.repeat(points, 2, axis=0)[::2]
    np.testing.assert_array_equal(many.execute(doubled, coefficients), expected)


def test_cpu_plans_run_concurrently():
    rng = np.random.default_rng(13)
    plan = wgpu_nufft.CpuType1Plan((64, 64), eps=1e-6)
    jobs = [(_points(rng, 20_000, 2, np.float64), _values(rng, 20_000, np.complex128))
            for _ in range(4)]
    expected = [plan.execute(points, strengths) for points, strengths in jobs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda job: plan.execute(*job), jobs))
    for result, reference in zip(results, expected):
        np.testing.assert_array_equal(result, reference)


def test_cpu_empty_point_sets():
    type1 = wgpu_nufft.CpuType1Plan((5,))
    empty = np.empty(0)
    np.testing.assert_array_equal(
        type1.execute(empty, np.empty(0, np.complex128)), np.zeros(5, np.complex128)
    )
    prepared = type1.prepare_points(empty)
    assert len(prepared) == 0
    type2 = wgpu_nufft.CpuType2Plan((5,))
    assert type2.execute(prepared, np.ones(5, np.complex128)).shape == (0,)
    bounds = np.array([[-1.0, 1.0]])
    type3 = wgpu_nufft.CpuType3Plan(bounds, bounds)
    np.testing.assert_array_equal(
        type3.execute(empty, np.empty(0, np.complex128), np.array([0.5, -0.25])),
        np.zeros(2, np.complex128),
    )


def test_cpu_validation_errors_are_clear():
    plan = wgpu_nufft.CpuType1Plan((5,))
    values = np.ones(3, np.complex128)
    with pytest.raises(TypeError, match="float64"):
        plan.execute(np.zeros(3, np.float32), values)
    with pytest.raises(TypeError, match="complex128"):
        plan.execute(np.zeros(3), values.astype(np.complex64))
    with pytest.raises(ValueError, match="non-finite"):
        plan.execute(np.array([0.0, np.nan, 1.0]), values)
    with pytest.raises(ValueError, match=r"outside \[-3\*pi, 3\*pi\]"):
        plan.execute(np.array([0.0, 10.0, 1.0]), values)
    with pytest.raises(ValueError, match="shape"):
        plan.execute(np.zeros((3, 2)), values)
    with pytest.raises(ValueError, match="dtype"):
        wgpu_nufft.CpuType1Plan((5,), dtype=np.float64)
    with pytest.raises(ValueError, match="threads"):
        wgpu_nufft.CpuType1Plan((5,), threads=0)
    with pytest.raises(ValueError, match="invalid NUFFT plan"):
        wgpu_nufft.CpuType1Plan((5,), eps=0.0)

    # Prepared points fit only plans of their configuration and dtype.
    prepared = plan.prepare_points(np.zeros(3))
    with pytest.raises(ValueError, match="does not fit"):
        wgpu_nufft.CpuType1Plan((64,)).execute(prepared, values)
    with pytest.raises(ValueError, match="prepared for complex128"):
        wgpu_nufft.CpuType1Plan((5,), dtype="complex64").execute(
            prepared, values.astype(np.complex64)
        )

    bounds = np.array([[-1.0, 1.0]])
    type3 = wgpu_nufft.CpuType3Plan(bounds, bounds)
    with pytest.raises(ValueError, match="outside"):
        type3.execute(np.array([1.1]), np.ones(1, np.complex128), np.array([0.0]))
    with pytest.raises(ValueError, match="target_points"):
        type3.execute(np.array([0.5]), np.ones(1, np.complex128))
    points3 = type3.prepare_points(np.array([0.5]), np.array([0.0]))
    with pytest.raises(ValueError, match="omitted"):
        type3.execute(points3, np.ones(1, np.complex128), np.array([0.0]))


def test_cpu_plan_metadata():
    plan = wgpu_nufft.CpuType2Plan((16, 12), eps=1e-6, isign=-1, mode_order="fft")
    assert plan.n_modes == [16, 12]
    assert plan.dimensions == 2
    assert (plan.eps, plan.isign, plan.mode_order) == (1e-6, -1, "fft")
    assert plan.dtype == "complex128"
    assert all(fine >= modes for fine, modes in zip(plan.fine_grid_shape, plan.n_modes))
    assert plan.threads >= 1
    bounds = np.array([[-1.0, 1.0], [0.0, 2.0]])
    type3 = wgpu_nufft.CpuType3Plan(bounds, bounds * 10, batch=2)
    assert type3.source_bounds == [(-1.0, 1.0), (0.0, 2.0)]
    assert type3.target_bounds == [(-10.0, 10.0), (0.0, 20.0)]
    assert len(type3.outer_grid_shape) == 2
    assert type3.batch_capacity == 2
