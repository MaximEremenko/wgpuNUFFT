import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import wgpu_nufft
from direct_sums import direct_type1, direct_type2, direct_type3


@pytest.fixture(scope="session")
def context():
    return wgpu_nufft.Context()


def test_package_metadata_and_exception_round_trip():
    assert wgpu_nufft.__version__ == "0.3.2"
    assert wgpu_nufft.WgpuNufftError.__module__ == "wgpu_nufft"
    error = pickle.loads(pickle.dumps(wgpu_nufft.WgpuNufftError("test error")))
    assert isinstance(error, wgpu_nufft.WgpuNufftError)
    assert str(error) == "test error"
    assert wgpu_nufft.Context.__doc__
    assert wgpu_nufft.Type1Plan.execute.__doc__


def test_gpu_resources_shutdown_cleanly_in_fresh_process():
    script = """
import numpy as np
import wgpu_nufft

context = wgpu_nufft.Context()
plan = wgpu_nufft.Type1Plan(context, (8,), eps=1e-4)
points = np.array([-0.5, 0.25], dtype=np.float32)
strengths = np.array([1.0 + 0.2j, -0.3 + 0.4j], dtype=np.complex64)
plan.execute(points, strengths)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_gpu_plans_can_be_destroyed_on_another_thread():
    script = """
import gc
import queue
import threading

import wgpu_nufft

objects = queue.Queue()
stop = object()

def worker():
    while True:
        obj = objects.get()
        if obj is stop:
            objects.task_done()
            return
        del obj
        gc.collect()
        objects.task_done()

thread = threading.Thread(target=worker)
thread.start()
for _ in range(8):
    context = wgpu_nufft.Context()
    plan = wgpu_nufft.Type1Plan(context, (8,), eps=1e-4)
    del context
    objects.put(plan)
    del plan
    objects.join()
objects.put(stop)
objects.join()
thread.join()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def _assert_gpu_close(actual, expected):
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3 * scale)


@pytest.mark.parametrize("isign", [-1, 1])
@pytest.mark.parametrize("mode_order", ["centered", "fft"])
def test_type1_and_type2_1d_match_direct(context, isign, mode_order):
    points = np.array([-2.1, -0.4, 0.0, 0.7, 2.4], dtype=np.float32)
    strengths = np.array(
        [1.0 + 0.2j, -0.4 + 0.7j, 0.3 - 0.1j, 1.2 + 0.5j, -0.2 - 0.8j],
        dtype=np.complex64,
    )
    coefficients = np.array(
        [0.1 + 0.3j, -0.2 + 0.4j, 0.8 - 0.1j, 0.5 + 0.2j,
         -0.7 + 0.6j, 0.2 - 0.5j, 0.9 + 0.0j],
        dtype=np.complex64,
    )
    n_modes = (7,)

    type1 = wgpu_nufft.Type1Plan(
        context, n_modes, eps=1e-4, isign=isign, mode_order=mode_order
    )
    actual1 = type1.execute(points, strengths)
    expected1 = direct_type1(points, strengths, n_modes, isign, mode_order)
    assert actual1.dtype == np.complex64
    assert actual1.shape == n_modes
    _assert_gpu_close(actual1, expected1)

    type2 = wgpu_nufft.Type2Plan(
        context, n_modes, eps=1e-4, isign=isign, mode_order=mode_order
    )
    actual2 = type2.execute(points, coefficients)
    expected2 = direct_type2(points, coefficients, n_modes, isign, mode_order)
    assert actual2.dtype == np.complex64
    assert actual2.shape == (points.size,)
    _assert_gpu_close(actual2, expected2)


def test_non_square_2d_numpy_axis_order(context):
    points = np.array(
        [[-1.2, 0.4], [0.1, -0.7], [1.0, 1.3], [2.2, -1.8]], dtype=np.float32
    )
    strengths = np.array(
        [1.0 + 0.1j, -0.3 + 0.8j, 0.5 - 0.2j, -0.7 - 0.4j],
        dtype=np.complex64,
    )
    n_modes = (3, 5)
    type1 = wgpu_nufft.Type1Plan(context, n_modes, eps=1e-4)
    modes = type1.execute(points, strengths)
    assert modes.shape == n_modes
    _assert_gpu_close(modes, direct_type1(points, strengths, n_modes, 1, "centered"))

    coefficients = (
        np.arange(15, dtype=np.float32).reshape(n_modes) / 10
        + 1j * np.flip(np.arange(15, dtype=np.float32).reshape(n_modes)) / 20
    ).astype(np.complex64)
    type2 = wgpu_nufft.Type2Plan(context, n_modes, eps=1e-4)
    values = type2.execute(points, coefficients)
    _assert_gpu_close(values, direct_type2(points, coefficients, n_modes, 1, "centered"))


@pytest.mark.parametrize("n_modes", [(3, 2, 4, 3), (2, 3, 2, 2, 3, 2)])
def test_higher_rank_numpy_axis_order(context, n_modes):
    rng = np.random.default_rng(len(n_modes))
    points = rng.uniform(-np.pi, np.pi, (9, len(n_modes))).astype(np.float32)
    strengths = (rng.standard_normal(9) + 1j * rng.standard_normal(9)).astype(np.complex64)
    type1 = wgpu_nufft.Type1Plan(context, n_modes, eps=1e-4, isign=-1)
    modes = type1.execute(points, strengths)
    assert modes.shape == n_modes
    _assert_gpu_close(modes, direct_type1(points, strengths, n_modes, -1, "centered"))

    coefficients = (
        rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)
    ).astype(np.complex64)
    type2 = wgpu_nufft.Type2Plan(context, n_modes, eps=1e-4, isign=-1)
    values = type2.execute(points, coefficients)
    assert values.shape == (9,)
    _assert_gpu_close(values, direct_type2(points, coefficients, n_modes, -1, "centered"))


def test_batched_reuse_and_explicit_batch_axis(context):
    points = np.array([-1.3, -0.2, 0.5, 1.7], dtype=np.float32)
    strengths = np.array(
        [
            [1.0 + 0.0j, 0.2 + 0.4j, -0.3 + 0.1j, 0.7 - 0.5j],
            [-0.2 + 0.8j, 0.9 + 0.1j, 0.1 - 0.6j, 0.4 + 0.3j],
            [0.3 - 0.2j, -0.5 + 0.7j, 1.1 + 0.0j, -0.1 - 0.4j],
        ],
        dtype=np.complex64,
    )
    plan = wgpu_nufft.Type1Plan(context, (6,), eps=1e-4, batch=3)

    batched = plan.execute(points, strengths)
    assert batched.shape == (3, 6)
    for transform in range(3):
        _assert_gpu_close(
            batched[transform],
            direct_type1(points, strengths[transform], (6,), 1, "centered"),
        )

    single = plan.execute(points, strengths[0])
    repeated = plan.execute(points, strengths[0:1])
    assert single.shape == (6,)
    assert repeated.shape == (1, 6)
    np.testing.assert_array_equal(repeated[0], single)


def test_same_plan_serializes_concurrent_grow_and_shrink(context):
    plan = wgpu_nufft.Type1Plan(context, (8,), eps=1e-4)
    large_points = np.linspace(-2.5, 2.5, 32, dtype=np.float32)
    large_strengths = (
        np.linspace(-0.7, 0.9, 32) + 1j * np.linspace(0.6, -0.4, 32)
    ).astype(np.complex64)
    small_points = np.array([-1.1, 0.2, 1.4], dtype=np.float32)
    small_strengths = np.array(
        [0.3 + 0.8j, -0.5 + 0.1j, 0.9 - 0.4j], dtype=np.complex64
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        large_future = pool.submit(plan.execute, large_points, large_strengths)
        small_future = pool.submit(plan.execute, small_points, small_strengths)
        large = large_future.result()
        small = small_future.result()

    _assert_gpu_close(
        large, direct_type1(large_points, large_strengths, (8,), 1, "centered")
    )
    _assert_gpu_close(
        small, direct_type1(small_points, small_strengths, (8,), 1, "centered")
    )
    np.testing.assert_array_equal(small, plan.execute(small_points, small_strengths))


def test_type3_batched_matches_direct(context):
    source = np.array([[-0.8, 0.3], [0.2, -0.5], [0.9, 0.7]], dtype=np.float32)
    target = np.array([[-1.2, 0.1], [0.4, -0.9], [1.1, 0.8], [1.7, -1.4]], dtype=np.float32)
    strengths = np.array(
        [[1.0 + 0.2j, -0.3 + 0.7j, 0.4 - 0.5j],
         [-0.2 + 0.1j, 0.8 - 0.4j, 0.6 + 0.3j]],
        dtype=np.complex64,
    )
    source_bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    target_bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)
    plan = wgpu_nufft.Type3Plan(
        context, source_bounds, target_bounds, eps=1e-4, isign=-1, batch=2
    )
    actual = plan.execute(source, strengths, target)
    assert actual.shape == (2, 4)
    for transform in range(2):
        _assert_gpu_close(
            actual[transform], direct_type3(source, strengths[transform], target, -1)
        )


def test_empty_transforms_have_mathematical_shapes(context):
    empty_points = np.empty(0, dtype=np.float32)
    empty_values = np.empty(0, dtype=np.complex64)

    type1 = wgpu_nufft.Type1Plan(context, (5,), eps=1e-4)
    np.testing.assert_array_equal(
        type1.execute(empty_points, empty_values), np.zeros(5, dtype=np.complex64)
    )

    type2 = wgpu_nufft.Type2Plan(context, (5,), eps=1e-4)
    assert type2.execute(empty_points, np.ones(5, dtype=np.complex64)).shape == (0,)

    bounds = np.array([[-1.0, 1.0]], dtype=np.float64)
    type3 = wgpu_nufft.Type3Plan(context, bounds, bounds, eps=1e-4)
    target = np.array([-0.5, 0.25], dtype=np.float32)
    np.testing.assert_array_equal(
        type3.execute(empty_points, empty_values, target),
        np.zeros(2, dtype=np.complex64),
    )


def test_validation_errors_are_clear(context):
    plan = wgpu_nufft.Type1Plan(context, (5,), eps=1e-4)
    values = np.ones(3, dtype=np.complex64)

    with pytest.raises(TypeError):
        plan.execute(np.zeros(3, dtype=np.float64), values)
    with pytest.raises(ValueError, match="C-contiguous"):
        plan.execute(np.arange(6, dtype=np.float32)[::2], values)
    with pytest.raises(ValueError, match="non-finite"):
        plan.execute(np.array([0.0, np.nan, 1.0], dtype=np.float32), values)
    with pytest.raises(ValueError, match=r"outside \[-3\*pi, 3\*pi\]"):
        plan.execute(np.array([0.0, 10.0, 1.0], dtype=np.float32), values)
    with pytest.raises(ValueError, match="plan capacity"):
        plan.execute(
            np.zeros(3, dtype=np.float32), np.ones((2, 3), dtype=np.complex64)
        )

    bounds = np.array([[-1.0, 1.0]], dtype=np.float64)
    type3 = wgpu_nufft.Type3Plan(context, bounds, bounds, eps=1e-4)
    with pytest.raises(ValueError, match="outside"):
        type3.execute(
            np.array([1.1], dtype=np.float32),
            np.ones(1, dtype=np.complex64),
            np.array([0.0], dtype=np.float32),
        )
