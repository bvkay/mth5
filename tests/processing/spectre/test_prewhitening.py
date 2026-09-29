"""apply_prewhitening gives the Dataset that xarray's differentiate gives."""

import numpy as np
import pytest
import xarray as xr

from mth5.processing.spectre.prewhitening import apply_prewhitening


def random_run(n=5000, step_ns=1_000_000, unit="ns", seed=0):
    """A multivariate run with the variable kinds a run Dataset can hold."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2023-09-23T00:00:00", unit)
    time = start + (np.arange(n) * np.timedelta64(step_ns, "ns")).astype(
        f"timedelta64[{unit}]"
    )
    return xr.Dataset(
        {
            "ex": ("time", rng.standard_normal(n), {"units": "millivolts"}),
            "ey": ("time", rng.standard_normal(n).astype(np.float32)),
            "hx": ("time", rng.integers(-(2**20), 2**20, n).astype(np.int32)),
            "hy": ("time", rng.standard_normal(n) + 1j * rng.standard_normal(n)),
            "stack": (("channel", "time"), rng.standard_normal((3, n))),
            "gain": ("channel", np.array([1.0, 2.0, 3.0])),
        },
        coords={"time": time, "channel": ["a", "b", "c"], "station": "mt01"},
        attrs={"sample_rate": 1e9 / step_ns},
    )


def check_first_difference(run_xrds, monkeypatch, through_differentiate):
    """Compare with differentiate and count the calls apply_prewhitening makes."""
    expected = run_xrds.differentiate("time")
    calls = []
    differentiate = xr.Dataset.differentiate

    def counted(self, *args, **kwargs):
        calls.append(args)
        return differentiate(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "differentiate", counted)
    result = apply_prewhitening("first difference", run_xrds)
    monkeypatch.undo()

    xr.testing.assert_identical(result, expected)
    assert list(result.data_vars) == list(expected.data_vars)
    assert list(result.coords) == list(expected.coords)
    for name in expected.data_vars:
        assert result[name].dtype == expected[name].dtype
        assert np.array_equal(result[name].data, expected[name].data, equal_nan=True)
    assert len(calls) == int(through_differentiate)


@pytest.mark.parametrize(
    "step_ns, unit",
    [
        (1_000_000, "ns"),  # 1000 Hz
        (244_141, "ns"),  # 4096 Hz on a whole-ns index
        (1_000_000, "us"),
        (16_000_000_000, "ns"),  # 1/16 Hz
    ],
)
def test_first_difference_evenly_sampled(step_ns, unit, monkeypatch):
    run_xrds = random_run(step_ns=step_ns, unit=unit)
    check_first_difference(run_xrds, monkeypatch, through_differentiate=False)


@pytest.mark.parametrize(
    "dtype, step_ns",
    [
        ("complex64", 1_000_000),  # 1000 Hz
        ("float32", 33_333_331),  # twice the step is not exact in float32
    ],
)
def test_first_difference_single_precision(dtype, step_ns, monkeypatch):
    # differentiate divides by a float64 step; a single-precision step differs
    rng = np.random.default_rng(1)
    run_xrds = random_run(step_ns=step_ns)
    data = rng.standard_normal(run_xrds.time.size)
    if np.dtype(dtype).kind == "c":
        data = data + 1j * rng.standard_normal(run_xrds.time.size)
    run_xrds = run_xrds.assign(hx=("time", data.astype(dtype)))
    check_first_difference(run_xrds, monkeypatch, through_differentiate=False)


def test_first_difference_uneven_and_missing_times(monkeypatch):
    run_xrds = random_run()
    time = run_xrds.time.values.copy()
    time[2500:] += np.timedelta64(500, "us")
    uneven = run_xrds.assign_coords(time=time)
    check_first_difference(uneven, monkeypatch, through_differentiate=True)

    time = run_xrds.time.values.copy()
    time[0] = np.datetime64("NaT")
    missing = run_xrds.assign_coords(time=time)
    with np.errstate(invalid="ignore"):
        check_first_difference(missing, monkeypatch, through_differentiate=True)


def test_first_difference_numeric_time_coordinate(monkeypatch):
    run_xrds = random_run()
    seconds = run_xrds.assign_coords(time=np.arange(run_xrds.time.size) / 1000.0)
    check_first_difference(seconds, monkeypatch, through_differentiate=True)


def test_first_difference_long_span(monkeypatch):
    # 2**53 ns is about 104 days: past it the float64 times are not exact
    run_xrds = random_run(n=40, step_ns=3 * 86400 * 10**9)
    check_first_difference(run_xrds, monkeypatch, through_differentiate=True)
