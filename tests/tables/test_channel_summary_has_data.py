"""The channel summary does not read channels that store no data."""

import h5py
import numpy as np
import pytest

from mth5.mth5 import MTH5


@pytest.fixture
def archive(tmp_path):
    """A run with an unwritten, a zero, a non-zero and a fill-valued channel."""
    n_samples = 100_000
    path = tmp_path / "has_data.h5"
    m = MTH5(file_version="0.2.0")
    m.open_mth5(path, mode="w")
    m.add_survey("test")
    station = m.add_station("mt01", survey="test")
    run = station.add_run("a")
    run.add_channel("ex", "electric", None, shape=(n_samples,), channel_dtype="float64")
    run.add_channel("ey", "electric", np.zeros(n_samples))
    run.add_channel("hx", "magnetic", np.arange(n_samples, dtype=float))
    m.close_mth5()

    # never written, but every sample reads back as 1
    with h5py.File(path, "r+") as h5:
        run_group = h5["Experiment/Surveys/test/Stations/mt01/a"]
        hy = run_group.create_dataset(
            "hy", shape=(n_samples,), dtype="float64", chunks=True, fillvalue=1.0
        )
        for key, value in run_group["hx"].attrs.items():
            hy.attrs[key] = value
        hy.attrs["component"] = "hy"
    yield path


def test_summary_skips_reading_unwritten_channels(archive, monkeypatch):
    """summarize gives the has_data column without reading the unwritten channel."""
    with h5py.File(archive, "r") as h5:
        storage = {
            name: ds.id.get_storage_size()
            for name, ds in h5["Experiment/Surveys/test/Stations/mt01/a"].items()
        }
    assert storage["ex"] == 0 and storage["hy"] == 0
    assert storage["ey"] > 0 and storage["hx"] > 0

    read = []
    getitem = h5py.Dataset.__getitem__

    def counted(self, args, *more, **kwargs):
        read.append(self.name.split("/")[-1])
        return getitem(self, args, *more, **kwargs)

    m = MTH5()
    m.open_mth5(archive, mode="a")
    try:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", counted)
        m.channel_summary.summarize()
        monkeypatch.undo()
        df = m.channel_summary.to_dataframe()
    finally:
        monkeypatch.undo()
        m.close_mth5()

    has_data = dict(zip(df.component, df.has_data))
    assert has_data == {"ex": False, "ey": False, "hx": True, "hy": True}
    assert "ex" not in read
    assert {"ey", "hx", "hy"} <= set(read)
