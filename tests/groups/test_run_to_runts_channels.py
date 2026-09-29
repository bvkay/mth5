"""RunGroup.to_runts reads only the channels asked for."""

import h5py
import numpy as np
import pytest

from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error
from mth5.utils.helpers import close_open_files


@pytest.fixture
def run_group(global_test1_mth5):
    m = MTH5()
    m.open_mth5(global_test1_mth5, mode="r")
    yield m.get_run("test1", "001")
    m.close_mth5()
    close_open_files()


def read_runts(run_group, monkeypatch, **kwargs):
    """to_runts, with the names of the datasets it read."""
    read = set()
    getitem = h5py.Dataset.__getitem__

    def counted(self, args, *more, **more_kwargs):
        read.add(self.name.split("/")[-1])
        return getitem(self, args, *more, **more_kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counted)
    try:
        run_ts = run_group.to_runts(**kwargs)
    finally:
        monkeypatch.undo()
    return run_ts, read


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"start": "1980-01-01T00:00:00", "n_samples": 4096}],
    ids=["whole_run", "time_slice"],
)
def test_to_runts_channels(run_group, monkeypatch, kwargs):
    """to_runts reads only the channels asked for, and they equal the full read."""
    full, read_full = read_runts(run_group, monkeypatch, **kwargs)
    assert {"ex", "ey", "hx", "hy", "hz"} <= read_full

    part, read_part = read_runts(
        run_group, monkeypatch, channels=["hy", "HX"], **kwargs
    )
    assert part.channels == ["hx", "hy"]
    assert not {"ex", "ey", "hz"} & read_part
    np.testing.assert_array_equal(part.dataset.time.data, full.dataset.time.data)
    for channel in ["hx", "hy"]:
        np.testing.assert_array_equal(
            part.dataset[channel].data, full.dataset[channel].data
        )
        assert part.dataset[channel].attrs == full.dataset[channel].attrs
    assert part.run_metadata.to_dict(single=True) == full.run_metadata.to_dict(
        single=True
    )


def test_to_runts_unknown_channel(run_group):
    """A channel the run does not have raises MTH5Error."""
    with pytest.raises(MTH5Error, match="bx"):
        run_group.to_runts(channels=["hx", "bx"])
