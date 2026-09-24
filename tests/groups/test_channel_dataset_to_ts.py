"""ChannelDataset.to_channel_ts and time_slice against the expressions they replaced."""

import shutil

import numpy as np
import pytest
from mt_metadata.common.mttime import MTime
from mt_timeseries import ChannelTS
from mt_timeseries.channel_ts import make_dt_coordinates

from mth5.mth5 import MTH5
from mth5.utils.helpers import close_open_files


@pytest.fixture(
    params=["global_test1_mth5", "global_test1_v2_mth5", "global_test3_mth5"]
)
def channels(request, tmp_path):
    source = request.getfixturevalue(request.param)
    target = tmp_path / "archive.h5"
    shutil.copy2(source, target)
    m = MTH5()
    m.open_mth5(target, "r")
    df = m.channel_summary.to_dataframe()
    yield [m.from_reference(ref) for ref in df.hdf5_reference]
    m.close_mth5()
    close_open_files()


def _old_to_channel_ts(ch):
    return ChannelTS(
        channel_type=ch.metadata.type,
        data=ch.hdf5_dataset[()],
        channel_metadata=ch.metadata.copy(),
        run_metadata=ch.run_metadata.copy(),
        station_metadata=ch.station_metadata.copy(),
        survey_metadata=ch.survey_metadata.copy(),
        channel_response=ch.channel_response,
    )


def _old_time_slice(ch, start, n_samples):
    start_index, end_index, npts = ch._get_slice_index_values(start, None, n_samples)
    dt_index = make_dt_coordinates(start, ch.sample_rate, npts)
    meta_dict = ch.metadata.to_dict()[ch.metadata._class_name]
    meta_dict["time_period.start"] = dt_index[0].isoformat()
    meta_dict["time_period.end"] = dt_index[-1].isoformat()
    return ChannelTS(
        ch.metadata.type,
        data=ch.hdf5_dataset[start_index:end_index],
        survey_metadata=ch.survey_metadata.copy(),
        station_metadata=ch.station_metadata.copy(),
        run_metadata=ch.run_metadata.copy(),
        channel_metadata={ch.metadata.type: meta_dict},
        channel_response=ch.channel_response,
    )


def _assert_same(new, old):
    assert new.data_array.identical(old.data_array)
    assert new.survey_metadata.to_dict() == old.survey_metadata.to_dict()
    assert new.station_metadata.to_dict() == old.station_metadata.to_dict()
    assert new.run_metadata.to_dict() == old.run_metadata.to_dict()
    assert new.channel_metadata.to_dict() == old.channel_metadata.to_dict()
    assert new.channel_response == old.channel_response


def test_to_channel_ts_unchanged(channels):
    for ch in channels:
        _assert_same(ch.to_channel_ts(), _old_to_channel_ts(ch))


def test_time_slice_unchanged(channels):
    for ch in channels:
        start = MTime(time_stamp=ch.metadata.time_period.start)
        for offset, n_samples in [(0.0, 1), (10.0, 2), (12.5, 100), (1.0, 4095)]:
            begin = start + offset
            _assert_same(
                ch.time_slice(begin, n_samples=n_samples),
                _old_time_slice(ch, begin, n_samples),
            )


@pytest.mark.parametrize(
    "sample_rate", [1000.0, 256.0, 24000.0, 10.00064, 8.0, 1.0, 0.1, 1 / 3]
)
@pytest.mark.parametrize("start", ["2020-01-01T00:00:00", "2009-06-16T02:01:04.123456"])
@pytest.mark.parametrize("npts", [1, 2, 3, 1000, 360_001])
def test_slice_time_bounds_match_full_index(sample_rate, start, npts):
    class Channel:
        pass

    ch = Channel()
    ch.sample_rate = sample_rate
    full = make_dt_coordinates(start, sample_rate, npts)
    from mth5.groups.channel_dataset import ChannelDataset

    first, last = ChannelDataset._slice_time_bounds(ch, start, npts)
    assert first == full[0]
    assert last == full[-1]
