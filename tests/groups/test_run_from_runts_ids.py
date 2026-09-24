"""Run ids through RunGroup.from_runts."""

import numpy as np
import pytest
from loguru import logger
from mt_metadata.timeseries import Electric, Magnetic, Run, Station
from mt_timeseries import ChannelTS, RunTS

from mth5.mth5 import MTH5


def _run_ts(run_id):
    channels = []
    for comp, cls, ch_type in [
        ("ex", Electric, "electric"),
        ("hx", Magnetic, "magnetic"),
    ]:
        meta = cls(component=comp, sample_rate=8.0)
        meta.time_period.start = "2020-01-01T00:00:00+00:00"
        channels.append(
            ChannelTS(
                ch_type,
                data=np.arange(64, dtype=float),
                channel_metadata=meta,
                run_metadata=Run(id=run_id),
                station_metadata=Station(id="mt01"),
            )
        )
    return RunTS(
        array_list=channels,
        run_metadata=Run(id=run_id),
        station_metadata=Station(id="mt01"),
    )


@pytest.fixture
def run_warnings():
    records = []
    sink = logger.add(
        lambda message: records.append(message.record["message"]),
        level="WARNING",
        filter="mth5.groups.run",
    )
    yield records
    logger.remove(sink)


def _write(tmp_path, run_ts, group_id):
    m = MTH5(file_version="0.2.0")
    m.open_mth5(tmp_path / "ids.h5", "w")
    try:
        m.add_survey("test")
        station = m.add_station("mt01", survey="test")
        run_group = station.add_run(group_id)
        run_group.from_runts(run_ts)
        return run_group.metadata.id, [
            run_group.get_channel(comp).run_metadata.id for comp in ("ex", "hx")
        ]
    finally:
        m.close_mth5()


def test_renamed_run_does_not_warn(tmp_path, run_warnings):
    run_ts = _run_ts("a")
    run_ts.run_metadata.id = "sr8_0001"
    group_id, channel_ids = _write(tmp_path, run_ts, "sr8_0001")
    assert run_warnings == []
    assert group_id == "sr8_0001"
    assert channel_ids == ["sr8_0001", "sr8_0001"]


def test_run_id_mismatch_warns_once(tmp_path, run_warnings):
    group_id, channel_ids = _write(tmp_path, _run_ts("a"), "sr8_0001")
    assert len(run_warnings) == 1
    assert "RunTS run.id a != group run.id sr8_0001" in run_warnings[0]
    assert group_id == "a"
    assert channel_ids == ["a", "a"]
