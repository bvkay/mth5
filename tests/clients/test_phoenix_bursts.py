"""PhoenixClient writes every burst of a segmented Phoenix file as its run.

The station is written here in the MTU-5C layout: a 128 byte header, then a
td_150 file of float32 samples, or a td_24k file of bursts, each a 32 byte
sub-header (GPS stamp, sample count, counts, min, max, mean) and its float32
samples. With an mt-io whose read_file gives a td_24k file as one ChannelTS
the collection lists one burst per file, and the tests follow its run table.
"""

import json
import struct
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mt_io.phoenix import PhoenixCollection
from mt_metadata.common.mttime import MTime

from mth5 import read_file
from mth5.clients.phoenix import PhoenixClient
from mth5.mth5 import MTH5
from mth5.utils.helpers import close_open_files


def header(sample_rate):
    """A 128 byte header of channel 0 (E1), file sequence 1."""
    head = bytearray(128)
    struct.pack_into("<BBH", head, 0, 2, 2, 128)
    head[4:12] = b"MTU-5C\x00\x00"
    head[12:20] = b"10128\x00\x00\x00"
    struct.pack_into("<IBIH", head, 20, RECORDING_ID, 0, 1, 360)
    head[52] = 0x08  # electric channel
    struct.pack_into("<HbBIH", head, 59, sample_rate, 0, 4, 64, 0)
    struct.pack_into("<fff", head, 71, 139.0, -29.7, 120.0)
    return bytes(head)


RECORDING_ID = 1619493876
FIRST_BURST = 1619493900
N_BURST = 480
N_CONTINUOUS = 600


def burst_values(k):
    """The samples written for burst `k`, different in every burst."""
    return np.arange(N_BURST, dtype=np.float32) + 1000.0 * (k + 1)


def continuous_values():
    """The samples written to the td_150 file."""
    return np.arange(N_CONTINUOUS, dtype=np.float32) - 5000.0


def write_station(folder, stamps):
    """Write a station with a td_150 file and a td_24k file of `stamps`."""
    channel = {
        "ty": "E",
        "ga": 1,
        "lp": 10000,
        "length1": 50,
        "length2": 50,
        "pot_p": 1,
        "pot_n": 1,
        "tag": "E1",
    }
    recmeta = {
        "instid": "10128",
        "start": "2021-04-27T03:24:36+00:00",
        "stop": "2021-04-27T04:24:36+00:00",
        "receiver_commercial_name": "MTU-5C",
        "receiver_model": "MTU-5C",
        "motherboard": {"mb_fw_ver": "1"},
        "layout": {
            "Station_Name": "MT001",
            "Survey_Name": "TEST",
            "Company_Name": "",
            "Notes": "",
            "Operator": "",
        },
        "channel_map": {"mapping": [{"idx": 0, "tag": "E1"}]},
        "chconfig": {"chans": [channel]},
        "timing": {
            "gps_lat": -29.7,
            "gps_lon": 139.0,
            "gps_alt": 120.0,
            "tm_drift": 0.0,
        },
    }
    (folder / "0").mkdir(parents=True)
    (folder / "recmeta.json").write_text(json.dumps(recmeta))
    stem = folder / "0" / f"10128_{RECORDING_ID:08X}_0_00000001"

    body = b""
    for k, stamp in enumerate(stamps):
        values = burst_values(k)
        sub = struct.pack(
            "<IIHHfff", stamp, N_BURST, 0, 0, values.min(), values.max(), values.mean()
        )
        body += sub.ljust(32, b"\x00") + values.tobytes()
    stem.with_suffix(".td_24k").write_bytes(header(24000) + body)
    stem.with_suffix(".td_150").write_bytes(header(150) + continuous_values().tobytes())
    return folder


@pytest.fixture
def station(tmp_path):
    """A station with a td_150 file and a td_24k file of two bursts."""
    return write_station(
        tmp_path / "data" / "10128_2021-04-27-032436",
        [FIRST_BURST, FIRST_BURST + 30],
    )


def make_archive(station, tmp_path):
    """Run the client on the station; its run table and a count of reads."""
    calls = Counter()

    def counted_read_file(fn, **kwargs):
        calls[fn.name] += 1
        return read_file(fn, **kwargs)

    client = PhoenixClient(
        station.parent,
        sample_rates=[150, 24000],
        save_path=tmp_path / "phoenix.h5",
    )
    with patch("mth5.clients.phoenix.read_file", side_effect=counted_read_file):
        h5_path = client.make_mth5_from_phoenix()
    runs = PhoenixCollection(station.parent).get_runs([150, 24000])["MT001"]
    return h5_path, runs, calls


class TestPhoenixClientBursts:
    def test_every_burst_is_its_run(self, station, tmp_path):
        """Each run of the collection holds its own burst or file"""
        h5_path, runs, _ = make_archive(station, tmp_path)
        try:
            with MTH5() as m:
                m.open_mth5(h5_path, "r")
                summary = m.channel_summary.to_dataframe()
                assert sorted(summary.run) == sorted(runs)
                n_bursts = 0
                for run_id, run_df in runs.items():
                    row = next(run_df.itertuples())
                    ch = m.get_channel(
                        "MT001", run_id, "e1", survey="TEST"
                    ).to_channel_ts()
                    assert ch.start == MTime(time_stamp=row.start)
                    if row.sample_rate < 1000:
                        np.testing.assert_array_equal(ch.ts, continuous_values())
                        continue
                    n_bursts += 1
                    segment = getattr(row, "segment", 0)
                    np.testing.assert_array_equal(ch.ts, burst_values(int(segment)))
                    stamp = FIRST_BURST + 30 * int(segment)
                    assert ch.start == MTime(time_stamp=stamp, gps_time=True)
                assert n_bursts == len(runs) - 1
        finally:
            close_open_files()

    def test_each_file_read_once(self, station, tmp_path):
        """The td_24k file is read once for the runs of all its bursts"""
        h5_path, _, calls = make_archive(station, tmp_path)
        close_open_files()
        assert dict(calls) == {
            f"10128_{RECORDING_ID:08X}_0_00000001.td_150": 1,
            f"10128_{RECORDING_ID:08X}_0_00000001.td_24k": 1,
        }


@patch("mth5.clients.phoenix.read_file")
@patch("mth5.clients.phoenix.MTH5")
def test_burst_picked_by_segment(mock_mth5, mock_read_file, tmp_path):
    """A row takes the burst its segment names, a float or an integer"""
    run_group = MagicMock()
    m = mock_mth5.return_value.__enter__.return_value
    station_group = m.add_survey.return_value.stations_group.add_station.return_value
    station_group.add_run.return_value = run_group

    bursts = [MagicMock(component="e1"), MagicMock(component="e1")]
    mock_read_file.return_value = bursts

    def run_df(segment):
        row = MagicMock(fn=tmp_path / "a.td_24k", segment=segment)
        return MagicMock(
            sample_rate=MagicMock(unique=MagicMock(return_value=[24000])),
            itertuples=MagicMock(return_value=[row]),
        )

    client = PhoenixClient(tmp_path)
    client.collection = MagicMock(
        metadata_dict={"MT001": MagicMock(instrument_id="10128")}
    )
    client.collection.get_runs.return_value = {
        "MT001": {"sr24k_0001": run_df(1.0), "sr24k_0002": run_df(0)}
    }
    client.make_mth5_from_phoenix()

    written = [c.args[0] for c in run_group.from_channel_ts.call_args_list]
    assert written == [bursts[1], bursts[0]]
    assert mock_read_file.call_count == 1
