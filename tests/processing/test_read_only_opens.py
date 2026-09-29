"""RunSummary and KernelDataset only read the archives they summarize."""

import os
import shutil

import h5py
import pandas as pd
import pytest

from mth5.mth5 import MTH5
from mth5.processing.kernel_dataset import KernelDataset
from mth5.processing.run_summary import RunSummary
from mth5.tables import ChannelSummaryTable
from mth5.utils.helpers import close_open_files


@pytest.fixture
def archive(global_test12rr_mth5, tmp_path):
    target = tmp_path / "test12rr.h5"
    shutil.copy2(global_test12rr_mth5, target)
    yield target
    close_open_files()


def test_summary_and_kernel_dataset_open_read_only(archive):
    """Run summaries and kernel datasets read a held archive and leave its mtime."""
    mtime = os.stat(archive).st_mtime_ns
    # a reader holding the file read-only blocks any read-write open
    with h5py.File(archive, "r"):
        run_summary = RunSummary()
        run_summary.from_mth5s([archive])
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test1", "test2")
    assert sorted(kernel_dataset.df.station.unique()) == ["test1", "test2"]
    assert os.stat(archive).st_mtime_ns == mtime


def empty_channel_summary(path):
    """Reset the channel summary to the empty row of a new file.

    This is what a writer leaves behind when it stops before close_mth5.
    """
    with h5py.File(path, "r+") as h5:
        root = "Experiment" if "Experiment" in h5 else "Survey"
        ChannelSummaryTable(h5[f"{root}/channel_summary"]).clear_table()


@pytest.fixture
def unsummarized_archives(global_test12rr_mth5, tmp_path):
    paths = []
    for name in ["read_write.h5", "read_only.h5"]:
        target = tmp_path / name
        shutil.copy2(global_test12rr_mth5, target)
        empty_channel_summary(target)
        paths.append(target)
    yield paths
    close_open_files()


def run_summary_of(path, mode):
    """Run summary of an archive, with each run reference given by its name."""
    m = MTH5()
    m.open_mth5(path, mode=mode)
    try:
        df = m.run_summary
        df["run_name"] = [
            m.from_reference(ref).hdf5_group.name if ref else None
            for ref in df.run_hdf5_reference
        ]
    finally:
        m.close_mth5()
    return df.drop(
        columns=["mth5_path", "run_hdf5_reference", "station_hdf5_reference"]
    )


def test_run_summary_of_unsummarized_archive_read_only(unsummarized_archives):
    """An unsummarized archive gives the same run summary read-only."""
    read_write, read_only = unsummarized_archives
    # a read-write open writes the summary to the file, then reads it
    expected = run_summary_of(read_write, "a")
    assert len(expected) == 2

    mtime = os.stat(read_only).st_mtime_ns
    pd.testing.assert_frame_equal(run_summary_of(read_only, "r"), expected)

    run_summary = RunSummary()
    run_summary.from_mth5s([read_only])
    assert sorted(run_summary.df.station) == ["test1", "test2"]
    assert os.stat(read_only).st_mtime_ns == mtime


def test_run_summary_of_empty_archive_read_only(tmp_path):
    """An archive without channels gives the same run summary read-only."""
    read_write = tmp_path / "read_write.h5"
    read_only = tmp_path / "read_only.h5"
    m = MTH5(file_version="0.2.0")
    m.open_mth5(read_write, mode="w")
    m.close_mth5()
    shutil.copy2(read_write, read_only)

    expected = run_summary_of(read_write, "a")
    assert len(expected) == 1  # the empty row, which RunSummary drops
    mtime = os.stat(read_only).st_mtime_ns
    pd.testing.assert_frame_equal(run_summary_of(read_only, "r"), expected)
    assert os.stat(read_only).st_mtime_ns == mtime
