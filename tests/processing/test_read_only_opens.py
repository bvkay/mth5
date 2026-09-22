"""RunSummary and KernelDataset only read the archives they summarize."""

import os
import shutil

import h5py
import pytest

from mth5.processing.kernel_dataset import KernelDataset
from mth5.processing.run_summary import RunSummary
from mth5.utils.helpers import close_open_files


@pytest.fixture
def archive(global_test12rr_mth5, tmp_path):
    target = tmp_path / "test12rr.h5"
    shutil.copy2(global_test12rr_mth5, target)
    yield target
    close_open_files()


def test_summary_and_kernel_dataset_open_read_only(archive):
    mtime = os.stat(archive).st_mtime_ns
    # a reader holding the file read-only blocks any read-write open
    with h5py.File(archive, "r"):
        run_summary = RunSummary()
        run_summary.from_mth5s([archive])
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test1", "test2")
    assert sorted(kernel_dataset.df.station.unique()) == ["test1", "test2"]
    assert os.stat(archive).st_mtime_ns == mtime
