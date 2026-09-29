# -*- coding: utf-8 -*-
from __future__ import annotations


"""Channel summary utilities for MTH5 tables."""

from typing import Any, Iterable

import h5py
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from mt_metadata.transfer_functions import (
    ALLOWED_INPUT_CHANNELS,
    ALLOWED_OUTPUT_CHANNELS,
)

from mth5 import CHANNEL_DTYPE, RUN_SUMMARY_COLUMNS
from mth5.tables import MTH5Table


# =============================================================================


def _channel_summary_to_dataframe(array: np.ndarray) -> pd.DataFrame:
    """Decode a channel summary array into a pandas DataFrame.

    Parameters
    ----------
    array : numpy.ndarray
        Structured array with ``CHANNEL_DTYPE`` fields, read from the table
        or built in memory.

    Returns
    -------
    pandas.DataFrame
        Channel summary with decoded string columns and parsed datetimes.
    """

    df = pd.DataFrame(array)
    for key in [
        "survey",
        "station",
        "run",
        "component",
        "measurement_type",
        "units",
    ]:
        setattr(df, key, getattr(df, key).str.decode("utf-8"))
    try:
        df.start = pd.to_datetime(df.start.str.decode("utf-8"), format="mixed")
        df.end = pd.to_datetime(df.end.str.decode("utf-8"), format="mixed")
    except ValueError:
        df.start = pd.to_datetime(df.start.str.decode("utf-8"))
        df.end = pd.to_datetime(df.end.str.decode("utf-8"))

    return df


# =============================================================================


class ChannelSummaryTable(MTH5Table):
    """Convenience wrapper around the channel summary dataset.

    Provides helpers to summarize channels, convert to pandas, and derive
    run-level summaries.

    Examples
    --------
    >>> ch_table = ChannelSummaryTable(hdf5_dataset)
    >>> df = ch_table.to_dataframe()  # doctest: +SKIP
    >>> run_df = ch_table.to_run_summary()  # doctest: +SKIP
    """

    def __init__(self, hdf5_dataset: h5py.Dataset) -> None:
        super().__init__(hdf5_dataset, CHANNEL_DTYPE)

    def _has_entries(self) -> bool:
        """Return ``True`` if the summary table contains data."""

        if len(self.array) == 1:
            if self.array[0][0] == b"" and self.array[0][1] == b"":
                return False
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the channel summary to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Channel summary with decoded string columns and parsed datetimes.

        Examples
        --------
        >>> df = ch_table.to_dataframe()  # doctest: +SKIP
        >>> df.head()  # doctest: +SKIP
        """

        return _channel_summary_to_dataframe(self.array[()])

    def _can_write(self) -> bool:
        """Return ``True`` if the file holding the table is open for writing."""
        file_mode = getattr(getattr(self.array, "file", None), "mode", "")
        return "w" in file_mode or "+" in file_mode

    def summarize(self) -> None:
        """Populate the summary table from channel datasets in the file."""

        self.clear_table()
        for ch_entry in self._channel_entries():
            try:
                self.add_row(ch_entry)
            except ValueError as error:
                msg = (
                    f"{error}. "
                    "it is possible that the OS that made the table is not the OS operating on it."
                )
                self.logger.warning(msg)

    def _channel_entries(self) -> list[np.ndarray]:
        """Build one summary entry for each channel dataset in the file.

        It only reads the file, so it also works on a file open read-only.

        Returns
        -------
        list of numpy.ndarray
            One single-row ``CHANNEL_DTYPE`` array per electric, magnetic or
            auxiliary channel dataset, in the order the file is traversed.
        """

        entries = []

        def has_data(h5_dataset: h5py.Dataset) -> bool:
            """Return True when the dataset has any non-zero data."""
            # a dataset allocated but never written stores 0 bytes and reads
            # back as its fill value; skip the scan when that value is zero
            if (
                h5_dataset.id.get_storage_size() == 0
                and not h5_dataset.is_virtual
                and h5_dataset.external is None
                and not np.any(h5_dataset.fillvalue)
            ):
                return False
            # scan in blocks and stop at the first non-zero sample rather
            # than reading the whole channel back
            step = 2**22
            for start in range(0, len(h5_dataset), step):
                if np.any(h5_dataset[start : start + step]):
                    return True
            return False

        def get_channel_entry(
            group: h5py.Dataset, dtype: Any = CHANNEL_DTYPE
        ) -> np.ndarray:
            ch_entry = np.array(
                [
                    (
                        group.parent.parent.parent.parent.attrs["id"].encode("utf-8"),
                        group.parent.parent.attrs["id"].encode("utf-8"),
                        group.parent.attrs["id"].encode("utf-8"),
                        group.parent.parent.attrs["location.latitude"],
                        group.parent.parent.attrs["location.longitude"],
                        group.parent.parent.attrs["location.elevation"],
                        group.attrs["component"],
                        group.attrs["time_period.start"],
                        group.attrs["time_period.end"],
                        group.size,
                        group.attrs["sample_rate"],
                        group.attrs["type"],
                        group.attrs["measurement_azimuth"],
                        group.attrs["measurement_tilt"],
                        group.attrs["units"],
                        has_data(group),
                        group.ref,
                        group.parent.ref,
                        group.parent.parent.ref,
                    )
                ],
                dtype=dtype,
            )
            return ch_entry

        def recursive_get_channel_entry(group: h5py.Group | h5py.File) -> None:
            """Traverse HDF5 tree and collect channel entries."""
            if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
                for key, node in group.items():
                    recursive_get_channel_entry(node)
            elif isinstance(group, h5py._hl.dataset.Dataset):
                try:
                    ch_type = group.attrs["type"]
                    if ch_type in ["electric", "magnetic", "auxiliary"]:
                        entries.append(get_channel_entry(group))

                except KeyError:
                    pass

        recursive_get_channel_entry(self.array.parent)
        return entries

    def to_run_summary(
        self,
        allowed_input_channels: Iterable[str] = ALLOWED_INPUT_CHANNELS,
        allowed_output_channels: Iterable[str] = ALLOWED_OUTPUT_CHANNELS,
        sortby: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compress channel summary into a run-level summary (one row per run).

        Parameters
        ----------
        allowed_input_channels : Iterable[str], optional
            Allowed input channel names, by default ``ALLOWED_INPUT_CHANNELS``.
        allowed_output_channels : Iterable[str], optional
            Allowed output channel names, by default ``ALLOWED_OUTPUT_CHANNELS``.
        sortby : list of str or None, optional
            Columns to sort by; defaults to ``["station", "start"]`` when ``None``.

        Returns
        -------
        pandas.DataFrame
            Run-level summary including channels, durations, and references.

        Examples
        --------
        >>> run_df = ch_table.to_run_summary()  # doctest: +SKIP
        >>> run_df.columns[:4].tolist()  # doctest: +SKIP
        ['survey', 'station', 'run', 'start']
        """

        if self._has_entries():
            ch_summary_df = self.to_dataframe()
        elif self._can_write():
            self.summarize()
            ch_summary_df = self.to_dataframe()
        else:
            # the summary was never written (an interrupted writer) and the
            # file is open read-only: summarize in memory only
            self.logger.debug(
                f"Channel summary of {self.array.file.filename} has no entries "
                "and the file is read-only; summarizing in memory."
            )
            entries = self._channel_entries()
            if entries:
                array = np.concatenate(entries)
            else:
                # what summarize leaves in a file without channels
                array = np.zeros(1, dtype=self.dtype)
            ch_summary_df = _channel_summary_to_dataframe(array)

        group_by_columns = ["survey", "station", "run"]
        grouper = ch_summary_df.groupby(group_by_columns)
        row_list = []
        for group_values, group in grouper:
            # for entry in group.itertuples():
            row = dict([(key, None) for key in RUN_SUMMARY_COLUMNS])
            row["survey"] = group.survey.iloc[0]
            row["station"] = group.station.iloc[0]
            row["run"] = group.run.iloc[0]
            row["start"] = group.start.iloc[0]
            row["end"] = group.end.iloc[0]
            row["sample_rate"] = group.sample_rate.iloc[0]
            # max
            row["n_samples"] = group.n_samples.max()
            channels_list = group.component.to_list()
            num_channels = len(channels_list)
            row["input_channels"] = [
                x for x in channels_list if x in allowed_input_channels
            ]
            row["output_channels"] = [
                x for x in channels_list if x in allowed_output_channels
            ]
            row["channel_scale_factors"] = dict(
                zip(channels_list, num_channels * [1.0])
            )
            row["has_data"] = True
            if False in group.has_data.values:
                row["has_data"] = False

            row["run_hdf5_reference"] = group.run_hdf5_reference.iloc[0]
            row["station_hdf5_reference"] = group.station_hdf5_reference.iloc[0]

            row_list.append(row)

        run_summary_df = pd.DataFrame(data=row_list)
        if sortby is None:
            sortby = ["station", "start"]
        if sortby:
            run_summary_df.sort_values(by=sortby, inplace=True)

        # add durations
        timedeltas = run_summary_df.end - run_summary_df.start
        durations = [x.total_seconds() for x in timedeltas]
        run_summary_df["duration"] = durations

        return run_summary_df
