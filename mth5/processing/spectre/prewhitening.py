"""
    This module has methods for pre-whitening time series to reduce spectral leakage before FFT.
"""
from typing import Literal

import numpy as np
import xarray as xr
from loguru import logger


def apply_prewhitening(
    prewhitening_type: Literal["first difference", ""],
    run_xrds_input: xr.Dataset,
) -> xr.Dataset:
    """
    Applies pre-whitening to time series to avoid spectral leakage when FFT is applied.

    Parameters
    ----------
    prewhitening_type: Literal["first difference", ]
        Placeholder to allow for multiple methods of pre-whitening.  Currently only
        "first difference" is supported.
    run_xrds_input : xr.Dataset
        Time series to be pre-whitened (can be multivariate).

    Returns
    -------
    run_xrds : xr..Dataset
        pre-whitened time series

    """
    if not prewhitening_type:
        msg = "No prewhitening specified - skipping this step"
        logger.info(msg)
        return run_xrds_input

    if prewhitening_type == "first difference":
        run_xrds = _differentiate_time(run_xrds_input)
    else:
        msg = f"{prewhitening_type} pre-whitening not implemented"
        logger.exception(msg)
        raise NotImplementedError(msg)
    return run_xrds


def _differentiate_time(run_xrds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    """
    Differentiate each variable along time, as ``run_xrds.differentiate(dim)`` does.

    ``differentiate`` turns a datetime coordinate into numbers in its own unit
    (counted from the earliest time) and passes that array to ``np.gradient``
    for every variable, which checks the spacing each time and, when it is
    even, uses the scalar step.  Here the spacing is checked once on the
    integer times and ``np.gradient`` gets the scalar step directly, which
    gives the same numbers.  Uneven or missing times, a span that float64
    does not hold exactly, and data that is not an in-memory numeric array
    go through ``differentiate``.

    Parameters
    ----------
    run_xrds : xr.Dataset
        Time series to differentiate (can be multivariate).
    dim : str, optional
        Name of the time coordinate, by default "time".

    Returns
    -------
    xr.Dataset
        Derivative of each variable with respect to time, per unit of the
        time coordinate.
    """
    names = [name for name, da in run_xrds.data_vars.items() if dim in da.dims]
    has_time = dim in run_xrds.coords and run_xrds[dim].dims == (dim,)
    time = run_xrds[dim].values if has_time else None
    if (
        time is None
        or time.dtype.kind != "M"
        or time.size < 2
        or np.isnat(time[[0, -1]]).any()
        or any(
            not isinstance(run_xrds[name].data, np.ndarray)
            or run_xrds[name].dtype.kind not in "iufc"
            for name in names
        )
    ):
        return run_xrds.differentiate(dim)

    ticks = time.view("int64")
    steps = np.diff(ticks)
    if (
        steps[0] == 0
        or not (steps == steps[0]).all()
        or abs(int(ticks[-1]) - int(ticks[0])) > 2**53
    ):
        return run_xrds.differentiate(dim)
    # a float64 step, as differentiate's is: with a Python float NumPy 2
    # would divide single-precision data in single precision
    step = np.float64(steps[0])

    data_vars = {}
    for name, da in run_xrds.data_vars.items():
        if name in names:
            data_vars[name] = (
                da.dims,
                np.gradient(da.data, step, axis=da.get_axis_num(dim)),
            )
        else:
            data_vars[name] = da.variable
    differentiated = xr.Dataset(data_vars, coords=run_xrds.coords, attrs=run_xrds.attrs)
    differentiated.encoding = dict(run_xrds.encoding)
    return differentiated


def apply_recoloring(
    prewhitening_type: Literal["first difference",],
    stft_obj: xr.Dataset,
) -> xr.Dataset:
    """
    Inverts the pre-whitening operation in frequency domain.  Modifies the input xarray in-place.

    Parameters
    ----------
    prewhitening_type: Literal["first difference", ]
        Placeholder to allow for multiple methods of pre-whitening.  Currently only
        "first difference" is supported.
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be recoloured


    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Recolored time series of Fourier coefficients.
    """

    # No recoloring needed if prewhitening not appiled, or recoloring set to False
    if not prewhitening_type:
        msg = "No prewhitening was defined -- skipping recoloring"
        logger.debug(msg)
        return stft_obj

    if prewhitening_type == "first difference":
        freqs = stft_obj.frequency.data
        jw = 1.0j * 2 * np.pi * freqs
        stft_obj /= jw

        # suppress nan and inf to mute later warnings
        if jw[0] == 0.0:
            cond = stft_obj.frequency != 0.0
            stft_obj = stft_obj.where(cond, complex(0.0))
    # elif prewhitening_type == "ARMA":
    #     from statsmodels.tsa.arima.model import ARIMA
    #     AR = 3 # TODO: add this to processing config
    #     MA = 4 # TODO: add this to processing config

    else:
        msg = f"{prewhitening_type} recoloring not yet implemented"
        logger.error(msg)
        raise NotImplementedError(msg)

    return stft_obj
