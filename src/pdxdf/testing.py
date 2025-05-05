import numpy as np
import pandas as pd

from pdxdf.resampling import nominal_sample_index


def lslfmt2np(channel_format):
    lslfmt_map = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float32": np.float32,
        "double64": np.float64,
        "string": str,
    }
    return lslfmt_map[channel_format]


def sine(
    freq=1,
    amp=1,
    phase=0,
    start=0,
    stop=1,
    fs=100,
    endpoint=True,
    dtype=None,
    segment=None,
    time_stamp_index=False,
):
    """
    freq: Fundamental frequency (Hz)
    amp: Amplitude
    phase: Phase in radians (range: 0-2pi)
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    dtype: Series data type
    segment: Segment index value
    time_stamps: Include time_stamps in index
    """
    period = 2 * np.pi

    t = nominal_sample_index(start, stop, fs, endpoint=endpoint, segment=segment)
    sig = pd.Series(
        np.sin(phase + (period * freq * t)) * amp, name=f"sine {freq}Hz", dtype=dtype
    )
    if time_stamp_index:
        sig = pd.concat([sig, t], axis=1).set_index("time_stamp", append=True)
        return sig.iloc[:, 0]
    else:
        return sig, t


def counter(
    start=0,
    stop=1,
    fs=100,
    endpoint=True,
    dtype=None,
    segment=None,
    time_stamp_index=False,
):
    """
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    dtype: Series data type
    segment: Segment index value
    time_stamps: Include time_stamps in index
    """

    t = nominal_sample_index(start, stop, fs=fs, endpoint=endpoint, segment=segment)
    sig = pd.Series(
        np.arange(0, t.shape[0], 1),
        name=f"counter {fs}",   # FIXME: Should be Hz.
        dtype=dtype,
        index=t.index,
    )
    if time_stamp_index:
        sig = pd.concat([sig, t], axis=1).set_index("time_stamp", append=True)
        return sig.iloc[:, 0]
    else:
        return sig, t
