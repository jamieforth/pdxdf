import numpy as np
import pandas as pd


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


def nominal_ts_index(start, stop, fs, endpoint=False, segment=None):
    """
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    segment: Segment index value
    """
    tdiff = 1 / fs
    if endpoint:
        t = pd.Series(
            np.arange(start, stop + tdiff / 2, tdiff),
            name="time_stamp",
            dtype=np.float64,
        )
    else:
        t = pd.Series(
            np.arange(start, stop, tdiff),
            name="time_stamp",
            dtype=np.float64,
        )
    if segment is None:
        t.index.set_names("sample", inplace=True)
    else:
        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(segment, t.shape[0]),
                t.index,
            ],
            names=["segment", "sample"],
        )
        t.index = index
    return t


def sine(
    freq=1,
    amp=1,
    phase=0,
    start=0,
    stop=1,
    fs=100,
    endpoint=False,
    dtype=None,
    segment=None,
):
    """
    freq: Fundamental frequency (Hz)
    amp: Amplitude
    phase: Phase in radians (range: 0-2pi)
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    segment: Segment index value
    """
    period = 2 * np.pi

    t = nominal_ts_index(start, stop, fs, endpoint=endpoint, segment=segment)
    sig = pd.Series(
        np.sin(phase + (period * freq * t)) * amp, name=f"sine {freq}Hz", dtype=dtype
    )
    return sig, t


def counter(start=0, stop=1, fs=100, endpoint=False, dtype=None, segment=None):
    """
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    segment: Segment index value
    """

    t = nominal_ts_index(start, stop, fs, endpoint=endpoint, segment=segment)
    sig = pd.Series(
        np.arange(0, t.shape[0], 1),
        name=f"counter 1",
        dtype=dtype,
        index=t.index,
    )
    return sig, t
