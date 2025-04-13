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


def nominal_ts_index(start, stop, fs, endpoint=False):
    """
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
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
    t.index.set_names("sample", inplace=True)
    return t


def sine(freq=1, amp=1, phase=0, start=0, stop=1, fs=100, endpoint=False, dtype=None):
    """
    freq: Fundamental frequency (Hz)
    amp: Amplitude
    phase: Phase in radians (range: 0-2pi)
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    """
    period = 2 * np.pi

    t = nominal_ts_index(start, stop, fs, endpoint=endpoint)
    sig = pd.Series(
        np.sin(phase + (period * freq * t)) * amp, name=f"sine {freq}Hz", dtype=dtype
    )
    return sig, t


def counter(start=0, stop=1, step=1, fs=100, endpoint=False, dtype=None):
    """
    start: Start time in seconds
    stop: End point in seconds
    step: Step interval
    fs: Sampling frequency
    endpoint: Include end point
    """

    t = nominal_ts_index(start, stop, fs, endpoint=endpoint)
    if endpoint:
        sig = pd.Series(
            np.arange(start, stop * fs + 1, step), name=f"counter {step}", dtype=dtype
        )
    else:
        sig = pd.Series(
            np.arange(start, stop * fs, step), name=f"counter {step}", dtype=dtype
        )
    sig.index.set_names("sample", inplace=True)
    return sig, t
