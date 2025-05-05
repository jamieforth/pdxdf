from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import scipy.signal as signal


def nominal_sample_count(start, stop, fs):
    if stop < start:
        raise ValueError(
            f"stop ({stop}) cannot be earlier than start ({start})."
        )
    count = int(Decimal((stop - start) * fs).quantize(0, ROUND_HALF_UP))
    return count + 1


def nominal_sample_offset(start, start_min, fs):
    if start < start_min:
        raise ValueError(
            f"start ({start}) cannot be earlier than start_min ({start_min})."
        )
    return int(Decimal((start - start_min) * fs).quantize(0, ROUND_HALF_UP))


def nominal_sample_index(start, stop, fs, endpoint=True, segment=None):
    """
    start: Start time in seconds
    stop: End point in seconds
    fs: Sampling frequency
    endpoint: Include end point
    segment: Segment index value
    """
    num = nominal_sample_count(start, stop, fs)
    #stop = stop * fs
    #tdiff = 1 / fs
    if not endpoint:
        num -= 1
    t = pd.Series(
        np.arange(num) / fs + start,
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


def interp(df, fs_new, params={}, **kwargs):
    t_old = df.index.get_level_values("time_stamp")
    first_time = t_old.min()
    last_time = t_old.max()
    t_new = nominal_sample_index(first_time, last_time, fs_new)
    n_samples = t_new.shape[0]
    n_chans = df.shape[1]
    aligned = np.full((n_samples, n_chans), np.nan)
    for i in range(n_chans):
        aligned[:, i] = np.interp(
            t_new.values, t_old.values, df.iloc[:, i].values, **params
        )
    return aligned, first_time


def resample_fft(df, fs_new, params={}, **kwargs):
    first_time = df.index.get_level_values("time_stamp").min()
    last_time = df.index.get_level_values("time_stamp").max()
    num = nominal_sample_count(first_time, last_time, fs_new)
    return signal.resample(df, num, **params), first_time


def resample_stream(
    df,
    fs_old,
    fs_new,
    first_time_min,
    last_time_max,
    fn=resample_fft,
    params={},
):
    resampled = df.groupby(level="segment").apply(
        lambda seg: fn(seg, fs_old=fs_old, fs_new=fs_new, params=params)
    )

    n_samples = nominal_sample_count(first_time_min, last_time_max, fs_new)
    n_chans = df.shape[1]
    aligned = np.full((n_samples, n_chans), np.nan)

    for x_new, first_time in resampled:
        start_i = nominal_sample_offset(first_time, first_time_min, fs_new)
        end_i = start_i + x_new.shape[0]
        aligned[start_i:end_i] = x_new

    aligned = pd.DataFrame(aligned, columns=df.columns)
    aligned.index.rename("sample", inplace=True)
    aligned.attrs.update(df.attrs)
    aligned.attrs.update({"resample_params": {"fs": fs_new}})
    return aligned
