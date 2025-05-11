from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import scipy.signal as signal


def nominal_sample_count(start, stop, fs):
    if stop < start:
        raise ValueError(f"stop ({stop}) cannot be earlier than start ({start}).")
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


def duration(n_samples, fs):
    return n_samples / fs

def resample_count(n_samples, fs_old, fs_new):
    return int((n_samples * Decimal(fs_new) / Decimal(fs_old)).quantize(0, ROUND_HALF_UP))

def resample_index(n_samples, start, fs_old, fs_new):
    n_samples = resample_count(n_samples, fs_old, fs_new)
    return (np.arange(0, n_samples) / fs_new) + start

def interp(df, fs_old, fs_new, **kwargs):
    t_old = df.index.get_level_values("time_stamp")
    first_time = t_old.min()
    last_time = t_old.max()
    t_new = resample_index(
        nominal_sample_count(first_time, last_time, fs_old),
        first_time,
        fs_old,
        fs_new,
    )
    n_samples = t_new.shape[0]
    n_chans = df.shape[1]
    aligned = np.full((n_samples, n_chans), np.nan)
    for i in range(n_chans):
        aligned[:, i] = np.interp(
            t_new, t_old.values, df.iloc[:, i].values, **kwargs
        )
    return aligned, first_time


def resample_fft(df, fs_old, fs_new, **kwargs):
    first_time = df.index.get_level_values("time_stamp").min()
    last_time = df.index.get_level_values("time_stamp").max()
    num = resample_count(
        nominal_sample_count(first_time, last_time, fs_old),
        fs_old,
        fs_new,
    )
    return signal.resample(df, num, **kwargs), first_time


def align_segment_start(df, first_time_min, fs_new):
    first_time = df.index.get_level_values("time_stamp").min()
    sample_offset = nominal_sample_offset(first_time, first_time_min, fs_new)
    if sample_offset != 0:
        time_offset = sample_offset / fs_new
        aligned_start_time = first_time_min + time_offset
        shift = first_time - aligned_start_time
        df.reset_index("time_stamp", inplace=True)
        df["time_stamp"] = df["time_stamp"] - shift
        df.set_index("time_stamp", append=True, inplace=True)
    return df


def resample_stream(
    df,
    fs_old,
    fs_new,
    first_time_min,
    fn=resample_fft,
    **kwargs,
):
    # Shift first timestamp to closest resampled index.
    df = df.groupby(level="segment", group_keys=False).apply(
        lambda seg: align_segment_start(seg, first_time_min, fs_new)
    )
    # Recalculate last_time_max in case the last stream rounds up when aligned
    # to the closest resampled index - should only make the resampled data one
    # sample longer than the synchronised `last_time_max`.
    last_time_max = df.index.get_level_values("time_stamp").max()

    resampled = df.groupby(level="segment").apply(
        lambda seg: fn(seg, fs_old=fs_old, fs_new=fs_new, **kwargs)
    )
    n_samples = resample_count(
        nominal_sample_count(first_time_min, last_time_max, fs_old),
        fs_old,
        fs_new,
    )
    n_chans = df.shape[1]
    aligned = np.full((n_samples, n_chans), np.nan)

    for x_new, first_time in resampled:
        start_i = nominal_sample_offset(first_time, first_time_min, fs_new)
        end_i = start_i + x_new.shape[0]
        aligned[start_i:end_i] = x_new

    aligned = pd.DataFrame(aligned, columns=df.columns)
    aligned.index.rename("sample", inplace=True)
    aligned.attrs.update(df.attrs)
    aligned.attrs.update({"resample_params": {"fs": fs_new} | kwargs})
    return aligned
