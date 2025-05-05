from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdxdf import Xdf
from pdxdf.resampling import (
    nominal_sample_count,
    nominal_sample_index,
    nominal_sample_offset,
    resample_stream,
    resample_fft,
    interp,
)
from pdxdf.testing import counter, sine


# requires git clone https://github.com/jamieforth/example-files.git into the
# root pdxdf folder
path = Path("example-files")
files = {
    key: path / value
    for key, value in {
        "resampling": "resampling.xdf",
        "clock_resets": "clock_resets.xdf",
    }.items()
    if (path / value).exists()
}


@pytest.mark.parametrize(
    "first_time, last_time, fs, expected",
    [
        # fs=1
        (0, 0, 1, 1),
        (0, 1, 1, 2),
        (-1, 0, 1, 2),
        (-2, -1, 1, 2),
        (0, 1.49, 1, 2),
        (0, 1.5, 1, 3),  # rounds up
        (1, 2.49, 1, 2),
        (1, 2.5, 1, 3),  # rounds up
        # fs=2
        (0, 1, 2, 3),
        (0, 1.5, 2, 4),  # rounds up
        (1, 3, 2, 5),
        (1, 3.5, 2, 6),  # rounds up
        # fs=100
        (0, Decimal(1) / 100, 100, 2),
        (0, Decimal(1) / 100 - Decimal(1) / 100 / 2, 100, 2),  # rounds up
        (0, Decimal(1) / 100 + Decimal(1) / 100 / 2, 100, 3),  # rounds up
        (0, 1, 100, 101),
        (0, 1 - Decimal(1) / 100 / 2, 100, 101),  # rounds up
        (0, 1 + Decimal(1) / 100 / 2, 100, 102),  # rounds up
        (-1, 1, 100, 201),
        (-1, 1 - Decimal(1) / 100 / 2, 100, 201),  # rounds up
        (-1, 1 + Decimal(1) / 100 / 2, 100, 202),  # rounds up
        # fs=512
        (0, Decimal(1) / 512, 512, 2),
        (0, Decimal(1) / 512 - Decimal(1) / 512 / 2, 512, 2),  # rounds up
        (0, Decimal(1) / 512 + Decimal(1) / 512 / 2, 512, 3),  # rounds up
        (0, 1, 512, 513),
        (0, 1 - Decimal(1) / 512 / 2, 512, 513),  # rounds up
        (0, 1 + Decimal(1) / 512 / 2, 512, 514),  # rounds up
        (-1, 1, 512, 1025),
        (-1, 1 - Decimal(1) / 512 / 2, 512, 1025),  # rounds up
        (-1, 1 + Decimal(1) / 512 / 2, 512, 1026),  # rounds up
        (0, 60, 512, 30721),
    ],
)
def test_nominal_sample_count(first_time, last_time, fs, expected):
    assert nominal_sample_count(first_time, last_time, fs) == expected


@pytest.mark.parametrize("endpoint", ([True, False]))
@pytest.mark.parametrize(
    "first_time, last_time, fs",
    [
        # fs=1
        (0, 1, 1),
        (-1, 0, 1),
        (-2, -1, 1),
        (0, 1.49, 1),
        (0, 1.5, 1),  # rounds up
        (0, 2.49, 1),
        (0, 2.5, 1),  # rounds up
        # fs=2
        (0, 1, 2),
        (0, 1.5, 2),  # rounds up
        (1, 3, 2),
        (1, 3.5, 2),  # rounds up
        # fs=100
        (0, Decimal(1) / 100, 100),
        (0, Decimal(1) / 100 - Decimal(1) / 100 / 2, 100),  # rounds up
        (0, Decimal(1) / 100 + Decimal(1) / 100 / 2, 100),  # rounds up
        (0, 1, 100),
        (0, 1 - Decimal(1) / 100 / 2, 100),  # rounds up
        (0, 1 + Decimal(1) / 100 / 2, 100),  # rounds up
        (-1, 1, 100),
        (-1, 1 - Decimal(1) / 100 / 2, 100),  # rounds up
        (-1, 1 + Decimal(1) / 100 / 2, 100),  # rounds up
        # fs=512
        (0, Decimal(1) / 512, 512),
        (0, Decimal(1) / 512 - Decimal(1) / 512 / 2, 512),  # rounds up
        (0, Decimal(1) / 512 + Decimal(1) / 512 / 2, 512),  # rounds up
        (0, 1, 512),
        (0, 1 - Decimal(1) / 512 / 2, 512),  # rounds up
        (0, 1 + Decimal(1) / 512 / 2, 512),  # rounds up
        (-1, 1, 512),
        (-1, 1 - Decimal(1) / 512 / 2, 512),  # rounds up
        (-1, 1 + Decimal(1) / 512 / 2, 512),  # rounds up
        (0, 60, 512),
    ],
)
def test_nominal_sample_index(first_time, last_time, fs, endpoint):
    index = nominal_sample_index(first_time, last_time, fs, endpoint)
    assert index.iloc[0] == first_time
    if endpoint:
        assert index.size == nominal_sample_count(first_time, last_time, fs)
        assert Decimal(index.iloc[-1]) == pytest.approx(
            Decimal(last_time), abs=(1 / fs / 2) + np.finfo(float).eps
        )
    else:
        assert index.size == nominal_sample_count(first_time, last_time, fs) - 1


@pytest.mark.parametrize(
    "first_time, first_time_min, fs, expected",
    [
        # fs=1
        (0, 0, 1, 0),
        (1, 1, 1, 0),
        (-1, -1, 1, 0),
        (0.49, 0, 1, 0),
        (0.5, 0, 1, 1),  # rounds up
        (1, 0, 1, 1),
        (1, -1, 1, 2),
        (1.49, 0, 1, 1),
        (1.5, 0, 1, 2),  # rounds up
        (2, 0, 1, 2),
        (2.49, 0, 1, 2),
        (2.5, 0, 1, 3),  # rounds up
        # fs=2
        (0, 0, 2, 0),
        (1, 1, 2, 0),
        (-1, -1, 2, 0),
        (0.249, 0, 2, 0),
        (0.25, 0, 2, 1),  # rounds up
        (1, 0, 2, 2),
        (1, -1, 2, 4),
        (1.249, 0, 2, 2),
        (1.25, 0, 2, 3),  # rounds up
        (1.5, 0, 2, 3),
        (1.749, 0, 2, 3),
        (1.75, 0, 2, 4),  # rounds up
        (2, 0, 2, 4),
        # fs=100
        (1, 0, 100, 100),
        (1 - 1 / 100, 0, 100, 99),
        (1 - (1 / 100 / 2) - np.finfo(float).eps, 0, 100, 99),
        (1 - (1 / 100 / 2), 0, 100, 100),  # rounds up
        (2, 0, 100, 2 * 100),
        (3, 1, 100, 2 * 100),
        # fs=512
        (1, 0, 512, 512),
        (1 - 1 / 512, 0, 512, 511),
        (1 - (1 / 512 / 2) - np.finfo(float).eps, 0, 512, 511),
        (1 - (1 / 512 / 2), 0, 512, 512),  # rounds up
        (60, 0, 512, 60 * 512),
    ],
)
def test_nominal_sample_offset(first_time, first_time_min, fs, expected):
    assert nominal_sample_offset(first_time, first_time_min, fs) == expected


@pytest.mark.parametrize(
    "fn",
    [
        resample_fft, interp
    ]
)
@pytest.mark.parametrize(
    "segment, fs",
    [
        # fs=1
        ((0, 0), 1),
        ((0, 1), 1),
        ((1, 1), 1),
        ((1, 2), 1),
        ((0, 10), 1),
        ((1, 10), 1),
        ((-1, 10), 1),
        # fs=2
        ((0, 0), 2),
        ((0, 1), 2),
        ((1, 1), 2),
        ((1, 2), 2),
        ((0, 10), 2),
        ((1, 10), 2),
        ((-1, 10), 2),
        # fs=100
        ((0, 1), 100),
        ((0, 60), 100),
        ((1, 60), 100),
        ((-1, 60), 100),
        # fs=512
        ((0, 1), 512),
        ((0, 60), 512),
        ((1, 60), 512),
        ((-1, 60), 512),
    ],
)
def test_resample_segment(segment, fs, fn):
    start, stop = segment
    if fn is resample_fft:
        s = sine(
            freq=1, start=start, stop=stop, fs=fs, endpoint=True, time_stamp_index=True
        )
    else:
        s = counter(start=start, stop=stop, fs=fs, endpoint=True, time_stamp_index=True)
    df = s.to_frame()
    df_new, first_time = fn(df, fs_new=fs, fs_old=fs)
    assert first_time == start
    assert df_new.shape == df.shape
    np.testing.assert_allclose(df_new, df, atol=1e-14)


@pytest.mark.parametrize(
    "fn",
    [
        resample_fft, interp
    ]
)
@pytest.mark.parametrize(
    "segments, fs",
    [
        # fs=1
        ([(0, 1), (1, 2)], 1),
        ([(0, 5), (5, 10)], 1),
        ([(0, 1), (1.1, 2.2)], 1),
        # fs=2
        ([(0, 5), (5, 10)], 2),
        # fs=100
        ([(0, 5), (5, 10)], 100),
        ([(0, 100), (100, 100)], 100),
    ],
)
def test_resample_stream(segments, fs, fn):
    if fn is resample_fft:
        s = pd.concat(
            [
                sine(
                    freq=1,
                    start=start,
                    stop=stop,
                    fs=fs,
                    endpoint=False,
                    segment=segment,
                    time_stamp_index=True,
                )
                for (start, stop), segment in zip(segments, range(0, len(segments)))
            ]
        )
    else:
        s = pd.concat(
            [
                counter(
                    start=start,
                    stop=stop,
                    fs=fs,
                    endpoint=False,
                    segment=segment,
                    time_stamp_index=True,
                )
                for (start, stop), segment in zip(segments, range(0, len(segments)))
            ]
        )
    df = s.to_frame()
    df_new = resample_stream(
        df,
        fs_old=fs,
        fs_new=fs,
        first_time_min=df.index.get_level_values("time_stamp").min(),
        last_time_max=df.index.get_level_values("time_stamp").max(),
    )
    np.testing.assert_allclose(
        df_new,
        df.droplevel(["segment", "time_stamp"]),
        atol=1e-14,
    )


@pytest.mark.skip
@pytest.mark.parametrize(
    "segments, fs, fs_new, expected",
    [
        # fs=1
        ([(0, 2), (2, 4)], 1, 1, [0, 1, 0, 1]),
        ([(0, 2), (2, 4)], 1, 2, [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]),
        # ([(0, 5), (5, 10)], 1),
        # ([(0, 1), (1.1, 2.2)], 1),
        # # fs=2
        # ([(0, 5), (5, 10)], 2),
        # # fs=100
        # ([(0, 5), (5, 10)], 100),
        # ([(0, 100), (100, 100)], 100),
    ],
)
def test_resample_fs(segments, fs, fs_new, expected):
    s = pd.concat(
        [
            counter(
                first_time, last_time, fs=fs, segment=segment, time_stamp_index=True
            )
            for (first_time, last_time), segment in zip(
                segments, range(0, len(segments))
            )
        ]
    )
    df = s.to_frame()
    print(df)
    df_new = resample_stream(
        df,
        fs,
        fs_new,
        df.index.get_level_values("time_stamp").min(),
        df.index.get_level_values("time_stamp").max(),
    )
    expected = pd.DataFrame(expected, columns=df.columns)
    expected.rename_axis("sample", inplace=True)
    print(df_new, expected)
    np.testing.assert_allclose(
        df_new,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


# @pytest.mark.parametrize("fs_new_ratio", [2, 1, 1 / 2, 1 / 4])
@pytest.mark.skip
@pytest.mark.parametrize("dejitter_timestamps", [False, True])
@pytest.mark.parametrize("synchronize_clocks", [False, True])
@pytest.mark.skipif("resampling" not in files, reason="File not found.")
def test_resample_file_parse(synchronize_clocks, dejitter_timestamps):
    path = files["resampling"]
    xdf = Xdf(path).load(
        synchronize_clocks=synchronize_clocks,
        dejitter_timestamps=dejitter_timestamps,
        channel_name_field="label",
    )


@pytest.mark.skip
@pytest.mark.skipif("clock_resets" not in files, reason="File not found.")
def test_resample_clock_resets():
    path = files["clock_resets"]
    xdf = Xdf(path).load(
        synchronize_clocks=True,
        dejitter_timestamps=True,
        channel_name_field="label",
    )
    df = xdf.data(concat=True)
    segment_size = df.groupby(level=["stream_id", "segment"]).size()
    pd.testing.assert_series_equal(segment_size, xdf.segment_size(), check_names=False)
