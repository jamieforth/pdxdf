from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdxdf import Xdf
from pdxdf.testing import counter, lslfmt2np, nominal_ts_index, sine

# requires git clone https://github.com/jamieforth/example-files.git into the
# root pdxdf folder
path = Path("example-files")
files = {
    key: path / value
    for key, value in {
        "resampling": "resampling.xdf",
    }.items()
    if (path / value).exists()
}


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

    header = pd.Series(
        {
            "version": "1.0",
            "datetime": pd.to_datetime("2025-03-23 18:33:35+00:00"),
        }
    )
    pd.testing.assert_series_equal(xdf.header(), header)

    info = pd.DataFrame(
        {
            "name": {
                1: "Resample: Test data stream 0",
                2: "Resample: Test marker stream 0",
                3: "Resample: Test data stream 1",
                4: "Resample: Test marker stream 1",
            },
            "type": {1: "eeg", 2: "marker", 3: "eeg", 4: "marker"},
            "channel_count": {1: 8, 2: 1, 3: 8, 4: 1},
            "channel_format": {1: "double64", 2: "string", 3: "double64", 4: "string"},
            "source_id": {
                1: "test_stream.py:106940",
                2: "test_stream.py:106942",
                3: "test_stream.py:106941",
                4: "test_stream.py:106943",
            },
            "nominal_srate": {1: 512.0, 2: 0.0, 3: 512.0, 4: 0.0},
            "version": {
                1: "1.100000000000000",
                2: "1.100000000000000",
                3: "1.100000000000000",
                4: "1.100000000000000",
            },
            "created_at": {
                1: "80693.65084625500",
                2: "80693.65198860300",
                3: "80693.65176273401",
                4: "80693.65221820401",
            },
            "uid": {
                1: "08b8c9ed-2ff4-4147-a451-62027e8d43d3",
                2: "f380306e-ea4c-42f1-af9f-34143e238d43",
                3: "3e1ef77a-8c5b-43d5-830d-4b62b74fe233",
                4: "8d489ab3-8435-438c-b6bf-48c9abc25c62",
            },
            "session_id": {1: "default", 2: "default", 3: "default", 4: "default"},
            "hostname": {1: "kassia", 2: "kassia", 3: "kassia", 4: "kassia"},
            "v4address": {1: None, 2: None, 3: None, 4: None},
            "v4data_port": {1: 16572, 2: 16576, 3: 16574, 4: 16578},
            "v4service_port": {1: 16572, 2: 16576, 3: 16574, 4: 16578},
            "v6address": {1: None, 2: None, 3: None, 4: None},
            "v6data_port": {1: 16573, 2: 16577, 3: 16575, 4: 16579},
            "v6service_port": {1: 16573, 2: 16577, 3: 16575, 4: 16579},
            "effective_srate": {
                1: 512 if not synchronize_clocks else 511.999902,
                2: 0.0,
                3: 512 if not synchronize_clocks else 511.999904,
                4: 0.0,
            },
        },
    )
    info.index.set_names("stream_id", inplace=True)
    info = info.astype(
        {
            "channel_count": np.int16,
            "v4data_port": np.int32,
            "v4service_port": np.int32,
            "v6data_port": np.int32,
            "v6service_port": np.int32,
        },
    )
    pd.testing.assert_frame_equal(xdf.info(), info, check_index_type=False, rtol=1e-09)

    footer = pd.DataFrame(
        {
            "first_timestamp": {
                1: 80754.151175457,
                2: 80754.151175457,
                3: 80754.151175457,
                4: 80754.151175457,
            },
            "last_timestamp": {
                1: 80814.151175457,
                2: 80814.151175457,
                3: 80814.151175457,
                4: 80814.151175457,
            },
            "sample_count": {
                1: 30721,
                2: 61,
                3: 30721,
                4: 61,
            },
        },
    )
    footer.index.set_names("stream_id", inplace=True)
    pd.testing.assert_frame_equal(xdf.footer(), footer)

    # Data vs. marker streams.
    assert not xdf.is_marker_stream(1)
    assert xdf.is_marker_stream(2)
    assert not xdf.is_marker_stream(3)
    assert xdf.is_marker_stream(4)

    # Segments
    segments = {1: [(0, 30720)], 2: [(0, 60)], 3: [(0, 30720)], 4: [(0, 60)]}
    assert xdf.segments() == segments
    if synchronize_clocks:
        assert xdf.clock_segments() == segments
    else:
        assert xdf.clock_segments() == {1: [], 2: [], 3: [], 4: []}

    if synchronize_clocks:
        segment_info = pd.DataFrame(
            {
                "segments": {1: 1, 2: 1, 3: 1, 4: 1},
                "clock_segments": {1: 1, 2: 1, 3: 1, 4: 1},
            },
        )
        pd.testing.assert_frame_equal(xdf.segment_info(), segment_info)
    else:
        segment_info = pd.DataFrame(
            {
                "segments": {1: 1, 2: 1, 3: 1, 4: 1},
                "clock_segments": {1: 0, 2: 0, 3: 0, 4: 0},
            },
        )
        pd.testing.assert_frame_equal(xdf.segment_info(), segment_info)

    # Channels
    data_channels = pd.DataFrame(
        {
            "label": {
                0: "sine 1Hz",
                1: "sine 2Hz",
                2: "sine 4Hz",
                3: "sine 8Hz",
                4: "sine 16Hz",
                5: "sine 32Hz",
                6: "sine 64Hz",
                7: "counter 1",
            },
            "type": {
                0: "misc",
                1: "misc",
                2: "misc",
                3: "misc",
                4: "misc",
                5: "misc",
                6: "misc",
                7: "counter",
            },
        },
    )
    data_channels.index.set_names("channel", inplace=True)
    marker_channels = pd.DataFrame(
        {
            "label": {0: "counter 1"},
            "type": {0: "counter"},
        },
    )
    marker_channels.index.set_names("channel", inplace=True)

    for stream_id, channel_info in xdf.channel_info().items():
        if not xdf.is_marker_stream(stream_id):
            pd.testing.assert_frame_equal(channel_info, data_channels)
        else:
            pd.testing.assert_frame_equal(channel_info, marker_channels)

    # Clock offsets
    clock_offsets_expected = {
        1: {
            "first_clock_offset": {
                "time": 80715.102474946,
                "value": -3.2353993447031826e-05,
            },
            "last_clock_offset": {
                "time": 80825.106114817,
                "value": -4.5650995161850005e-05,
            },
        },
        2: {
            "first_clock_offset": {
                "time": 80715.10266161899,
                "value": -1.568199513712898e-05,
            },
            "last_clock_offset": {
                "time": 80825.10609295151,
                "value": -2.3822503862902522e-05,
            },
        },
        3: {
            "first_clock_offset": {
                "time": 80715.10248696551,
                "value": -2.7883499569725245e-05,
            },
            "last_clock_offset": {
                "time": 80825.1060673295,
                "value": -2.380350633757189e-05,
            },
        },
        4: {
            "first_clock_offset": {
                "time": 80715.102545386,
                "value": -3.144900256302208e-05,
            },
            "last_clock_offset": {
                "time": 80825.106114797,
                "value": -4.56389898317866e-05,
            },
        },
    }

    for stream_id, clock_offsets in xdf.clock_offsets(with_stream_id=True).items():
        pd.testing.assert_series_equal(
            clock_offsets.iloc[0],
            pd.Series(clock_offsets_expected[stream_id]["first_clock_offset"]),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            clock_offsets.iloc[-1],
            pd.Series(clock_offsets_expected[stream_id]["last_clock_offset"]),
            check_names=False,
        )

    # Time-stamps: Data streams
    for stream_id, time_stamps in xdf.time_stamps(1, 3).items():
        nominal_ts = nominal_ts_index(
            xdf.time_stamps(stream_id).iloc[0],
            xdf.time_stamps(stream_id).iloc[-1],
            xdf.info(stream_id)["nominal_srate"].item(),
            endpoint=True,
        )
        if not synchronize_clocks:
            if not dejitter_timestamps:
                pd.testing.assert_series_equal(
                    xdf.time_stamps(stream_id), nominal_ts, check_exact=True
                )
            else:
                pd.testing.assert_series_equal(
                    xdf.time_stamps(stream_id), nominal_ts, rtol=1e-10, atol=1e-20
                )
        else:
            pd.testing.assert_series_equal(
                xdf.time_stamps(stream_id), nominal_ts, rtol=1e-09, atol=1e-12
            )

    # Time-stamps: Marker streams
    for stream_id, time_stamps in xdf.time_stamps(2, 4).items():
        nominal_ts = nominal_ts_index(
            xdf.time_stamps(stream_id).iloc[0],
            xdf.time_stamps(stream_id).iloc[-1],
            1,  # Defined in test
            endpoint=True,
        )
        if not synchronize_clocks:
            if not dejitter_timestamps:
                pd.testing.assert_series_equal(
                    xdf.time_stamps(stream_id), nominal_ts, check_exact=True
                )
            else:
                pd.testing.assert_series_equal(
                    xdf.time_stamps(stream_id), nominal_ts, check_exact=True
                )
        else:
            pd.testing.assert_series_equal(
                xdf.time_stamps(stream_id), nominal_ts, rtol=1e-09, atol=1e-12
            )

    # Time-series: Data streams
    ch_freq = 2 ** np.array(range(0, 7))
    for stream_id, ts in xdf.time_series(1, 3).items():
        fs = xdf.info(stream_id)["nominal_srate"].item()
        dtype = lslfmt2np(xdf.info(stream_id)["channel_format"].item())
        for f in ch_freq:
            sig = ts[f"sine {f}Hz"]
            sig_expected, _ = sine(f, stop=60, fs=fs, endpoint=True, dtype=dtype)
            pd.testing.assert_series_equal(sig, sig_expected)
        count = ts["counter 1"]
        count_expected, _ = counter(start=0, stop=60, fs=fs, endpoint=True, dtype=dtype)
        pd.testing.assert_series_equal(count, count_expected)

    # Time-series: Marker streams
    for stream_id, ts in xdf.time_series(2, 4).items():
        fs = 1  # Defined in test
        dtype = lslfmt2np(xdf.info(stream_id)["channel_format"].item())
        marker = ts[f"counter {fs}"]
        marker_expected, _ = counter(
            start=0, stop=60, fs=fs, endpoint=True, dtype=dtype
        )
        pd.testing.assert_series_equal(marker, marker_expected)
