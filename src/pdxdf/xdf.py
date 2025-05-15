"""Main Xdf class for working with XDF data."""

import logging

import mne
import numpy as np
import pandas as pd

from .constants import microvolts
from .errors import NoLoadableStreamsError, XdfAlreadyLoadedError
from .rawxdf import RawXdf, XdfDecorators
from .resampling import resample_stream, resample_fft

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Xdf(RawXdf):
    """Main class for XDF data processing with pandas.

    Provides a pandas-based layer of abstraction over raw XDF data.
    """

    # Data types for XDF info.
    _info_types = {
        "channel_count": np.int16,
        "nominal_srate": np.float64,
        "v4data_port": np.int32,
        "v4service_port": np.int32,
        "v6data_port": np.int32,
        "v6service_port": np.int32,
        "effective_srate": np.float64,
    }

    _footer_types = {
        "first_timestamp": np.float64,
        "last_timestamp": np.float64,
        "sample_count": np.int64,
    }

    def __init__(self, filename, verbose=False):
        """Initialise XdfData via super class."""
        super().__init__(filename, verbose)

    def resolve_streams(self):
        """Return a DataFrame containing available streams.

        Results are not cached - the data is always read from file.
        """
        streams = pd.DataFrame(super().resolve_streams())
        streams.set_index("stream_id", inplace=True)
        streams.sort_index(inplace=True)
        return streams

    def load(
        self,
        *select_streams,
        channel_scale_field=None,
        channel_name_field=None,
        **kwargs,
    ):
        """Load XDF data from file using pyxdf.load_xdf().

        Any pyxdf.load_xdf() kwargs provided will be passed to that
        function. All other kwargs are assumed to be stream properties and will
        be passed to parsing methods.
        """
        try:
            self._load(
                *select_streams,
                channel_scale_field=channel_scale_field,
                channel_name_field=channel_name_field,
                **kwargs,
            )
        except (NoLoadableStreamsError, XdfAlreadyLoadedError) as exc:
            print(exc)
        return self

    @XdfDecorators.loaded
    def info(self, *stream_ids, exclude=[], cols=None, ignore_missing_cols=False):
        """Return stream info as a DataFrame.

        Select data for stream_ids or default all loaded streams.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._info,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
        )

    def is_marker_stream(self, stream_id):
        """Test if stream is a marker stream."""
        srate = self.info(stream_id)["nominal_srate"].item()
        return srate == 0

    def segment_counts(self, *stream_ids, exclude=[]):
        segment_counts = super().segment_counts(*stream_ids, exclude=exclude)
        df = pd.DataFrame(segment_counts._asdict())
        df.index.rename("stream_id", inplace=True)
        return df

    def segment_size(self, *stream_ids, exclude=[], min_segment=0):
        segment_size = super().segment_size(*stream_ids, exclude=exclude)
        segment_size = pd.Series(segment_size, name="segment_size")
        segment_size.index.rename(["stream_id", "segment"], inplace=True)
        if min_segment > 0:
            segment_size = segment_size.loc[segment_size > min_segment]
        return segment_size

    def clock_segment_size(self, *stream_ids, exclude=[]):
        segment_size = super().clock_segment_size(*stream_ids, exclude=exclude)
        if len(segment_size) > 0:
            segment_size = pd.Series(segment_size, name="clock_segment_size")
            segment_size.index.rename(["stream_id", "segment"], inplace=True)
            return segment_size
        else:
            return None

    def segment_index(self, stream_id):
        idx = super().segment_index(stream_id)
        idx = pd.MultiIndex.from_tuples(idx, names=("segment", "sample"))
        return idx

    @XdfDecorators.loaded
    def channel_info(
        self,
        *stream_ids,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        with_stream_id=False,
        concat=False,
    ):
        """Return channel info as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: DataFrame}
        where number of items is equal to the number of streams. Single streams
        are returned as is unless with_stream_id=True.

        When concat=True return data concatenated into a single DataFrame along
        columns.
        """
        if not self._channel_info:
            print("No channel info.")
            return None
        channel_info = self._get_stream_data(
            *stream_ids,
            data=self._channel_info,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=with_stream_id,
        )
        if isinstance(channel_info, dict) and concat:
            channel_info = pd.concat(channel_info, axis=1)
        return channel_info

    def channel_scalings(self, *stream_ids, channel_scale_field):
        """Return a dictionary of DataFrames with channel scaling values."""
        stream_units = self.channel_info(
            *stream_ids,
            cols=channel_scale_field,
            ignore_missing_cols=True,
            with_stream_id=True,
        )
        if stream_units is not None:
            scaling = {
                stream_id: ch_units.apply(
                    lambda units: [1e-6 if u in microvolts else 1 for u in units]
                )
                for stream_id, ch_units in stream_units.items()
            }
            return scaling

    @XdfDecorators.loaded
    def footer(self, *stream_ids, exclude=[], cols=None, ignore_missing_cols=False):
        """Return stream footer info as a DataFrame.

        Select data for stream_ids or default all loaded streams.
        """
        if self._footer is None:
            print("No footer data.")
            return None
        return self._get_stream_data(
            *stream_ids,
            data=self._footer,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
        )

    @XdfDecorators.loaded
    def clock_offsets(
        self,
        *stream_ids,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        with_stream_id=False,
    ):
        """Return clock offset data as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: DataFrame}
        where number of items is equal to the number of streams. Single streams
        are returned as is unless with_stream_id=True.
        """
        if not self._clock_offsets:
            print("No clock-offset data.")
            return None
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_offsets,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_series(
        self,
        *stream_ids,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        with_stream_id=False,
    ):
        """Return stream time-series data as a segmented DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: DataFrame}
        where number of items is equal to the number of streams. Single streams
        are returned as is unless with_stream_id=True.
        """
        if not self._time_series:
            print("No time-series data.")
            return None
        return self._get_stream_data(
            *stream_ids,
            exclude=exclude,
            data=self._time_series,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_stamps(self, *stream_ids, exclude=[], concat=False, with_stream_id=False):
        """Return stream time-stamps as a segmented Series.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: Series} where
        number of items is equal to the number of streams. Single streams are
        returned as is unless with_stream_id=True.

        When concat=True return data concatenated into a single Series along
        the index.
        """
        if not self._time_stamps:
            print("No time-stamp data.")
            return None
        ts = self._get_stream_data(
            *stream_ids,
            exclude=exclude,
            data=self._time_stamps,
            with_stream_id=True,
        )
        if concat:
            ts = pd.concat(ts, axis=0, names=["stream_id", "segment", "sample"])
            return ts
        else:
            return self._single_or_multi_stream_data(ts, with_stream_id)

    def data(
        self,
        *stream_ids,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        with_stream_id=False,
        concat=False,
    ):
        """Return stream time-series and time-stamps as DataFrames.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: DataFrame}
        where number of items is equal to the number of streams. Single streams
        are returned as is unless with_stream_id=True.

        When concat=True return data concatenated into a single DataFrame along
        the index.

        This does not align samples to a common time index, for that see
        `resample`.
        """
        time_series = self.time_series(
            *stream_ids,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=True,
        )
        if not time_series:
            return None
        time_stamps = self.time_stamps(
            *stream_ids, exclude=exclude, with_stream_id=True
        )
        if not time_stamps:
            return None

        ts = {
            stream_id: ts.join(time_stamps[stream_id])
            .set_index(
                "time_stamp",
                append=True,
            )
            .rename_axis(columns="channel")
            for stream_id, ts in time_series.items()
        }
        if concat:
            ts = pd.concat(ts, axis=0, names=["stream_id"]).sort_index()
            ts.attrs.update({"load_params": self.load_params})
            return ts
        else:
            for t in ts.values():
                t.attrs.update({"load_params": self.load_params})
            return self._single_or_multi_stream_data(ts, with_stream_id)

    def time_stamp_info(self, *stream_ids, exclude=[], min_segment=0):
        """Generate a summary of loaded time-stamp data."""
        time_stamps = self.time_stamps(*stream_ids, exclude=exclude, concat=True)
        if time_stamps.empty:
            return None
        data = time_stamps.groupby(level=["stream_id", "segment"]).agg(
            sample_count="count",
            first_timestamp="min",
            last_timestamp="max",
        )
        if min_segment > 0:
            data = data.loc[data["sample_count"] > min_segment]
        nominal_srate = self.info()["nominal_srate"]
        data = data.join(nominal_srate)
        data["nominal_duration"] = (data["sample_count"] - 1) / data["nominal_srate"]
        data["effective_duration"] = data["last_timestamp"] - data["first_timestamp"]
        data["effective_srate"] = (data["sample_count"] - 1) / data[
            "effective_duration"
        ]
        data["srate_ratio"] = data["effective_srate"] / data["nominal_srate"]
        data["duration_error"] = data["nominal_duration"] - data["effective_duration"]
        data.attrs.update({"load_params": self.load_params})
        return data

    def time_stamp_intervals(
        self,
        *stream_ids,
        exclude=[],
        min_segment=0,
    ):
        """Return time-stamp intervals for each stream.

        Multiple streams are returned as a single Series indexed by [stream_id,
        segment, sample].
        """
        time_stamps = self.time_stamps(*stream_ids, exclude=exclude, concat=True)
        if time_stamps.empty:
            return None
        if min_segment > 0:
            seg_size = self.segment_size(*stream_ids, exclude)
            time_stamps = time_stamps.loc[seg_size > min_segment]
        data = time_stamps.groupby(["stream_id", "segment"]).diff()
        return data

    @XdfDecorators.loaded
    def resample(
        self,
        *stream_ids,
        fs_max_ratio=None,
        fs_new=None,
        fn=resample_fft,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        concat=True,
        with_stream_id=False,
        **kwargs,
    ):
        """
        Resample multiple XDF streams to a given frequency.

        Parameters
        ----------
        stream_ids : list[int]
            The IDs of the desired streams.
        fs_max_ratio : int
            Resample at this ratio to the maximum stream nominal sample rate. Defaults
            to `1` if `fs_new` is not provided.
        fs_new : int
            Resampling target frequency in Hz.
        fn : function
            Resampling function to apply.
        concat : bool (default: True)
            When concat=True return resampled time-series as a single DataFrame,
            concatenated along columns, otherwise return a dictionary {stream_id: df}.
        kwargs
            Optional keyword arguments to pass to resampling function.

        Returns
        -------
        time_series, markers, new_fs : tuple
          - time_series:
              - DataFrame (num_samples, total_columns) when concat=True
              - dict {stream_id: DataFrame (num_samples , num_columns)} when concat=False
            DataFrames are sample-indexed. Where a stream has no data corresponding to
            the resampled sampled-index sample values are `np.nan`.
          - markers:
              - dict {stream_id: DataFrame (num_markers, num_channels)}
            DataFrames are time-indexed (seconds)
          - new_fs : int
            Either the provided `fs_new` value or `fs_new` calculated from
            `fs_max_ratio`.
        """
        if fs_max_ratio is not None and fs_new is not None:
            raise ValueError("Must provide either `fs_max_ratio` or `fs_new`.")
        if fs_new is None:
            if fs_max_ratio is None:
                fs_max_ratio = 1
            # Get fs_new from all loaded streams (except excluded).
            fs_new = int(
                self.info(cols="nominal_srate", exclude=exclude).max().item()
                * fs_max_ratio
            )
        print(f"Resampling to {fs_new} Hz.")

        data = self.data(
            *stream_ids,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=True,
        )
        if data is None:
            print("No data to resample.")
            return None

        # Get resample time-stamp range from all loaded streams (except
        # excluded).
        ts_info = self.time_stamp_info(exclude=exclude)
        first_time_min = ts_info["first_timestamp"].min()

        all_resampled = {}
        all_markers = {}

        for stream_id, df in data.items():
            if df.empty:
                # Skip empty streams.
                continue
            if not self.is_marker_stream(stream_id):
                # Regular sample-rate stream.
                fs = self.info(stream_id)["nominal_srate"].item()
                resampled = resample_stream(
                    df,
                    fs_old=fs,
                    fs_new=fs_new,
                    first_time_min=first_time_min,
                    fn=fn,
                    **kwargs,
                )
                all_resampled[stream_id] = resampled
            else:
                # Marker stream.
                df = df.droplevel(["segment", "sample"])
                df.index = df.index - first_time_min
                all_markers[stream_id] = df

        if concat:
            all_resampled = pd.concat(all_resampled, axis="columns")
            all_resampled.columns.rename("stream", level=0, inplace=True)
            return (
                all_resampled,
                all_markers,
                fs_new,
            )
        else:
            return (
                self._single_or_multi_stream_data(all_resampled, with_stream_id),
                all_markers,
                fs_new,
            )

    def raw_mne(
        self,
        *stream_ids,
        fs_max_ratio=None,
        fs_new=None,
        fn=resample_fft,
        exclude=[],
        cols=None,
        channel_info_map=None,
        annotation_fn=None,
        ignore_missing_cols=False,
        with_stream_id=False,
        **kwargs,
    ):
        """Return mne.io.Raw objects from XDF streams.

        Multiple streams are returned as a dictionary {stream_id: DataFrame}
        where number of items is equal to the number of streams. Single streams
        are returned as is unless with_stream_id=True.
        """
        data, markers, fs = self.resample(
            *stream_ids,
            fs_max_ratio=fs_max_ratio,
            fs_new=fs_new,
            fn=fn,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            concat=False,
            with_stream_id=True,
            **kwargs,
        )
        data = {
            stream_id: self._xdf_to_mne(
                stream_id,
                df,
                markers,
                fs,
                channel_info_map=channel_info_map,
                annotation_fn=annotation_fn,
            )
            for stream_id, df in data.items()
        }
        return self._single_or_multi_stream_data(data, with_stream_id), markers

    def _xdf_to_mne(
        self, stream_id, df, markers, fs, channel_info_map=None, annotation_fn=None
    ):
        channels = None
        channel_info = self.channel_info(stream_id)
        if channel_info is not None:
            channels = channel_info.loc[channel_info["label"].isin(df.columns)]
            if channel_info_map is not None:
                channels = channels.replace(channel_info_map)

        if channels is not None and "label" in channels:
            # MNE info requires a list of channel names or the number of channels.
            channel_names = list(channels["label"])
        else:
            channel_names = df.shape[1]
        if channels is not None and "type" in channels:
            # MNE info requires a list of channel types. If not available it
            # defaults to 'misc', so we do the same.
            channel_types = list(channels["type"])
        else:
            channel_types = "misc"

        ts = df.T
        info = mne.create_info(channel_names, fs, channel_types)
        orig_time = self.header()["datetime"].tz_convert('utc')
        info.set_meas_date(orig_time)
        raw = mne.io.RawArray(ts, info)
        if annotation_fn is not None:
            annotations = annotation_fn(markers, orig_time)
            raw.annotations.append(
                annotations.onset,
                annotations.duration,
                annotations.description,
            )
        else:
            # Default extract first marker channels.
            annotations = [
                mne.Annotations(
                    marker_stream.index,
                    0,
                    marker_stream.iloc[:, 0],
                    orig_time=orig_time,
                )
                for marker_stream in markers.values()
            ]
            for annot in annotations:
                raw.annotations.append(
                    annot.onset,
                    annot.duration,
                    annot.description,
                )
        return raw

    # Non public methods.

    def _parse_header(self, data, **kwargs):
        """Convert raw header into a DataFrame."""
        header = super()._parse_header(data)
        header = pd.Series(header)
        if "datetime" in header:
            header["datetime"] = pd.to_datetime(header["datetime"])
        return header

    def _parse_info(self, data, **kwargs):
        """Parse info for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within this instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a DataFrame.
        """
        data = super()._parse_info(data)
        df = pd.DataFrame(data).T
        try:
            df = df.astype(self._info_types)
        except KeyError:
            # Don't throw an error if info does not contain all columns
            # specified in info types.
            pass
        assert all(df.index == df["stream_id"])
        df.set_index("stream_id", inplace=True)
        return df

    def _parse_channel_info(self, data, **kwargs):
        """Parse channel info for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        # Check that data streams have valid channel info.
        data = super()._parse_channel_info(data)
        data = self._check_empty_streams(data, "channel info")
        if not data:
            return None
        # Handle streams with only a single channel.
        data = {k: [v] if isinstance(v, dict) else v for k, v in data.items()}
        data = self._to_DataFrames(data, "channel")
        return data

    def _parse_footer(self, data, **kwargs):
        """Parse footer for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_footer(data)
        data = self._check_empty_streams(data, "footer")
        if not data:
            return None
        df = pd.DataFrame(data).T
        df = df.astype(self._footer_types)
        df.index.name = "stream_id"
        return df

    def _parse_clock_offsets(self, data, **kwargs):
        """Parse clock offsets for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_clock_offsets(data)
        data = self._to_DataFrames(data, "period")
        return data

    def _parse_time_series(
        self, data, channel_scale_field, channel_name_field, **kwargs
    ):
        """Parse time-series data for all loaded streams into a DataFrame.

        Optionally scales values and sets channel names according to channel
        info.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_time_series(data)
        data = self._to_DataFrames(data, segment_index=True, col_index_name="channel")

        if channel_scale_field is not None:
            scalings = self.channel_scalings(channel_scale_field=channel_scale_field)
            if scalings:
                data = {
                    stream_id: df * scalings[stream_id][channel_scale_field]
                    if (
                        stream_id in scalings
                        and not (scalings[stream_id] == 1).all().item()
                    )
                    else df
                    for stream_id, df in data.items()
                }

        if channel_name_field is not None:
            ch_labels = self.channel_info(cols=channel_name_field, with_stream_id=True)
            if ch_labels:
                data = {
                    stream_id: df.rename(
                        columns=ch_labels[stream_id].loc[:, channel_name_field]
                    )
                    if stream_id in ch_labels
                    else df
                    for stream_id, df in data.items()
                }

        return data

    def _parse_time_stamps(self, data, **kwargs):
        """Parse time-stamps for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_time_stamps(data)
        data = self._to_Series(data, name="time_stamp", segment_index=True)
        return data

    def _get_stream_data(
        self,
        *stream_ids,
        data,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        with_stream_id=False,
    ):
        if isinstance(data, dict):
            data = super()._get_stream_data(
                *stream_ids, data=data, exclude=exclude, with_stream_id=with_stream_id
            )
        elif isinstance(data, pd.DataFrame):
            if not isinstance(exclude, list):
                exclude = [exclude]
            if stream_ids:
                stream_ids = set(stream_ids)
                stream_ids = stream_ids - set(exclude)
                if set(self.loaded_stream_ids) != stream_ids:
                    data = data.loc[list(stream_ids), :]
            elif len(exclude) > 0:
                data = data.loc[~data.index.isin(exclude)]
        else:
            raise ValueError("Data should be a dictionary or DataFrame")
        # Subset data columns.
        if data is not None and cols is not None:
            if not isinstance(cols, list):
                cols = [cols]
            if isinstance(data, dict):
                subset = {}
                for stream_id in data.keys():
                    df_cols = self._check_columns(
                        data[stream_id], cols, ignore_missing_cols
                    )
                    if df_cols:
                        subset[stream_id] = data[stream_id].loc[:, df_cols]
                data = subset
            else:
                df_cols = self._check_columns(data, cols, ignore_missing_cols)
                data = data.loc[:, df_cols]
        return data

    def _to_Series(self, data, name, index_name=None, segment_index=False):
        # Map a dictionary of {stream-id: data} to a dictionary of {stream-id:
        # Series}.
        data = {
            stream_id: self._to_s(
                stream_id,
                d,
                name=name,
                index_name=index_name,
                segment_index=segment_index,
            )
            for stream_id, d in data.items()
        }
        return data

    def _to_s(self, stream_id, data, name, index_name=None, segment_index=False):
        if segment_index:
            idx = self.segment_index(stream_id)
            s = pd.Series(data, index=idx, name=name)
        else:
            s = pd.Series(data, name=name)
            s.index.set_names(index_name, inplace=True)
        s.attrs.update({"load_params": self.load_params})
        return s

    def _to_DataFrames(
        self, data, index_name=None, segment_index=False, col_index_name=None
    ):
        # Map a dictionary of {stream-id: data} to a dictionary of {stream-id:
        # DataFrames}.
        data = {
            stream_id: self._to_df(
                stream_id,
                d,
                index_name,
                segment_index=segment_index,
                col_index_name=col_index_name,
            )
            for stream_id, d in data.items()
        }
        return data

    def _to_df(
        self, stream_id, data, index_name=None, segment_index=False, col_index_name=None
    ):
        if segment_index:
            idx = self.segment_index(stream_id)
            df = pd.DataFrame(data, idx)
        else:
            df = pd.DataFrame(data)
            df.index.set_names(index_name, inplace=True)
        if col_index_name:
            df.columns.set_names(col_index_name, inplace=True)
        df.attrs.update({"load_params": self.load_params})
        return df

    def _remove_empty_streams(self, data):
        streams = {}
        empty = {}
        for stream_id, d in data.items():
            if (
                d is None
                or (isinstance(d, list) and len(d) == 0)
                or (isinstance(d, dict) and all([len(x) == 0 for x in d.values()]))
                or (isinstance(d, np.ndarray) and d.size == 0)
            ):
                empty[stream_id] = d
            else:
                streams[stream_id] = d
        return streams, empty

    def _check_empty_streams(self, data, name):
        data, empty = self._remove_empty_streams(data)
        if empty and self.verbose:
            print(
                f"""No {name} for streams: {
                    " ".join(str(i) for i in sorted(list(empty.keys())))
                }"""
            )
        if not data:
            logger.warning(f"No {name} found!")
            return None
        return data

    def _check_columns(self, df, columns, ignore_missing):
        columns = self._remove_duplicates(columns)
        valid_cols = [col for col in columns if col in df.columns]
        if not ignore_missing and len(valid_cols) != len(columns):
            invalid_cols = set(columns).difference(df.columns)
            raise KeyError(f"Invalid columns: {invalid_cols}")
        return valid_cols
