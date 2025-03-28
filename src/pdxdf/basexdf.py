"""Thin wrapper for inspecting raw XDF data."""

from warnings import warn

import pyxdf


class BaseXdf:
    """Thin wrapper for inspecting raw XDF data.

    Provides convenience methods for accessing raw XDF file stream information
    without loading stream data.

    Properties:
        filename: XDF file - string or Path.
        available_stream_ids: a list of available stream IDs.

    Attributes:
        verbose: Boolean determining additional logging.
    """

    def __init__(self, filename, verbose=False):
        """Initialise XDF file."""
        self._filename = filename
        self.verbose = verbose

    # Properties

    @property
    def filename(self):
        """XDF file - string or Path."""
        return self._filename

    @property
    def available_stream_ids(self):
        """Return a list of available stream IDs."""
        return self.__available_stream_ids()

    # Public methods.

    def resolve_streams(self):
        """Resolve streams in the current file."""
        return self.__resolve_streams()

    def match_streaminfos(self, *parameters):
        """Match streams given property values.

        See pyxdf.match_streaminfos for matching options.
        """
        return pyxdf.match_streaminfos(BaseXdf.resolve_streams(self), parameters)

    def remove_duplicates(self, values):
        """Remove duplicate values from a list preserving order."""
        unique = set(values)
        if len(unique) == len(values):
            unique = values
        else:
            unique = [v for v in values if values.count(v) == 1]
            duplicates = set([v for v in values if values.count(v) > 1])
            if self.verbose:
                warn(f"Duplicate values: {duplicates}.")
        return unique

    # Abstract methods.

    def load(self):
        """Load an XDF file."""
        raise NotImplementedError()

    def metadata(self):
        """Return loaded stream metadata."""
        raise NotImplementedError()

    def desc(self):
        """Return loaded stream extensible desc metadata."""
        raise NotImplementedError()

    def segments(self):
        """Return loaded stream segments."""
        raise NotImplementedError()

    def clock_segments(self):
        """Return loaded stream clock segments."""
        raise NotImplementedError()

    def channel_metadata(self):
        """Return loaded stream channel metadata."""
        raise NotImplementedError()

    def footer(self):
        """Return loaded stream footer metadata."""
        raise NotImplementedError()

    def clock_offsets(self):
        """Return loaded stream clock offsets."""
        raise NotImplementedError()

    def time_series(self):
        """Return loaded stream time-series data."""
        raise NotImplementedError()

    def time_stamps(self):
        """Return loaded stream data time-stamps."""
        raise NotImplementedError()

    def data(self):
        """Return combined time-series and time-stamp data."""
        raise NotImplementedError()

    def resample(self):
        """Resample multiple XDF streams to a given frequency."""
        raise NotImplementedError()

    # Name-mangled private methods to be used only by this class.

    def __resolve_streams(self):
        """Resolve streams in the current file."""
        return pyxdf.resolve_streams(str(self.filename))

    def __available_stream_ids(self):
        streams = self.__resolve_streams()
        stream_ids = sorted([stream["stream_id"] for stream in streams])
        return stream_ids
