"""Thin wrapper for inspecting raw XDF data."""

from abc import ABC, abstractmethod
from warnings import warn

import pyxdf


class BaseXdf(ABC):
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

    # Abstract methods.

    @abstractmethod
    def load(self):
        """Load an XDF file."""
        pass

    @abstractmethod
    def unload(self):
        """Free memory."""
        pass

    @abstractmethod
    def header(self):
        """Return the raw header info dictionary."""
        pass

    @abstractmethod
    def info(self):
        """Return loaded stream info."""
        pass

    @abstractmethod
    def is_marker_stream(self, stream_id):
        """Test if stream is a marker stream."""
        pass

    @abstractmethod
    def desc(self):
        """Return loaded stream extensible desc info."""
        pass

    @abstractmethod
    def segments(self):
        """Return loaded stream segments."""
        pass

    @abstractmethod
    def clock_segments(self):
        """Return loaded stream clock segments."""
        pass

    @abstractmethod
    def segment_info(self):
        """Return stream segment info."""
        pass

    @abstractmethod
    def channel_info(self):
        """Return loaded stream channel info."""
        pass

    @abstractmethod
    def footer(self):
        """Return loaded stream footer info."""
        pass

    @abstractmethod
    def clock_offsets(self):
        """Return loaded stream clock offsets."""
        pass

    @abstractmethod
    def time_series(self):
        """Return loaded stream time-series data."""
        pass

    @abstractmethod
    def time_stamps(self):
        """Return loaded stream data time-stamps."""
        pass

    @abstractmethod
    def data(self):
        """Return combined time-series and time-stamp data."""
        pass

    # Private methods.

    def _remove_duplicates(self, values):
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

    def _single_or_multi_stream_data(self, data, with_stream_id=False):
        """Return single stream data or dictionary."""
        if len(data) == 1 and not with_stream_id:
            return data[list(data.keys())[0]]
        else:
            return data

    # Name-mangled private methods to be used only by this class.

    def __resolve_streams(self):
        """Resolve streams in the current file."""
        return pyxdf.resolve_streams(str(self.filename))

    def __available_stream_ids(self):
        streams = self.__resolve_streams()
        stream_ids = sorted([stream["stream_id"] for stream in streams])
        return stream_ids
