"""Decorator functions for common error handling."""

from functools import wraps

from .errors import XdfStreamParseError


class XdfDecorators:
    """Class providing shared decorator functions."""

    @staticmethod
    def loaded(f):
        """Decorate loading methods for error handling."""

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            self._assert_loaded()
            return f(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def parse(f):
        """Decorate parsing methods for error handling."""

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except Exception as exc:
                raise XdfStreamParseError(exc)

        return wrapper
