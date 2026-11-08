"""Errors in what the user asked for, shown by the command line as one line."""


class InputError(ValueError):
    """The input or an option is not usable: no program error, so no traceback."""


class TasksFailedError(RuntimeError):
    """Some conversions of a run failed, each named in an ERROR line as it failed: the
    command line ends with this one line and exit code 1, no traceback (a RuntimeError,
    as before, for scripts that catch it)."""
