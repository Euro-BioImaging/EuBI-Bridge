"""Stand-ins for the GUI's conversion subprocess (spawned by the tests, so they live
in an importable module)."""


def fail_with_a_long_traceback(call_args, log_queue, result_queue):
    """A conversion that fails with a long error, as a batch whose every row fails
    does: far more than a pipe holds before it is read."""
    log_queue.put_nowait("converting...")
    result_queue.put(("err", "Traceback (most recent call last):\n" + "x" * 1_000_000))


def succeed(call_args, log_queue, result_queue):
    log_queue.put_nowait("converting...")
    result_queue.put(("ok", None))
