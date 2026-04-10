"""Module for multiprocessing logging setup - must be importable for spawn context"""


def setup_mp_logging(log_queue):
    """Setup logging in child processes to use the shared queue"""
    import sys

    class QueueWriter:
        """Redirect stdout/stderr to the queue for RichHandler output"""

        def __init__(self, queue, original, is_stderr=False):
            self.queue = queue
            self.original = original
            self.is_stderr = is_stderr

        def write(self, text):
            if text and text.strip():
                self.queue.put(text.rstrip())
            if self.original:
                self.original.write(text)

        def flush(self):
            if self.original:
                self.original.flush()

        def isatty(self):
            return True

    sys.stdout = QueueWriter(log_queue, sys.__stdout__)
    sys.stderr = QueueWriter(log_queue, sys.__stderr__, is_stderr=True)

    from eubi_bridge.utils.logging_config import setup_logging
    setup_logging()


def setup_mp_logging_with_worker_init(log_queue, tensorstore_data_copy_concurrency='default'):
    """Combined initializer for subprocess workers: tensorstore+JVM init, then queue logging.

    Must be a module-level function so it is picklable by the spawn process context.
    """
    from eubi_bridge.conversion.worker_init import initialize_worker_process
    initialize_worker_process(tensorstore_data_copy_concurrency)
    setup_mp_logging(log_queue)
