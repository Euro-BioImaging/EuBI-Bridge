"""tensorstore_data_copy_concurrency: retired in 0.2.0, accepted, does nothing.

The worker used to build a TensorStore context from it that nothing read, and logged
"Tensorstore context configured: data_copy_concurrency=..." as if it applied (a user's
cluster benchmark, 2026-11-27). The context and the log line are gone; the setting
still loads from saved configs and warns once when set to a value of its own.
"""
import inspect
import logging

from eubi_bridge.conversion import conversion_worker, dispatcher, worker_init
from eubi_bridge.core.config_models import ClusterConfig


def test_the_worker_builds_no_tensorstore_context():
    assert not hasattr(worker_init, "build_tensorstore_context")
    assert not hasattr(worker_init, "_tensorstore_context")
    assert not inspect.signature(worker_init.initialize_worker_process).parameters
    assert not inspect.signature(dispatcher._slurm_worker_init).parameters
    source = inspect.getsource(worker_init)
    assert "Tensorstore context configured" not in source
    assert "EUBI_TENSORSTORE_DATA_COPY_CONCURRENCY" not in inspect.getsource(dispatcher)


def test_the_pools_pass_the_initializer_nothing():
    _cls, kwargs = dispatcher._build_executor(ClusterConfig(), use_threading=False)
    assert kwargs["initializer"] is worker_init.initialize_worker_process
    assert "initargs" not in kwargs


def test_a_saved_config_with_the_setting_still_loads_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(conversion_worker, "_warned_unused", set())
    default = ClusterConfig()
    caplog.set_level(logging.WARNING)
    conversion_worker._warn_unused_writer_settings(default)
    assert "tensorstore_data_copy_concurrency" not in caplog.text

    custom = ClusterConfig(tensorstore_data_copy_concurrency=8)
    conversion_worker._warn_unused_writer_settings(custom)
    conversion_worker._warn_unused_writer_settings(custom)
    assert caplog.text.count("tensorstore_data_copy_concurrency=8 does nothing") == 1
