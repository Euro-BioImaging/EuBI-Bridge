"""memory_budget='auto' inside a cluster job (Workplan E25, BARD report 2026-11-04).

On a SLURM node, psutil's free RAM is the whole node's; the job's limit is a cgroup, and
a job that passes it is killed whole (the BARD remote desktop with it). available_mb()
takes the least of the free RAM, what the process's cgroup limit leaves and the SLURM
job's memory. The cgroup files here are fakes under tmp_path.
"""
from __future__ import annotations

import logging
import os

import pytest

from eubi_bridge.core import memory

GB = 1 << 30


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for name in ("SLURM_JOB_ID", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
                 "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    # no cgroup unless a test writes one
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", str(tmp_path / "none"))
    monkeypatch.setattr(memory, "PROC_SELF_MOUNTINFO", str(tmp_path / "none"))
    memory._said_budget.clear()
    memory._warned.clear()


@pytest.fixture
def node_ram(monkeypatch):
    """psutil reports 500 GB free: a big node."""
    class _VM:
        available = 500 * GB

    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _cgroup_v2(monkeypatch, tmp_path, leaf, limits, root="/"):
    """A cgroup2 mount at tmp_path/cg; *limits*: {relative dir: (max, current,
    inactive_file)}; the process sits in *leaf*."""
    mount = tmp_path / "cg"
    for rel, (limit, current, inactive) in limits.items():
        d = mount / rel if rel else mount
        _write(d / "memory.max", "max" if limit is None else str(limit))
        _write(d / "memory.current", str(current))
        _write(d / "memory.stat", f"anon 1\ninactive_file {inactive}\nactive_file 5\n")
    _write(tmp_path / "proc_cgroup", f"0::{leaf}\n")
    _write(tmp_path / "mountinfo",
           f"30 25 0:26 {root} {mount.as_posix()} rw,nosuid - cgroup2 cgroup2 rw\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", str(tmp_path / "proc_cgroup"))
    monkeypatch.setattr(memory, "PROC_SELF_MOUNTINFO", str(tmp_path / "mountinfo"))


def test_off_a_cluster_auto_is_the_free_ram(node_ram):
    assert memory.available_mb() == pytest.approx(500 * 1024)
    assert memory.cgroup_free_mb() is None


def test_a_cgroup_v2_limit_wins_over_the_nodes_ram(monkeypatch, tmp_path, node_ram):
    # the job may use 32 GB and holds 10 GB, 2 GB of it droppable file pages
    _cgroup_v2(monkeypatch, tmp_path, "/system.slice/slurmstepd.scope/job_7/step_0", {
        "": (None, 0, 0),
        "system.slice/slurmstepd.scope/job_7": (32 * GB, 10 * GB, 2 * GB),
        "system.slice/slurmstepd.scope/job_7/step_0": (None, 9 * GB, 1 * GB),
    })
    free, limit = memory.cgroup_free_mb()
    assert (free, limit) == (pytest.approx(24 * 1024), pytest.approx(32 * 1024))
    assert memory.available_mb() == pytest.approx(24 * 1024)


def test_the_tightest_of_nested_limits_counts(monkeypatch, tmp_path, node_ram):
    _cgroup_v2(monkeypatch, tmp_path, "/a/b", {
        "a": (16 * GB, 4 * GB, 0),          # 12 GB left
        "a/b": (64 * GB, 4 * GB, 0),        # 60 GB left
    })
    assert memory.cgroup_free_mb()[0] == pytest.approx(12 * 1024)


def test_a_container_mount_shows_its_own_subtree(monkeypatch, tmp_path, node_ram):
    # inside a container, /proc/self/cgroup names the host path the mount's root shows
    _cgroup_v2(monkeypatch, tmp_path, "/kubepods/pod1/c1", {"": (8 * GB, 2 * GB, 0)},
               root="/kubepods/pod1/c1")
    assert memory.cgroup_free_mb()[0] == pytest.approx(6 * 1024)


def test_cgroup_v1(monkeypatch, tmp_path, node_ram):
    mount = tmp_path / "memory"
    d = mount / "slurm" / "uid_1" / "job_9"
    _write(d / "memory.limit_in_bytes", str(20 * GB))
    _write(d / "memory.usage_in_bytes", str(6 * GB))
    _write(d / "memory.stat", f"cache 1\ntotal_inactive_file {GB}\n")
    for parent in (mount, mount / "slurm", mount / "slurm" / "uid_1"):
        _write(parent / "memory.limit_in_bytes", str((1 << 63) - 4096))   # no limit
        _write(parent / "memory.usage_in_bytes", str(90 * GB))
    _write(tmp_path / "proc_cgroup", "5:cpuset:/slurm\n4:memory:/slurm/uid_1/job_9\n")
    _write(tmp_path / "mountinfo",
           f"40 25 0:30 / {mount.as_posix()} rw - cgroup cgroup rw,memory\n"
           f"41 25 0:31 / {(tmp_path / 'cpuset').as_posix()} rw - cgroup cgroup rw,cpuset\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", str(tmp_path / "proc_cgroup"))
    monkeypatch.setattr(memory, "PROC_SELF_MOUNTINFO", str(tmp_path / "mountinfo"))
    assert memory.cgroup_free_mb() == (pytest.approx(15 * 1024), pytest.approx(20 * 1024))


@pytest.mark.parametrize("env, mb", [
    ({"SLURM_MEM_PER_NODE": "16000"}, 16000),
    ({"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_ON_NODE": "4"}, 8000),
    ({"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_PER_TASK": "3"}, 6000),
    ({}, None),
])
def test_the_slurm_jobs_memory(monkeypatch, env, mb):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert memory.slurm_job_mb() == mb


def test_slurm_variables_outside_a_job_are_ignored(monkeypatch, node_ram):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1000")        # no SLURM_JOB_ID
    assert memory.slurm_job_mb() is None
    assert memory.available_mb() == pytest.approx(500 * 1024)


def test_one_log_line_names_the_budget_and_its_source(monkeypatch, tmp_path, node_ram,
                                                      caplog):
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(64 * 1024))
    _cgroup_v2(monkeypatch, tmp_path, "/job_7", {"job_7": (32 * GB, 16 * GB, 0)})
    with caplog.at_level(logging.INFO, logger=memory.logger.name):
        assert memory.total_mb("auto") == pytest.approx(8 * 1024)
        memory.total_mb("auto")                             # said once
    lines = [r.getMessage() for r in caplog.records if "memory_budget='auto'" in r.getMessage()]
    assert lines == ["memory_budget='auto': 8.0 GB, half of the 16.0 GB the SLURM job may "
                     "still use (its limit: 32.0 GB)."]


def test_an_explicit_budget_over_the_jobs_limit_warns(monkeypatch, tmp_path, node_ram,
                                                      caplog):
    # the BARD case: 120 GB asked for, the job has 16 GB left
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    _cgroup_v2(monkeypatch, tmp_path, "/job_7", {"job_7": (128 * GB, 112 * GB, 0)})
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        assert memory.total_mb("120GB") == pytest.approx(120 * 1024)
    assert any("more than the 16.0 GB the SLURM job may still use" in r.getMessage()
               for r in caplog.records)


def test_unreadable_cgroup_files_fall_back_to_the_ram(monkeypatch, tmp_path, node_ram):
    _write(tmp_path / "proc_cgroup", "garbage\n")
    _write(tmp_path / "mountinfo", "also garbage\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", str(tmp_path / "proc_cgroup"))
    monkeypatch.setattr(memory, "PROC_SELF_MOUNTINFO", str(tmp_path / "mountinfo"))
    assert memory.available_mb() == pytest.approx(500 * 1024)


@pytest.mark.skipif(not os.path.exists("/proc/self/cgroup"), reason="Linux only")
def test_this_machines_cgroup_reads_without_error(monkeypatch):
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", "/proc/self/cgroup")
    monkeypatch.setattr(memory, "PROC_SELF_MOUNTINFO", "/proc/self/mountinfo")
    found = memory.cgroup_free_mb()
    assert found is None or (found[0] >= 0 and found[1] > 0)
