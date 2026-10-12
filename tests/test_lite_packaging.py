"""eubi-bridge-lite's packaging: [tool.eubi-bridge.lite] in pyproject.toml and
tools/build_lite.py, which builds it from the same eubi_bridge source.

The lite list must stay a subset of the full one (same specifiers: one
package, two dependency sets), must never pull in the stack lite exists to
avoid, and the staged source must be eubi_bridge without bioformats/.
"""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

requirements = pytest.importorskip("packaging.requirements")
from packaging.utils import canonicalize_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: What lite leaves out: Bio-Formats (and the JVM it needs), bioio, dask.
FULL_ONLY = {
    "bfio", "bioformats-jar", "scyjava", "jpype1", "jgo", "install-jdk",
    "dask", "distributed", "dask-jobqueue", "xarray",
    "aicspylibczi", "pylibczirw", "nd2", "readlif",
}


def _config() -> dict:
    if not PYPROJECT.exists():
        pytest.skip("no pyproject.toml next to the tests (an installed copy)")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _requirements(lines) -> dict:
    reqs = [requirements.Requirement(line) for line in lines]
    return {canonicalize_name(r.name): r for r in reqs}


def _build_lite():
    path = ROOT / "tools" / "build_lite.py"
    spec = importlib.util.spec_from_file_location("build_lite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_lite_dependency_is_a_full_one_with_the_same_specifier():
    config = _config()
    full = _requirements(config["project"]["dependencies"])
    lite = _requirements(config["tool"]["eubi-bridge"]["lite"]["dependencies"])
    for name, req in lite.items():
        assert name in full, f"{name}: in lite but not in the full eubi-bridge"
        assert (req.specifier, req.marker) == (full[name].specifier, full[name].marker), \
            f"{name}: lite says {req}, full says {full[name]}"


def test_lite_pulls_in_none_of_the_full_stack():
    lite = _requirements(_config()["tool"]["eubi-bridge"]["lite"]["dependencies"])
    bad = {n for n in lite if n in FULL_ONLY or n.startswith(("bioio", "dask"))}
    assert not bad, f"lite depends on the full stack: {sorted(bad)}"


def test_the_staged_source_is_eubi_bridge_without_bioformats(tmp_path):
    _config()
    build_lite = _build_lite()
    build_lite.stage(tmp_path)
    staged = {p.relative_to(tmp_path / "eubi_bridge").as_posix()
              for p in (tmp_path / "eubi_bridge").rglob("*") if p.is_file()}
    source = {p.relative_to(ROOT / "eubi_bridge").as_posix()
              for p in (ROOT / "eubi_bridge").rglob("*")
              if p.is_file() and "bioformats" not in p.parts and "__pycache__" not in p.parts
              and "node_modules" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    assert not any(name.startswith("bioformats/") for name in staged)
    assert staged == source
    assert (tmp_path / "README.md").exists() and (tmp_path / "LICENSE").exists()


def test_the_lite_pyproject_comes_from_the_main_one(tmp_path):
    config = _config()
    _build_lite().stage(tmp_path)
    lite = tomllib.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))
    project, wanted = lite["project"], config["tool"]["eubi-bridge"]["lite"]
    assert project["name"] == "eubi-bridge-lite"
    assert project["version"] == config["project"]["version"]          # lockstep
    assert project["requires-python"] == wanted["requires-python"]
    assert project["dependencies"] == wanted["dependencies"]
    assert project["scripts"] == config["project"]["scripts"]
    assert project["license"] == config["project"]["license"]
    data = lite["tool"]["setuptools"]["package-data"]["eubi_bridge"]
    assert not any("bioformats" in pattern for pattern in data)
    # the GUI's server folder has no __init__.py: it ships as package data
    assert "qt_gui/server/**" in data
    # plain setuptools: the JDK-downloading backend is not used
    assert lite["build-system"]["build-backend"] == "setuptools.build_meta"


# -- both packages installed ------------------------------------------------------

def _import_seeing(tmp_path, *fake_dists) -> str:
    """Import eubi_bridge in a fresh process that also sees *fake_dists*
    (name, version) as installed; its stderr."""
    import os
    import subprocess
    import sys
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    for name, version in fake_dists:
        info = site / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-W", "always", "-c", "import eubi_bridge"],
                            capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stderr


def _the_other_package() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("eubi_bridge")
        return "eubi-bridge-lite"
    except PackageNotFoundError:
        pass
    try:
        version("eubi-bridge-lite")
        return "eubi-bridge"
    except PackageNotFoundError:
        pytest.skip("eubi-bridge is not installed here (a source checkout)")


def test_both_packages_installed_says_how_to_fix_it(tmp_path):
    """They share their files: removing one breaks the other, so the advice
    is to remove both."""
    stderr = _import_seeing(tmp_path, (_the_other_package(), "0.1.4"))
    assert "eubi-bridge and eubi-bridge-lite are both installed" in stderr
    assert "pip uninstall -y eubi-bridge eubi-bridge-lite" in stderr


def test_one_package_installed_is_quiet(tmp_path):
    _the_other_package()                                     # skips a source checkout
    assert "both installed" not in _import_seeing(tmp_path)
