"""The Docker image label carries the package version: the Dockerfile's EUBI_VERSION must follow
pyproject.toml, so a release cannot ship an image that still says an old version."""
import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_version_follows_pyproject():
    dockerfile = _ROOT / "Dockerfile"
    if not dockerfile.exists():
        pytest.skip("no Dockerfile next to the tests (installed package)")
    version = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    found = re.findall(r"^ARG EUBI_VERSION=(\S+)$", dockerfile.read_text(encoding="utf-8"), re.MULTILINE)
    assert found == [version], f"Dockerfile EUBI_VERSION {found} != pyproject version {version}"
    # The example commands in the header use the same tag.
    tags = set(re.findall(r"eubi-bridge:(\d[\w.]*)", dockerfile.read_text(encoding="utf-8")))
    assert tags <= {version}, f"Dockerfile examples use tags {sorted(tags)}, expected {version}"
