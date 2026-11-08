"""Version matrix for the eubi stack: install combinations of released and
candidate versions into throwaway venvs and smoke-test them before a release.

    python tools/release_matrix/matrix.py plan                # the cells and what each should do
    python tools/release_matrix/matrix.py build               # wheels of the local candidates
    python tools/release_matrix/matrix.py run                 # every cell, report in the workdir
    python tools/release_matrix/matrix.py run --only lite     # cells whose id contains "lite"
    python tools/release_matrix/matrix.py resolve             # what a fresh `pip install` picks
    python tools/release_matrix/matrix.py run --pypi          # after upload: PyPI only, no wheelhouse

Options: --config PATH (default: matrix.toml next to this file), --keep (keep
the venvs), --workdir PATH (overrides the config).

A cell is one consumer version (eubi-bridge or eubi-bridge-lite) plus one
value per axis (for example micro-reader 0.0.4 or the candidate).  Packages
not named in a cell are left to pip.  "candidate" means the wheel `build` put
into the wheelhouse.  Each cell gets a fresh venv, installs, runs the smoke
commands on the samples, and is judged against the first [[expect]] rule whose
specifiers all match the installed versions (default: "pass").

Outcomes:
  pass           every smoke command exits 0, writes OME-Zarr, logs no fallback
  guard          a smoke command fails with output matching the rule's `match`
  fallback       exit 0, but the output names a silent switch to older readers
  unresolvable   pip cannot install the cell
  error          anything else

The report (Markdown + JSON) goes into the workdir; the exit code is 1 when a
cell's outcome differs from its expectation.  Needs Python 3.11+ (tomllib) and
`packaging` (or pip's vendored copy).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

try:
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:                                   # pragma: no cover - pip always has it
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version

HERE = Path(__file__).resolve().parent
CANDIDATE = "candidate"
ORDER = ["unresolvable", "error", "fallback", "guard", "pass"]   # worst first


# -- config -------------------------------------------------------------------

def load_config(path: Path, workdir: str | None) -> dict:
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    wd = workdir or cfg.get("paths", {}).get("workdir") or str(
        Path(tempfile.gettempdir()) / "eubi-release-matrix")
    cfg["_workdir"] = Path(wd)
    cfg["_wheelhouse"] = cfg["_workdir"] / "wheelhouse"
    cfg["_stack"] = [canonicalize_name(n) for n in cfg.get("stack", [])]
    return cfg


def python_for(cfg: dict, key: str) -> str:
    exe = cfg.get("pythons", {}).get(key)
    if not exe:
        raise SystemExit(f"[pythons] has no entry {key!r}")
    return exe


# -- wheelhouse ---------------------------------------------------------------

def wheel_versions(wheelhouse: Path) -> dict[str, str]:
    """Distribution name -> version of the newest wheel in the wheelhouse."""
    found: dict[str, Version] = {}
    for whl in wheelhouse.glob("*.whl"):
        name, version = whl.name.split("-")[:2]
        name = canonicalize_name(name)
        if name not in found or Version(version) > found[name]:
            found[name] = Version(version)
    return {n: str(v) for n, v in found.items()}


def cmd_build(cfg: dict, only: str | None) -> int:
    wh = cfg["_wheelhouse"]
    wh.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, spec in cfg.get("candidates", {}).items():
        if only and only not in name:
            continue
        src = Path(spec["path"])
        py = python_for(cfg, spec.get("python", "build"))
        if "command" in spec:                         # e.g. eubi's tools/build_lite.py
            cmd = [a.format(python=py, path=src, wheelhouse=wh) for a in spec["command"]]
            print(f"[build] {name}: {' '.join(cmd)}")
            ok = run(cmd, cwd=src).returncode == 0
            out = src / spec.get("dist", "dist")
            pattern = canonicalize_name(name).replace("-", "_") + "-*.whl"
            wheels = sorted(out.glob(pattern), key=lambda p: p.stat().st_mtime)
            if ok and wheels:
                shutil.copy2(wheels[-1], wh / wheels[-1].name)
            ok = ok and bool(wheels)
        else:
            cmd = [py, "-m", "pip", "wheel", "--no-deps", "-w", str(wh), str(src)]
            print(f"[build] {name}: {' '.join(cmd)}")
            ok = run(cmd).returncode == 0
        print(f"[build] {name}: {'ok' if ok else 'FAILED'}")
        failed += not ok
    print(f"[build] wheelhouse {wh}: {wheel_versions(wh)}")
    return 1 if failed else 0


# -- cells --------------------------------------------------------------------

def cells(cfg: dict, use_wheelhouse: bool) -> list[dict]:
    """Every combination: one consumer version x one value per axis."""
    built = wheel_versions(cfg["_wheelhouse"]) if use_wheelhouse else {}
    axes = cfg.get("axes", {})
    out = []
    for consumer in cfg["consumers"]:
        for cv in consumer["versions"]:
            for combo in itertools.product(*axes.values()):
                pins = {consumer["name"]: cv, **dict(zip(axes, combo))}
                resolved, missing = {}, []
                for name, v in pins.items():
                    if v == CANDIDATE:
                        v = built.get(canonicalize_name(name))
                        if v is None:
                            missing.append(name)
                    resolved[name] = v
                label = " + ".join(f"{n}=={v}" for n, v in pins.items())
                out.append({"id": label, "consumer": consumer, "pins": resolved,
                            "missing": missing})
    return out


def expectation(cfg: dict, versions: dict[str, str]) -> dict:
    """The first [[expect]] rule whose specifiers all hold, else pass."""
    norm = {canonicalize_name(k): v for k, v in versions.items() if v}
    for rule in cfg.get("expect", []):
        when = {canonicalize_name(k): s for k, s in rule["when"].items()}
        if all(k in norm and Version(norm[k]) in SpecifierSet(s) for k, s in when.items()):
            return rule
    return {"outcome": "pass"}


def cmd_plan(cfg: dict, only: str | None, use_wheelhouse: bool) -> int:
    for c in cells(cfg, use_wheelhouse):
        if only and only not in c["id"]:
            continue
        exp = expectation(cfg, c["pins"])
        note = f"  (not built: {', '.join(c['missing'])})" if c["missing"] else ""
        print(f"{exp['outcome']:<13} {c['id']}  ->  {pins_text(c['pins'])}{note}")
    return 0


def pins_text(pins: dict) -> str:
    return ", ".join(f"{n}=={v}" for n, v in pins.items() if v)


# -- running ------------------------------------------------------------------

def run(cmd, cwd=None, timeout=None, env=None) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}
    try:
        return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=full_env)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "",
                                           (exc.stderr or "") + "\n[matrix] timed out")


def make_venv(base_python: str, where: Path) -> Path:
    if where.exists():
        rmtree(where)
    r = run([base_python, "-m", "venv", str(where)])
    if r.returncode != 0:
        raise SystemExit(f"cannot create a venv with {base_python}:\n{r.stderr}")
    return where / ("Scripts" if os.name == "nt" else "bin")


def rmtree(path: Path) -> None:
    def onerror(func, p, _):                          # Windows: read-only files
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onexc=onerror) if sys.version_info >= (3, 12) else \
        shutil.rmtree(path, onerror=onerror)


def installed(bin_dir: Path, names: list[str]) -> dict[str, str]:
    code = ("import json, importlib.metadata as m\n"
            f"names = {names!r}\n"
            "out = {}\n"
            "for n in names:\n"
            "    try: out[n] = m.version(n)\n"
            "    except m.PackageNotFoundError: pass\n"
            "print(json.dumps(out))")
    # -I: isolated, so a checkout's *.egg-info in the current folder is not seen
    r = run([bin_dir / "python", "-I", "-c", code])
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}


def has_zarr(out: Path) -> bool:
    return any(out.rglob("zarr.json")) or any(out.rglob(".zattrs"))


def judge(cfg: dict, exp: dict, rc: int, text: str, out: Path) -> str:
    fallback = [re.compile(p) for p in cfg.get("fallback_patterns", [])]
    if rc != 0:
        if exp.get("outcome") == "guard" and re.search(exp.get("match", "$^"), text):
            return "guard"
        return "error"
    if any(p.search(text) for p in fallback):
        return "fallback"
    return "pass" if has_zarr(out) else "error"


def cmd_run(cfg: dict, only: str | None, keep: bool, use_wheelhouse: bool) -> int:
    wd = cfg["_workdir"]
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    samples = cfg.get("samples", {})
    timeout = cfg.get("smoke_timeout_s", 900)
    for i, c in enumerate(cells(cfg, use_wheelhouse)):
        if only and only not in c["id"]:
            continue
        cell_dir = wd / "cells" / f"{stamp}-{i:03d}"
        exp = expectation(cfg, c["pins"])
        rec = {"id": c["id"], "pins": c["pins"], "expected": exp["outcome"], "samples": {}}
        print(f"[run] {c['id']}  (expected: {exp['outcome']})", flush=True)
        if c["missing"]:
            rec.update(outcome="error", detail=f"not built: {', '.join(c['missing'])}")
            results.append(rec)
            continue
        bin_dir = make_venv(python_for(cfg, c["consumer"]["python"]), cell_dir / "venv")
        # A fresh, empty home per cell: user-level settings (eubi's ~/.eubi_bridge
        # config, for one) must not leak into what a new user would get.
        home = cell_dir / "home"
        home.mkdir(parents=True, exist_ok=True)
        home_env = {"HOME": str(home), "USERPROFILE": str(home)}
        pip = [bin_dir / "python", "-m", "pip", "install", "--disable-pip-version-check"]
        if use_wheelhouse:
            pip += ["--find-links", str(cfg["_wheelhouse"])]
        r = run(pip + [f"{n}=={v}" for n, v in c["pins"].items()], cwd=cell_dir)
        rec["installed"] = installed(bin_dir, cfg["_stack"])
        if r.returncode != 0:
            rec.update(outcome="unresolvable", detail=tail(r.stdout + r.stderr))
        else:
            exp = expectation(cfg, {**c["pins"], **rec["installed"]})
            rec["expected"] = exp["outcome"]
            outcomes = []
            for key in c["consumer"]["samples"]:
                out = cell_dir / f"out-{key}"
                cmd = [a.format(bin=bin_dir, sample=samples[key], out=out)
                       for a in c["consumer"]["smoke"]]
                s = run(cmd, cwd=cell_dir, timeout=timeout, env=home_env)
                text = s.stdout + s.stderr
                o = judge(cfg, exp, s.returncode, text, out)
                outcomes.append(o)
                rec["samples"][key] = {"outcome": o, "rc": s.returncode, "tail": tail(text)}
            rec["outcome"] = min(outcomes, key=ORDER.index) if outcomes else "pass"
        rec["ok"] = rec["outcome"] == rec["expected"]
        print(f"[run]   -> {rec['outcome']}{'' if rec['ok'] else '   << differs'}", flush=True)
        results.append(rec)
        if not keep:
            rmtree(cell_dir / "venv")
    for rec in results:
        rec.setdefault("ok", False)
    report(wd, stamp, results, "pypi" if not use_wheelhouse else "wheelhouse")
    return 0 if all(r["ok"] for r in results) else 1


def tail(text: str, n: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def report(wd: Path, stamp: str, results: list[dict], source: str) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / f"report-{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [f"# Version matrix {stamp} ({source})", "",
             "| Cell | Installed | Expected | Outcome | OK |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        inst = ", ".join(f"{k} {v}" for k, v in r.get("installed", {}).items())
        lines.append(f"| {r['id']} | {inst} | {r['expected']} | {r['outcome']} | "
                     f"{'yes' if r['ok'] else '**no**'} |")
    bad = [r for r in results if not r["ok"]]
    for r in bad:
        lines += ["", f"## {r['id']}", "", "```", r.get("detail", "") or "\n\n".join(
            f"[{k}] rc={s['rc']}\n{s['tail']}" for k, s in r["samples"].items()), "```"]
    path = wd / f"report-{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[run] {len(results) - len(bad)}/{len(results)} cells as expected; report {path}")


# -- resolve ------------------------------------------------------------------

def cmd_resolve(cfg: dict, use_wheelhouse: bool) -> int:
    """What a fresh install resolves, per scenario, without installing."""
    wd = cfg["_workdir"]
    for sc in cfg.get("resolve", []):
        bin_dir = make_venv(python_for(cfg, sc["python"]), wd / "resolve-venv")
        rep = wd / "resolve-report.json"
        cmd = [bin_dir / "python", "-m", "pip", "install", "--disable-pip-version-check",
               "--dry-run", "--ignore-installed", "--report", str(rep)]
        if use_wheelhouse:
            cmd += ["--find-links", str(cfg["_wheelhouse"])]
        r = run(cmd + sc["requirements"])
        if r.returncode != 0:
            print(f"[resolve] {sc['name']}: UNRESOLVABLE\n{tail(r.stdout + r.stderr, 8)}")
            continue
        picked = {canonicalize_name(i["metadata"]["name"]): i["metadata"]["version"]
                  for i in json.loads(rep.read_text(encoding="utf-8"))["install"]}
        stack = {n: picked[n] for n in cfg["_stack"] if n in picked}
        print(f"[resolve] {sc['name']}: {pins_text(stack)}")
    rmtree(wd / "resolve-venv") if (wd / "resolve-venv").exists() else None
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["plan", "build", "run", "resolve"])
    ap.add_argument("--config", default=str(HERE / "matrix.toml"))
    ap.add_argument("--workdir")
    ap.add_argument("--only", help="only cells (or candidates, for build) containing this text")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway venvs")
    ap.add_argument("--pypi", action="store_true", help="ignore the wheelhouse: PyPI only")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config), a.workdir)
    use_wh = not a.pypi
    if a.command == "plan":
        return cmd_plan(cfg, a.only, use_wh)
    if a.command == "build":
        return cmd_build(cfg, a.only)
    if a.command == "run":
        return cmd_run(cfg, a.only, a.keep, use_wh)
    return cmd_resolve(cfg, use_wh)


if __name__ == "__main__":
    sys.exit(main())
