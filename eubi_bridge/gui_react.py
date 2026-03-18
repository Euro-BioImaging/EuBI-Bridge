"""
Entry point for the EuBI-Bridge React GUI server.

The front-end and Express server are pre-built (``npm run build``) and shipped
as ``eubi_bridge/eubi_gui/dist/``.  At runtime only Node.js >= 20 is required —
no npm, no node_modules installation, no writing to the user's home directory.
This makes ``eubi-gui-react`` safe for shared / multi-user HPC environments.

Usage (after ``pip install``)::

    eubi-gui-react
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


# Ports used by the React GUI and its companion zarr-plane server.
_PORTS = (5000, 5555)


def _kill_port(port: int) -> None:
    """Terminate any process listening on *port* (best-effort, cross-platform)."""
    try:
        import psutil
    except ImportError:
        # psutil is listed as a dependency, but guard defensively.
        _kill_port_fallback(port)
        return

    killed = False
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.pid:
            try:
                proc = psutil.Process(conn.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                print(f"[eubi-gui-react] Killed PID {conn.pid} on port {port}")
                killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    if not killed:
        print(f"[eubi-gui-react] No process found on port {port}")


def _kill_port_fallback(port: int) -> None:
    """Fallback using OS commands when psutil is unavailable."""
    import os

    if sys.platform == "win32":
        # netstat + taskkill
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                print(f"[eubi-gui-react] Killed PID {pid} on port {port} (fallback)")
    else:
        # lsof
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
        )
        for pid in result.stdout.strip().splitlines():
            os.kill(int(pid), 15)  # SIGTERM
            print(f"[eubi-gui-react] Killed PID {pid} on port {port} (fallback)")


def _get_working_gui_dir() -> Path:
    # kept for backwards compatibility — no longer used in main()
    return Path(__file__).resolve().parent / "eubi_gui"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Kill existing server processes
    # ------------------------------------------------------------------
    print("[eubi-gui-react] Stopping existing servers …")
    for port in _PORTS:
        _kill_port(port)

    # ------------------------------------------------------------------
    # 2. Locate the pre-built GUI bundle
    # ------------------------------------------------------------------
    gui_dir = Path(__file__).resolve().parent / "eubi_gui"
    dist_index = gui_dir / "dist" / "index.cjs"

    if not dist_index.exists():
        print(
            f"[eubi-gui-react] ERROR: pre-built GUI not found at {dist_index}.\n"
            "This is a packaging error — the dist/ directory was not included when\n"
            "the package was built.  Please file a bug report.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2b. Check Node.js version (>= 20 required)
    # ------------------------------------------------------------------
    try:
        node_version_output = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        ).stdout.strip()  # e.g. "v20.11.0"
        major = int(node_version_output.lstrip("v").split(".")[0])
        if major < 20:
            print(
                f"[eubi-gui-react] ERROR: Node.js {node_version_output} is too old. "
                "Node.js >= 20.19.0 is required.\n"
                "Install it from https://nodejs.org or via your system package manager "
                "(e.g. 'conda install -c conda-forge nodejs>=20' or 'nvm install 20').",
                file=sys.stderr,
            )
            sys.exit(1)
    except FileNotFoundError:
        print(
            "[eubi-gui-react] ERROR: 'node' not found. "
            "Install Node.js >= 20.19.0 from https://nodejs.org",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Launch the pre-built Express server
    # ------------------------------------------------------------------
    # Tell the bundled server where the Python worker scripts live.
    # Without this env var the server would look in dist/ (its own directory).
    env = os.environ.copy()
    env["EUBI_SCRIPTS_DIR"] = str(gui_dir / "server")

    print(f"[eubi-gui-react] Starting server ({dist_index}) …")
    try:
        subprocess.run(["node", str(dist_index)], env=env, check=True)
    except KeyboardInterrupt:
        print("\n[eubi-gui-react] Interrupted — shutting down.")
    except subprocess.CalledProcessError as exc:
        print(f"[eubi-gui-react] node exited with code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
