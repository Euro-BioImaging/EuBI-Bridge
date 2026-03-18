"""
Entry point for the EuBI-Bridge React GUI server.

Behaviour
---------
1. Kill any process currently listening on the GUI ports (5000, 5555).
2. Change into the *eubi_gui* directory (sibling of the project root).
3. Launch ``npm run dev`` — the Express + Vite development server.

Usage (after ``pip install -e .``)::

    eubi-gui-react
"""

from __future__ import annotations

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


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Kill existing server processes
    # ------------------------------------------------------------------
    print("[eubi-gui-react] Stopping existing servers …")
    for port in _PORTS:
        _kill_port(port)

    # ------------------------------------------------------------------
    # 2. Locate the eubi_gui directory
    # ------------------------------------------------------------------
    # This file lives at  <package>/eubi_bridge/gui_react.py
    # eubi_gui is at      <package>/eubi_bridge/eubi_gui/
    gui_dir = Path(__file__).resolve().parent / "eubi_gui"

    if not gui_dir.exists():
        print(
            f"[eubi-gui-react] ERROR: eubi_gui directory not found at {gui_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2b. Ensure npm dependencies are installed
    # ------------------------------------------------------------------
    if not (gui_dir / "node_modules").exists():
        print("[eubi-gui-react] node_modules not found — running npm install …")
        use_shell = sys.platform == "win32"
        result = subprocess.run(
            ["npm", "install"], cwd=str(gui_dir), shell=use_shell
        )
        if result.returncode != 0:
            print("[eubi-gui-react] ERROR: npm install failed", file=sys.stderr)
            sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # 3. Launch npm run dev
    # ------------------------------------------------------------------
    print(f"[eubi-gui-react] Starting server from {gui_dir} …")

    # On Windows, npm is a .cmd script and needs shell=True (or explicit .cmd).
    use_shell = sys.platform == "win32"
    cmd = ["npm", "run", "dev"]

    try:
        subprocess.run(cmd, cwd=str(gui_dir), check=True, shell=use_shell)
    except KeyboardInterrupt:
        print("\n[eubi-gui-react] Interrupted — shutting down.")
    except subprocess.CalledProcessError as exc:
        print(f"[eubi-gui-react] npm exited with code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
