#!/usr/bin/env python3
"""
Optional helper to launch CoppeliaSim headlessly with a given scene, for unattended training
runs once the scene has been built and validated interactively.

Not used by default — the CoppeliaSim variant assumes you already have an instance running
with the scene loaded (matching how the MuJoCo side assumes a pre-built model). This is only
useful once you've confirmed everything works with the GUI open, and want to skip the GUI for
a long unattended run.

Usage:
    python scripts/coppeliasim/launch_headless.py --coppeliasim-exe "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe" --scene scripts/coppeliasim/assets/fetch_push_scene.ttt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coppeliasim-exe", required=True,
        help="Full path to the CoppeliaSim executable (coppeliaSim.exe on Windows)",
    )
    parser.add_argument(
        "--scene", required=True,
        help="Path to the .ttt scene file to load",
    )
    parser.add_argument(
        "--startup-wait", type=float, default=5.0,
        help="Seconds to wait after launch before returning, to give the ZMQ remote API "
             "server time to come up (default: 5.0)",
    )
    args = parser.parse_args()

    exe_path = Path(args.coppeliasim_exe)
    scene_path = Path(args.scene)
    if not exe_path.exists():
        print(f"Error: CoppeliaSim executable not found at {exe_path}", file=sys.stderr)
        sys.exit(1)
    if not scene_path.exists():
        print(f"Error: scene file not found at {scene_path}", file=sys.stderr)
        sys.exit(1)

    # -h: headless, -s: start simulation immediately, -q: quit when simulation stops
    cmd = [str(exe_path), "-h", "-s", str(scene_path.resolve())]
    print(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    print(f"Waiting {args.startup_wait:.1f}s for the ZMQ remote API server to come up...")
    time.sleep(args.startup_wait)

    print(f"CoppeliaSim launched (PID {proc.pid}). It will keep running after this script "
          "exits — stop it manually (e.g. `taskkill /PID <pid> /F` on Windows) when done.")


if __name__ == "__main__":
    main()
