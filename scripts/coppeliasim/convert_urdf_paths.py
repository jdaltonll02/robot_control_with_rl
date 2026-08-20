#!/usr/bin/env python3
"""
One-off fixup for importing a ROS-style URDF into CoppeliaSim.

CoppeliaSim's URDF importer doesn't understand ROS's `package://<pkg>/...` mesh URIs — it
needs a real filesystem path. This rewrites every `package://<package-name>/...` reference in
a URDF to point at a local directory instead, leaving everything else in the file untouched.

Usage:
    python scripts/coppeliasim/convert_urdf_paths.py \
        --input scripts/coppeliasim/assets/fetch.urdf \
        --output scripts/coppeliasim/assets/fetch_fixed.urdf \
        --package-name fetch_description \
        --package-dir scripts/coppeliasim/assets/fetch_description
"""

import argparse
import re
from pathlib import Path


def rewrite_package_uris(urdf_text: str, package_name: str, package_dir: str) -> tuple[str, int]:
    """Replace package://<package_name>/... with an absolute path under package_dir."""
    package_dir_abs = Path(package_dir).resolve().as_posix()
    pattern = re.compile(rf"package://{re.escape(package_name)}/")
    replacement = f"{package_dir_abs}/"
    rewritten, count = pattern.subn(replacement, urdf_text)
    return rewritten, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source URDF (post-xacro)")
    parser.add_argument("--output", required=True, help="Path to write the fixed-up URDF")
    parser.add_argument(
        "--package-name", required=True,
        help="ROS package name as it appears in package:// URIs (e.g. fetch_description)",
    )
    parser.add_argument(
        "--package-dir", required=True,
        help="Local directory that stands in for the ROS package root "
             "(i.e. the directory containing meshes/, robots/, etc.)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    urdf_text = input_path.read_text(encoding="utf-8")

    rewritten, count = rewrite_package_uris(urdf_text, args.package_name, args.package_dir)
    if count == 0:
        print(
            f"Warning: no 'package://{args.package_name}/' URIs found in {input_path}. "
            "Check --package-name matches the URDF, or the file may already be fixed up."
        )
    else:
        print(f"Rewrote {count} package:// URI(s).")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rewritten, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
