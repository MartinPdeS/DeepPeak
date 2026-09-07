#!/usr/bin/env python3
"""Calculate the next semantic version from Git tags."""

from __future__ import annotations

import re
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"major", "minor", "patch"}:
        raise SystemExit("usage: next_release_version.py {major|minor|patch}")
    tags = subprocess.check_output(
        ["git", "tag", "--list", "v*"], text=True
    ).splitlines()
    versions = [
        tuple(map(int, match.groups()))
        for tag in tags
        if (match := re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag))
    ]
    major, minor, patch = max(versions, default=(0, 0, 0))
    match sys.argv[1]:
        case "major":
            major, minor, patch = major + 1, 0, 0
        case "minor":
            minor, patch = minor + 1, 0
        case "patch":
            patch += 1
    print(f"v{major}.{minor}.{patch}")


if __name__ == "__main__":
    main()
