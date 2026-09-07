#!/usr/bin/env python3
"""Create a release commit and annotated semantic-version tag."""

from __future__ import annotations

import re
import subprocess
import sys


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"v\d+\.\d+\.\d+", sys.argv[1]):
        raise SystemExit(
            "release aborted: tag must use the form vMAJOR.MINOR.PATCH, for example v0.5.0"
        )
    tag = sys.argv[1]
    if (
        subprocess.run(["git", "diff", "--quiet"]).returncode
        or subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode
    ):
        raise SystemExit("release aborted: working tree has uncommitted changes")
    if (
        subprocess.run(
            ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
            stdout=subprocess.DEVNULL,
        ).returncode
        == 0
    ):
        raise SystemExit(f"release aborted: tag {tag} already exists")
    run("git", "commit", "--allow-empty", "-m", f"Release {tag}")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")


if __name__ == "__main__":
    main()
