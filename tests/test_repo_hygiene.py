"""Checks on the files that git tracks, which is what the public repository shows."""
import json
import re
import subprocess
from pathlib import Path

import pytest

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

# A home directory: C:\Users\<name>\ on Windows, /Users/<name>/ on macOS,
# /home/<name>/ on Linux, /Workspace/Users/<name>/ on Databricks. Executed
# notebooks pick these paths up from temporary folders and warnings, and publish
# the account name with them. Templates such as /Users/{user}/ are fine.
HOME_PATH = re.compile(
    r"(?i:\b[a-z]:[\\/]+users[\\/]+)[^\\/\s\"'<>{}$%]+"
    r"|(?<![\w.])/(?:home|Users)/[^/\s\"'<>{}$%]+/"
    r"|/Workspace/Users/[^/\s\"'<>{}$%]+/"
)


def tracked_files():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / p for p in out.decode("utf-8").split("\0") if p]


def notebook_strings(node):
    """Every string in a notebook cell, except base64 image payloads."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from notebook_strings(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            if not key.startswith("image/"):
                yield from notebook_strings(value)


def test_pattern_matches_home_paths_only():
    for path in (r"C:\Users\jdoe\AppData\Local\Temp\x.py", "c:/users/jdoe/data",
                 "/home/jdoe/repo/a.png", "/Users/jdoe/Downloads",
                 "/Workspace/Users/jdoe@example.com/.bundle"):
        assert HOME_PATH.search(path), path
    for path in (r"C:\Users\<name>\x", r"C:\Users\%USERNAME%\x", "/home/<name>/",
                 "/home/$USER/x", "/Users/{ctx.userName().get()}/x",
                 "/Workspace/Users/${workspace.current_user.userName}/x",
                 "data/home/x/", "https://example.com/Users/jdoe/", r"D:\data\Users"):
        assert not HOME_PATH.search(path), path


def test_no_home_directory_path_in_tracked_files():
    found = []
    for path in tracked_files():
        # Skip files deleted in the working tree, and this file, whose samples
        # above are home-directory paths on purpose.
        if not path.is_file() or path.resolve() == THIS_FILE:
            continue
        raw = path.read_bytes()
        if b"\0" in raw:  # binary
            continue
        text = raw.decode("utf-8", errors="replace")
        name = path.relative_to(ROOT).as_posix()
        if path.suffix == ".ipynb":
            for ci, cell in enumerate(json.loads(text)["cells"]):
                found += [f"{name} cell {ci}: {m.group(0)}"
                          for s in notebook_strings(cell) for m in HOME_PATH.finditer(s)]
        else:
            found += [f"{name}:{i}: {m.group(0)}"
                      for i, line in enumerate(text.splitlines(), 1)
                      for m in HOME_PATH.finditer(line)]
    assert not found, "home-directory paths in tracked files:\n" + "\n".join(found)
