"""Verify the files in the release manifest using Python's standard library."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys


def main():
    root = Path(__file__).resolve().parent
    entries = (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    seen, errors = set(), []
    for entry in entries:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", entry)
        if not match:
            errors.append("Malformed manifest entry")
            continue
        expected, name = match.groups()
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or ":" in name or name in seen):
            errors.append("Invalid or repeated manifest path: " + name)
            continue
        seen.add(name)
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root):
            errors.append("Missing or out-of-tree file: " + name)
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(2**20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            errors.append("Hash mismatch: " + name)
    if not seen:
        errors.append("Manifest is empty")
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"Verified {len(seen)} manifest entries.")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
