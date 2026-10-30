"""Content fingerprint of a data file."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_fingerprint(path: Path | str) -> str:
    """First 12 characters of the SHA-256 digest, convenient for logs."""
    return file_sha256(path)[:12]