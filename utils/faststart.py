"""Pure-Python MP4 "faststart" remux.

OpenCV's ``VideoWriter`` writes the ``moov`` atom (the index) at the *end* of
the file, after ``mdat``. Browsers then cannot begin playback until they have
located that trailing index, which manifests as a ``<video>`` element that
stalls/"sticks" while loading and a burst of HTTP range requests.

``remux_faststart`` relocates ``moov`` to the front (right after ``ftyp``) and
patches the chunk-offset tables (``stco`` / ``co64``) so the file plays
progressively. This mirrors what ``qt-faststart`` / ``ffmpeg -movflags
+faststart`` do, but with no external dependency.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Atoms that contain child atoms we may need to descend into to reach stco/co64.
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}


def _read_top_atoms(data: bytes) -> List[Tuple[bytes, int, int, int]]:
    """Return ``(type, start, size, header_size)`` for each top-level atom."""
    atoms: List[Tuple[bytes, int, int, int]] = []
    off = 0
    n = len(data)
    while off + 8 <= n:
        size = struct.unpack(">I", data[off : off + 4])[0]
        typ = data[off + 4 : off + 8]
        hdr = 8
        if size == 1:
            if off + 16 > n:
                break
            size = struct.unpack(">Q", data[off + 8 : off + 16])[0]
            hdr = 16
        elif size == 0:
            size = n - off
        if size < hdr or off + size > n:
            break
        atoms.append((typ, off, size, hdr))
        off += size
    return atoms


def _patch_chunk_offsets(moov: bytearray, delta: int) -> None:
    """Add ``delta`` to every chunk offset (stco/co64) inside ``moov``."""

    def walk(start: int, end: int) -> None:
        off = start
        while off + 8 <= end:
            size = struct.unpack(">I", moov[off : off + 4])[0]
            typ = bytes(moov[off + 4 : off + 8])
            hdr = 8
            if size == 1:
                size = struct.unpack(">Q", moov[off + 8 : off + 16])[0]
                hdr = 16
            elif size == 0:
                size = end - off
            if size < hdr or off + size > end:
                break
            body = off + hdr
            if typ in _CONTAINERS:
                walk(body, off + size)
            elif typ == b"stco":
                count = struct.unpack(">I", moov[body + 4 : body + 8])[0]
                p = body + 8
                for _ in range(count):
                    if p + 4 > end:
                        break
                    val = struct.unpack(">I", moov[p : p + 4])[0]
                    struct.pack_into(">I", moov, p, (val + delta) & 0xFFFFFFFF)
                    p += 4
            elif typ == b"co64":
                count = struct.unpack(">I", moov[body + 4 : body + 8])[0]
                p = body + 8
                for _ in range(count):
                    if p + 8 > end:
                        break
                    val = struct.unpack(">Q", moov[p : p + 8])[0]
                    struct.pack_into(">Q", moov, p, val + delta)
                    p += 8
            off += size

    # Skip the moov atom's own header before descending into its children.
    size = struct.unpack(">I", moov[0:4])[0]
    hdr = 16 if size == 1 else 8
    walk(hdr, len(moov))


def remux_faststart(src: os.PathLike | str, dst: Optional[os.PathLike | str] = None) -> bool:
    """Move ``moov`` ahead of ``mdat`` so the MP4 streams progressively.

    Returns ``True`` if the file was rewritten, ``False`` if it was already
    fast-start, not an MP4, or could not be parsed (left untouched on failure).
    When ``dst`` is ``None`` the source file is replaced in place.
    """
    src = Path(src)
    try:
        data = src.read_bytes()
    except OSError:
        return False

    atoms = _read_top_atoms(data)
    types = [a[0] for a in atoms]
    if b"moov" not in types or b"mdat" not in types:
        return False

    moov_idx = types.index(b"moov")
    mdat_idx = types.index(b"mdat")
    if moov_idx < mdat_idx:
        return False  # already fast-start

    _, ms, msz, _ = atoms[moov_idx]
    moov = bytearray(data[ms : ms + msz])

    # moov is relocated before mdat, pushing mdat down by exactly len(moov).
    try:
        _patch_chunk_offsets(moov, msz)
    except (struct.error, IndexError):
        return False

    ftyp_blobs = [data[s : s + sz] for (t, s, sz, _h) in atoms if t == b"ftyp"]
    rest_blobs = [
        data[s : s + sz]
        for i, (t, s, sz, _h) in enumerate(atoms)
        if t != b"ftyp" and i != moov_idx
    ]
    new_bytes = b"".join(ftyp_blobs) + bytes(moov) + b"".join(rest_blobs)

    out = Path(dst) if dst is not None else src
    try:
        fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(new_bytes)
        os.replace(tmp, out)
    except OSError:
        try:
            os.unlink(tmp)  # type: ignore[name-defined]
        except OSError:
            pass
        return False
    return True
