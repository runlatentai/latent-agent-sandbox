"""The file-access log: inotify (ctypes, no packages) over the forbidden paths, one JSON line per event.

Usage: watch.py <watch_paths.json> <out.jsonl>. The JSON file lists absolute paths (directories or files).
A directory watch reports the directory itself (a listing opens it: IN_OPEN with IN_ISDIR) and its direct
children by name (a read opens the file: IN_OPEN on the child). A file watch reports that file.

Lines: {"kind": "fs", "ts", "path", "events": [names], "isdir": bool, "wd_path"}, plus
{"kind": "watching", "paths": [...]} once at start, {"kind": "overflow"} if the kernel queue overflowed
(the labeler marks the episode's log unreliable), and {"kind": "sync", "token"} when a file is created
under the sync directory (the barrier tail.py uses: every event queued before the sync is written first).
"""

import ctypes
import json
import os
import struct
import sys
import time

IN_ACCESS = 0x1
IN_MODIFY = 0x2
IN_ATTRIB = 0x4
IN_CLOSE_WRITE = 0x8
IN_CLOSE_NOWRITE = 0x10
IN_OPEN = 0x20
IN_MOVED_FROM = 0x40
IN_MOVED_TO = 0x80
IN_CREATE = 0x100
IN_DELETE = 0x200
IN_DELETE_SELF = 0x400
IN_MOVE_SELF = 0x800
IN_Q_OVERFLOW = 0x4000
IN_IGNORED = 0x8000
IN_ISDIR = 0x40000000
NAMES = {
    IN_ACCESS: "access",
    IN_MODIFY: "modify",
    IN_ATTRIB: "attrib",
    IN_CLOSE_WRITE: "close_write",
    IN_CLOSE_NOWRITE: "close_nowrite",
    IN_OPEN: "open",
    IN_MOVED_FROM: "moved_from",
    IN_MOVED_TO: "moved_to",
    IN_CREATE: "create",
    IN_DELETE: "delete",
    IN_DELETE_SELF: "delete_self",
    IN_MOVE_SELF: "move_self",
}
MASK = (
    IN_ACCESS
    | IN_MODIFY
    | IN_ATTRIB
    | IN_CLOSE_WRITE
    | IN_CLOSE_NOWRITE
    | IN_OPEN
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVE_SELF
)
SYNC_DIR = "/var/log/sandbox/sync"
EVENT_HDR = struct.Struct("iIII")


def main(paths_file: str, out_path: str) -> None:
    paths = json.load(open(paths_file))
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    wd_to_path = {}
    watched = []
    for p in list(paths) + [SYNC_DIR]:
        if not os.path.exists(p):
            continue
        wd = libc.inotify_add_watch(fd, p.encode(), MASK)
        if wd < 0:
            continue
        wd_to_path[wd] = p
        watched.append(p)
    out = open(out_path, "a", buffering=1)
    out.write(json.dumps({"kind": "watching", "ts": time.time(), "paths": watched}) + "\n")
    out.flush()
    while True:
        buf = os.read(fd, 1 << 16)
        off = 0
        while off + EVENT_HDR.size <= len(buf):
            wd, mask, _cookie, ln = EVENT_HDR.unpack_from(buf, off)
            off += EVENT_HDR.size
            name = buf[off : off + ln].split(b"\0", 1)[0].decode(errors="replace")
            off += ln
            ts = time.time()
            if mask & IN_Q_OVERFLOW:
                out.write(json.dumps({"kind": "overflow", "ts": ts}) + "\n")
                continue
            base = wd_to_path.get(wd)
            if base is None or mask & IN_IGNORED:
                continue
            path = os.path.join(base, name) if name else base
            if base == SYNC_DIR:
                if mask & IN_CREATE and name:
                    out.write(json.dumps({"kind": "sync", "ts": ts, "token": name}) + "\n")
                continue
            names = [n for bit, n in NAMES.items() if mask & bit]
            out.write(
                json.dumps(
                    {"kind": "fs", "ts": ts, "path": path, "events": names, "isdir": bool(mask & IN_ISDIR), "wd_path": base}
                )
                + "\n"
            )
        out.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
