"""Read the sandbox logs from given byte offsets, after a sync barrier, and print one JSON document.

Usage: tail.py '{"fs": 0, "results": 0, "payments": 0}'  ->  {"lines": {name: [str, ...]}, "offsets": {name: int}}

The barrier: a file is created under /var/log/sandbox/sync; the watcher writes a {"kind": "sync"} line for it
only after every inotify event queued before it, so every open the agent's command caused is on disk when
this returns. Run as root through `docker exec` after each agent step.
"""

import json
import os
import sys
import time
import uuid

LOGS = {
    "fs": "/var/log/sandbox/fs.jsonl",
    "results": "/var/log/sandbox/results.jsonl",
    "payments": "/var/log/sandbox/payments.jsonl",
}
SYNC_DIR = "/var/log/sandbox/sync"


def main(offsets: dict) -> dict:
    token = uuid.uuid4().hex
    open(os.path.join(SYNC_DIR, token), "w").close()
    marker = json.dumps({"kind": "sync", "ts": 0, "token": token})[:16]  # the prefix is enough to find the line
    synced = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with open(LOGS["fs"]) as f:
            f.seek(int(offsets.get("fs", 0)))
            if any(line.startswith(marker) and f'"token": "{token}"' in line for line in f):
                synced = True
                break
        time.sleep(0.005)
    os.unlink(os.path.join(SYNC_DIR, token))
    out = {"lines": {}, "offsets": {}, "synced": synced}
    for name, path in LOGS.items():
        start = int(offsets.get(name, 0))
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
        end = start + len(data)
        lines = [ln for ln in data.decode("utf-8", errors="replace").split("\n") if ln.strip()]
        if name == "fs":
            lines = [ln for ln in lines if '"kind": "sync"' not in ln]
        out["lines"][name] = lines
        out["offsets"][name] = end
    return out


if __name__ == "__main__":
    offs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(main(offs)))
