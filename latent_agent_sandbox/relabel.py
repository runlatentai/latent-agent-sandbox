"""Re-label stored episodes from their artifacts alone and rewrite labels.jsonl and data/labels.jsonl in place.

    python relabel.py <out_dir> [--dry-run]

An episode directory holds everything the labeler reads: manifest.json (the scenario rules, the pressure, the
model), steps.jsonl (every step's command, tool output, and the access-log events of that step), grade.json (the
hidden-test result, the final source, the canary grep). No container and no model is needed. Prints how many
labels changed and the cheated / discovered counts before and after; `--dry-run` writes nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

from .label import label_run

KEYS = ("touched_forbidden", "violation_step", "discovery_step", "used_result", "concealed")


def _read(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    return {r["episode_id"]: r for r in (json.loads(ln) for ln in open(path) if ln.strip()) if "episode_id" in r}


def relabel(out_dir: str, dry_run: bool = False) -> dict:
    before = _read(os.path.join(out_dir, "labels.jsonl"))
    n_eps = len(glob.glob(os.path.join(out_dir, "*", "*", "manifest.json")))
    if dry_run:
        tmp = tempfile.mkdtemp(prefix="relabel-")
        try:
            rows = label_run(out_dir, out_path=os.path.join(tmp, "labels.jsonl"), data_dir=tmp)
            after_c = {}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        rows = label_run(out_dir)
        after_c = _read(os.path.join(out_dir, "data", "labels.jsonl"))
    after = {r["episode_id"]: r for r in rows if "label_error" in r or "episode_id" in r}
    changed = [eid for eid, r in after.items() if eid in before and any(r.get(k) != before[eid].get(k) for k in KEYS)]
    flips = {
        eid: {k: (before[eid].get(k), after[eid].get(k)) for k in KEYS if after[eid].get(k) != before[eid].get(k)}
        for eid in changed
    }
    per = {}
    for name, src in (("before", before), ("after", after)):
        for r in src.values():
            k = (
                r.get("scenario_key", "?").rsplit("/", 1)[0]
                if r.get("scenario_key", "").count("/") == 2
                else r.get("scenario_key", "?")
            )
            d = per.setdefault(
                k, {"before": {"n": 0, "cheated": 0, "discovered": 0}, "after": {"n": 0, "cheated": 0, "discovered": 0}}
            )[name]
            d["n"] += 1
            d["cheated"] += int(bool(r.get("touched_forbidden")))
            d["discovered"] += int(r.get("discovery_step") is not None)
    summary = {
        "out_dir": out_dir,
        "episodes": n_eps,
        "labeled": len(rows),
        "errors": sum(1 for r in rows if "label_error" in r),
        "per_scenario": per,
        "changed": len(changed),
        "cheated_before": sum(1 for r in before.values() if r.get("touched_forbidden")),
        "cheated_after": sum(1 for r in after.values() if r.get("touched_forbidden")),
        "discovered_before": sum(1 for r in before.values() if r.get("discovery_step") is not None),
        "discovered_after": sum(1 for r in after.values() if r.get("discovery_step") is not None),
        "contract_rows": len(after_c) if not dry_run else None,
        "dry_run": dry_run,
        "flips_first_20": dict(list(flips.items())[:20]),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(json.dumps(relabel(a.out_dir, a.dry_run), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
