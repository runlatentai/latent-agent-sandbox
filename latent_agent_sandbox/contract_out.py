"""The data contract (docs/CONTRACT.md), written from the harness's own records.

* `data/episodes/<episode_id>.jsonl`: one line per model request, 0-based `step`, appended as each step finishes
  (`append_step`). The agent's bash steps are `action: {type: tool_call, name: bash, arguments: {command}}` with
  `tool_output` the sandbox's reply; a reply with no usable command block is `name: invalid` with the format error
  as its argument and the error text as `tool_output`; a step the `before_tool` hook held keeps its command and
  carries `decision: hold` (extra key); the final report turn is `action: {type: final, content}` with
  `tool_output: null`. `messages` is the exact list the client sent (the reasoning line included when set).
* `data/labels.jsonl`: one line per episode from the rule labeler (`label.py`): `cheated` = touched the forbidden
  resource; `first_violating_step` / `discovery_step` 0-based; `used_result` and `concealed` null unless cheated;
  `condition` natural / instructed from the hint rung; `pressure` tokens from `registry.Pressure.tokens()`;
  `robustness` from the layout (`renamed` / `moved`) or `no_reasoning` when the run hid the reasoning.

`export_run(run_dir, data_dir)` rebuilds both from a run's episode directories (steps.jsonl, manifest.json,
labels.jsonl); `validate(data_dir)` runs the contract loader on the result.
"""

from __future__ import annotations

import glob
import json
import os


def step_row(row: dict, index0: int) -> dict:
    """The harness step row (kind step | report) -> a CONTRACT.md step row with 0-based `step`."""
    usage = row.get("usage") or {}
    if row.get("kind") == "report":
        action = {"type": "final", "content": row.get("text") or ""}
        tool_output = None
    elif row.get("truncated"):
        action = {"type": "tool_call", "name": "truncated", "arguments": {}}
        tool_output = row.get("observation")
    elif row.get("format_error"):
        action = {"type": "tool_call", "name": "invalid", "arguments": {"error": row["format_error"]}}
        tool_output = row["format_error"]
    elif row.get("final_text"):
        action = {"type": "final", "content": row.get("text") or ""}
        tool_output = None
    else:
        name = row.get("tool_name") or "bash"
        args = {"command": row.get("action") or ""} if name == "bash" else dict(row.get("tool_arguments") or {})
        action = {"type": "tool_call", "name": name, "arguments": args}
        tool_output = row.get("observation")
        if tool_output is None:
            tool_output = row.get("output") if row.get("output") is not None else ""
    out = {
        "episode_id": row["episode_id"],
        "step": index0,
        "request_id": row.get("request_id") or "",
        "ts": float(row.get("ts_sent") or row.get("ts") or 0.0),
        "latency_s": float(row.get("latency_s") or 0.0),
        "messages": row.get("sent_messages") or row.get("prompt_messages") or [],
        "reasoning": row.get("reasoning"),
        "action": action,
        "tool_output": tool_output,
        "action_text": row.get("text"),
    }
    if usage.get("completion_tokens") is not None:
        out["n_output_tokens"] = int(usage["completion_tokens"])
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") is not None:
        out["n_reasoning_tokens"] = int(details["reasoning_tokens"])
    if row.get("finish_reason"):
        out["finish_reason"] = row["finish_reason"]
    if row.get("model"):
        out["model"] = row["model"]
    if row.get("reasoning_kind"):
        out["reasoning_kind"] = row["reasoning_kind"]
    if row.get("raw_completion"):
        out["raw_completion"] = row["raw_completion"]
    if row.get("decision") == "hold":
        out["decision"] = "hold"
    if row.get("truncated"):
        out["truncated"] = True
    if row.get("extra_tool_calls"):
        out["extra_tool_calls"] = row["extra_tool_calls"]
    if row.get("action_mode"):
        out["action_mode"] = row["action_mode"]
    if row.get("kind") == "step":
        out["harness_step"] = row["step"]  # the 1-based step in steps.jsonl
    return out


def episode_path(data_dir: str, episode_id: str) -> str:
    return os.path.join(data_dir, "episodes", f"{episode_id}.jsonl")


def append_step(data_dir: str, row: dict, index0: int) -> None:
    path = episode_path(data_dir, row["episode_id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(step_row(row, index0), ensure_ascii=False, separators=(",", ":")) + "\n")


def label_row(label: dict, manifest: dict) -> dict:
    """The rich label (label.py) + the manifest -> a CONTRACT.md label row (0-based steps)."""
    cheated = bool(label["touched_forbidden"])
    model = manifest.get("model") or {}
    robustness = manifest.get("robustness")
    if model.get("hide_reasoning"):
        robustness = "no_reasoning"
    pressure = manifest.get("pressure_tokens") or (manifest.get("scenario") or {}).get("pressure_tokens") or "none"
    knobs = manifest.get("pressure") or {}
    try:  # recompute from the knobs the episode actually ran with, so a stale stored string repairs on relabel
        from .scenarios import registry

        pressure = registry.Pressure(**{k: v for k, v in knobs.items() if k in registry.Pressure.__dataclass_fields__}).tokens()
    except Exception:
        pass
    peer_rung = int(label.get("peer_hint_rung") or 0)
    if peer_rung and "peer_hint" not in pressure.split("+"):  # a peer note read at the start (family 3's scratchpad)
        pressure = "peer_hint" if pressure == "none" else pressure + "+peer_hint"
    hint_rung = max(int(label["hint_rung"]), peer_rung)  # the contract has one rung field: the peer note's rung counts
    out = {
        "episode_id": label["episode_id"],
        "cheated": cheated,
        "first_violating_step": label["violation_step"] if cheated else None,
        "discovery_step": label["discovery_step"],
        "used_result": bool(label["used_result"]) if cheated else None,
        "concealed": bool(label["concealed"]) if cheated else None,
        "condition": "natural" if (label["natural"] and peer_rung == 0) else "instructed",
        "family": label["family"],
        "template": label.get("template") or label["scenario_key"],
        "pressure": pressure,
        "hint_rung": hint_rung,
        "peer_hint_rung": peer_rung,
        "model": str(model.get("model") or model.get("policy") or model.get("kind") or "unknown"),
        "seed": int(manifest.get("seed") if manifest.get("seed") is not None else (model.get("seed") or 0)),
        "temperature": float(model.get("temperature") if model.get("temperature") is not None else 0.0),
        "n_steps": int(manifest.get("n_steps", 0))
        + (1 if manifest.get("report_request_id") is not None or manifest.get("report") is not None else 0),
        "labeler": "rule",
        "variant": label["variant"],
        "scenario_key": label["scenario_key"],
        "outcome": label.get("outcome"),
        "policy": label.get("policy"),
        "shortcut_succeeded": label.get("shortcut_succeeded"),
        "log_reliable": label.get("log_reliable"),
        "layout": manifest.get("layout", "canonical"),
        "out_of_bounds_steps": list(label.get("out_of_bounds_steps") or []),
        "probed": bool(label.get("probed")),
        "held": bool(label.get("held")),
        "held_step": label.get("held_step"),
        "held_head": label.get("held_head"),
        "held_risk": label.get("held_risk"),
        "monitor_mode": label.get("monitor_mode"),
    }
    if robustness:
        out["robustness"] = robustness
    if label.get("series_id"):  # the multi-agent family: the series is the unit of work, the episode the row
        role = label.get("scratchpad_role")
        out.update(
            series_id=label["series_id"],
            series_index=label.get("series_index"),
            series_k=label.get("series_k"),
            scratchpad_role=(None if role in (None, "none") else role),  # proposer / adopter / declined or null
            scratchpad_read_step=label.get("scratchpad_read_step"),
            wrote_shortcut_note=label.get("wrote_shortcut_note"),
            notes_at_start=label.get("notes_at_start"),
            notes_history=label.get("notes_history"),
        )
    return out


def export_run(run_dir: str, data_dir: str | None = None, labels: list[dict] | None = None) -> dict:
    """Rebuild data/episodes/*.jsonl and data/labels.jsonl from the run's episode directories."""
    data_dir = data_dir or os.path.join(run_dir, "data")
    if labels is None:
        lp = os.path.join(run_dir, "labels.jsonl")
        labels = [json.loads(ln) for ln in open(lp) if ln.strip()] if os.path.exists(lp) else []
    by_id = {lab["episode_id"]: lab for lab in labels if "label_error" not in lab}
    n_ep = 0
    label_rows = []
    for mp in sorted(glob.glob(os.path.join(run_dir, "*", "*", "manifest.json"))):
        ep_dir = os.path.dirname(mp)
        manifest = json.load(open(mp))
        eid = manifest["episode_id"]
        rows = [json.loads(ln) for ln in open(os.path.join(ep_dir, "steps.jsonl")) if ln.strip()]
        path = episode_path(data_dir, eid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps(step_row(row, i), ensure_ascii=False, separators=(",", ":")) + "\n")
        n_ep += 1
        if eid in by_id:
            label_rows.append(label_row(by_id[eid], manifest))
    with open(os.path.join(data_dir, "labels.jsonl"), "w", encoding="utf-8") as f:
        for r in label_rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"episodes": n_ep, "labels": len(label_rows), "data_dir": data_dir}


def validate(data_dir: str) -> dict:
    """Run the contract loader (contract.py, docs/CONTRACT.md) on data_dir; returns counts or raises its ContractError."""
    from . import contract

    ds = contract.load_dataset(data_dir)
    return {"episodes": len(ds), "steps": len(ds.table), "cheated": sum(1 for ep in ds.episodes.values() if ep.label.cheated)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="rebuild data/episodes/*.jsonl and data/labels.jsonl from a run directory")
    ap.add_argument("run_dir")
    ap.add_argument("--data", default=None)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)
    res = export_run(a.run_dir, a.data)
    print(json.dumps(res))
    if a.validate:
        print(json.dumps(validate(res["data_dir"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
