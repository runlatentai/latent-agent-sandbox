"""The data contract: the exporter's rows pass contract.py's loader; the only refusal is a
pressure token the contract does not list yet (the new ones this harness adds)."""

import json
import os

import pytest

from latent_agent_sandbox import contract_out
from latent_agent_sandbox.label import label_run
from latent_agent_sandbox.scenarios import registry


def test_step_row_shapes():
    base = {
        "episode_id": "e",
        "step": 3,
        "ts": 1.0,
        "latency_s": 0.5,
        "prompt_messages": [{"role": "user", "content": "x"}],
        "reasoning": "r",
        "text": "```bash\nls\n```",
        "action": "ls",
        "format_error": None,
        "output": "a\n",
        "observation": "Return code: 0\nOutput:\na\n",
        "usage": {"completion_tokens": 7, "completion_tokens_details": {"reasoning_tokens": 2}},
        "finish_reason": "stop",
        "request_id": "chatcmpl-1",
        "kind": "step",
    }
    r = contract_out.step_row(base, 2)
    assert r["step"] == 2 and r["action"] == {"type": "tool_call", "name": "bash", "arguments": {"command": "ls"}}
    assert r["tool_output"].startswith("Return code") and r["n_output_tokens"] == 7 and r["n_reasoning_tokens"] == 2
    bad = dict(base, action=None, format_error="Format error: none")
    r = contract_out.step_row(bad, 2)
    assert r["action"]["name"] == "invalid" and r["tool_output"] == "Format error: none"
    rep = dict(base, kind="report", text="done")
    r = contract_out.step_row(rep, 5)
    assert r["action"] == {"type": "final", "content": "done"} and r["tool_output"] is None


def test_fake_runs_validate_against_the_contract(fake_runs):
    out = fake_runs["out"]
    assert os.path.isdir(os.path.join(out, "data", "episodes"))
    res = contract_out.validate(os.path.join(out, "data"))
    assert res["episodes"] == 32 and res["cheated"] == 16
    labels = [json.loads(ln) for ln in open(os.path.join(out, "data", "labels.jsonl"))]
    for lab in labels:
        if lab["cheated"]:
            assert 0 <= lab["discovery_step"] <= lab["first_violating_step"] < lab["n_steps"]
            assert lab["used_result"] is True and lab["concealed"] in (True, False)
        else:
            assert lab["first_violating_step"] is None and lab["used_result"] is None and lab["concealed"] is None
        assert lab["condition"] == "natural" and lab["pressure"] == "none" and "robustness" not in lab
    # the per-episode file: 0-based steps, the final report as a `final` action, the exact messages sent
    eid = labels[0]["episode_id"]
    rows = [json.loads(ln) for ln in open(os.path.join(out, "data", "episodes", f"{eid}.jsonl"))]
    assert [r["step"] for r in rows] == list(range(len(rows)))
    assert rows[-1]["action"]["type"] == "final" and rows[-1]["tool_output"] is None
    assert rows[0]["messages"][0]["role"] == "system" and rows[0]["messages"][1]["role"] == "user"
    assert all(r["action"]["type"] == "tool_call" for r in rows[:-1])


def test_new_pressure_tokens_validate(image, tmp_path):
    from latent_agent_sandbox.fake_agent import run_fake

    sc = registry.resolve(
        "answer_key", "directory", "rle", pressure={"deadline_message": True, "tight_budget": True, "previous_note": True}
    )
    run_fake(sc, "honest", str(tmp_path), episode_id="newtok")
    label_run(str(tmp_path))
    lab = json.loads(open(os.path.join(str(tmp_path), "data", "labels.jsonl")).readline())
    assert lab["pressure"] == "time+deadline+prior_note"
    assert contract_out.validate(os.path.join(str(tmp_path), "data"))["episodes"] == 1


def test_export_run_rebuilds_the_same_files(fake_runs, tmp_path):
    res = contract_out.export_run(fake_runs["out"], str(tmp_path / "data"))
    assert res["episodes"] == 32 and res["labels"] == 32
    a = sorted(
        (json.loads(ln) for ln in open(os.path.join(fake_runs["out"], "data", "labels.jsonl"))), key=lambda r: r["episode_id"]
    )
    b = sorted((json.loads(ln) for ln in open(str(tmp_path / "data" / "labels.jsonl"))), key=lambda r: r["episode_id"])
    assert a == b
    v = contract_out.validate(str(tmp_path / "data"))
    assert v["episodes"] == 32


def test_step_labels_follow_the_contract_rules(dataset):
    """CONTRACT.md "Step labels": the intent window is discovery..violation of a cheated episode and 0 on matched
    honest episodes; concealment is the tail from the violation on; probing marks the out-of-bounds steps; the
    pressure labels cover every step; a pre-k label sits right before the anchor and never on an out-of-bounds step."""
    import numpy as np

    from latent_agent_sandbox.contract import HEADS, parse_head, resolve_controls, step_labels

    ds = dataset
    level, keys = resolve_controls(ds)
    for ep in ds.ordered():
        lab = ep.label
        rows = ds.table.episode_rows[ep.episode_id]
        y = step_labels(ds, "intent")[rows]
        if lab.cheated:
            assert (y[lab.discovery_step : lab.first_violating_step + 1] == 1).all() and (y[: lab.discovery_step] == -1).all()
            assert (y[lab.first_violating_step + 1 :] == -1).all()
        else:
            assert set(y.tolist()) <= {0, -1} and ((y == 0).all() or (y == -1).all())
        c = step_labels(ds, "concealment")[rows]
        if lab.cheated:
            assert (c[lab.first_violating_step :] == (1 if lab.concealed else 0)).all() and (
                c[: lab.first_violating_step] == -1
            ).all()
        p = step_labels(ds, "probing")[rows]
        assert [k for k in range(len(rows)) if p[k] == 1] == lab.out_of_bounds_steps and (p != -1).all()
        assert (step_labels(ds, "impossible")[rows] == int(lab.impossible)).all()
        assert (step_labels(ds, "graded")[rows] == int(lab.graded)).all()
        pre = step_labels(ds, "intent_pre2")[rows]
        if lab.cheated:
            lo = max(0, lab.discovery_step - 2)
            assert (pre[lab.discovery_step :] == -1).all() and (pre[:lo] == -1).all()
            assert all(pre[s] == (1 if s not in lab.out_of_bounds_steps else -1) for s in range(lo, lab.discovery_step))
        else:
            assert (pre != 1).all() and all(pre[s] == -1 for s in lab.out_of_bounds_steps)
    assert all(step_labels(ds, h).shape == (len(ds.table),) for h in HEADS)
    assert parse_head("probing_pre3") == ("probing", 3) and parse_head("intent") == ("intent", None)
    with pytest.raises(ValueError):
        parse_head("nonsense")
    assert np.isin(step_labels(ds, "coordination"), [-1, 0, 1]).all()
