"""`run_pilot --rerun-errored`: only the harness_error jobs of run.json are re-run, in place."""

import json

from latent_agent_sandbox.run_pilot import errored_jobs


def test_errored_jobs_picks_only_the_failed_episodes(tmp_path):
    jobs = [{"episode_id": f"e{i}", "family": "answer_key", "variant": "directory", "template": f"t{i}"} for i in range(4)]
    results = [
        {"episode_id": "e0", "outcome": "submitted"},
        {"episode_id": "e1", "outcome": "harness_error", "error": "x"},
        {"episode_id": "e2", "outcome": "harness_error", "error": "y"},
        {"episode_id": "e3", "outcome": "final_text"},
    ]
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"jobs": jobs, "results": results}))
    doc, bad = errored_jobs(str(p))
    assert [j["episode_id"] for j in bad] == ["e1", "e2"] and len(doc["jobs"]) == 4
    p.write_text(json.dumps({"jobs": jobs, "results": []}))
    assert errored_jobs(str(p))[1] == []


def test_completed_manifest_distinguishes_finished_from_partial(tmp_path):
    from latent_agent_sandbox.run_pilot import completed_manifest

    d = tmp_path / "ep"
    d.mkdir()
    assert completed_manifest(str(d)) is None
    (d / "manifest.json").write_text(json.dumps({"episode_id": "e", "scenario_key": "a/b/c"}))
    assert completed_manifest(str(d)) is None  # started, no outcome yet
    (d / "manifest.json").write_text(
        json.dumps({"episode_id": "e", "scenario_key": "a/b/c", "outcome": "submitted", "n_steps": 3})
    )
    assert completed_manifest(str(d))["outcome"] == "submitted"
