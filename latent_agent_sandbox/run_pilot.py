"""Run a pilot: N episodes per scenario against an OpenAI-compatible base_url, in parallel, one container each.

    python run_pilot.py --base-url http://127.0.0.1:8001/v1 --model gpt-oss-20b --episodes 25 \
        --scenarios all --pressure '{"hint_rung": 0}' --workers 8 --out runs/pilot

`--scenarios` is `all` or a comma list of family/variant[/template]; `--pressure` applies to every episode;
`--pressure-grid` is a JSON list of pressure dicts cycled over the episodes (recorded per episode in the manifest).
`--series K` runs every job as a series of K sequential episodes on one shared scratchpad volume
(`series.run_series`; the job's id is the series id, its episodes are `<id>-e00` .. `<id>-e<K-1>`).
Writes <out>/run.json (the arguments, the image id, counts) and one episode directory per episode; then labels
the run (labels.jsonl) and prints the per-scenario summary. The API key is read from the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agent import CLIENT_ID_MODES, ChatModel, run_episode
from .label import label_run, summarize
from .sandbox import IMAGE, build_image, image_exists, image_id
from .scenarios import registry
from .series import run_series


def errored_jobs(run_json: str) -> tuple[dict, list[dict]]:
    """The run document and the jobs whose result was a harness_error (a model server restart can fail a whole
    batch with connection refused). A re-run replaces those episodes in place."""
    with open(run_json) as fh:
        doc = json.load(fh)
    bad = {r["episode_id"] for r in doc.get("results", []) if r.get("outcome") == "harness_error"}
    return doc, [j for j in doc["jobs"] if j["episode_id"] in bad]


def completed_manifest(ep_dir: str) -> dict | None:
    """The manifest of a finished episode (an `outcome` written by run_episode), else None."""
    p = os.path.join(ep_dir, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return None
    return m if m.get("outcome") else None


def plan(
    scenarios: list[str],
    episodes: int,
    pressures: list[dict],
    seed0: int,
    layouts: list[str],
    reasonings: list,
    templates: dict[str, list[str]] | None = None,
) -> list[dict]:
    """One job per (scenario, i): pressures, layouts and reasoning levels cycle over i; the template is drawn
    from `templates[family]` (the split side chosen on the command line) by the job's seed."""
    jobs = []
    for key in scenarios:
        parts = key.split("/")
        fam, var = parts[0], parts[1]
        tpl = parts[2] if len(parts) > 2 else None
        for i in range(episodes):
            seed = seed0 + i
            t = tpl
            if t is None and templates and templates.get(fam):
                t = templates[fam][seed % len(templates[fam])]
            jobs.append(
                {
                    "family": fam,
                    "variant": var,
                    "template": t,
                    "pressure": pressures[i % len(pressures)],
                    "layout": layouts[i % len(layouts)],
                    "reasoning": reasonings[i % len(reasonings)],
                    "seed": seed,
                    "episode_id": f"{time.strftime('%Y%m%d')}-{fam[:3]}-{var[:3]}-{i:04d}-{uuid.uuid4().hex[:6]}",
                }
            )
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--api", default="auto", choices=["auto", "chat", "responses"], help="auto = responses for gpt-oss, chat otherwise"
    )
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--episodes", type=int, default=10, help="per scenario")
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--pressure", default="{}")
    ap.add_argument("--pressure-grid", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=168)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reasoning", default="none", help="comma list of none|low|medium|high, cycled over episodes")
    ap.add_argument(
        "--hide-reasoning",
        action="store_true",
        help="never store the model's reasoning (the label row carries robustness: no_reasoning)",
    )
    ap.add_argument("--layouts", default="canonical", help="comma list of canonical|renamed|moved, cycled")
    ap.add_argument("--split", default=None, choices=["train", "test"], help="draw templates from one side of template_split()")
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument(
        "--client-id-mode",
        default="auto",
        choices=list(CLIENT_ID_MODES),
        help="send the per-step client id as vllm_xargs: auto = only for a gpt-oss model or a vLLM-looking local "
        "endpoint (a strict OpenAI-compatible host answers 400 to an unknown body field); vllm = always; none = never",
    )
    ap.add_argument(
        "--series",
        type=int,
        default=None,
        help="run every job as K sequential episodes on one shared scratchpad volume (series.run_series); "
        "not combined with --audit (the live monitor is per episode)",
    )
    ap.add_argument(
        "--actions",
        default="auto",
        choices=["auto", "text", "tools"],
        help="tools = native tool calls (default for gpt-oss models)",
    )
    ap.add_argument("--reasoning-key", default="reasoning")
    ap.add_argument(
        "--audit",
        default=None,
        help="an audit JSONL written by a serving-side scorer: enables the live monitor (one monitor per episode)",
    )
    ap.add_argument(
        "--hold-on",
        default=None,
        help="<name>[:<cut>]: hold the tool call when the named signal (the event's primary risk, or a meta.suite entry) is over the cut",
    )
    ap.add_argument("--shadow", action="store_true", help="record the scorer's verdicts, never hold")
    ap.add_argument("--audit-timeout", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--cmd-timeout", type=float, default=60.0)
    ap.add_argument(
        "--rerun-errored",
        action="store_true",
        help="re-run only the jobs of <out>/run.json whose outcome was harness_error, in place",
    )
    a = ap.parse_args(argv)
    if a.series is not None and a.series < 1:
        ap.error("--series needs K >= 1")
    if a.series and a.audit:
        ap.error("--series does not take --audit: the live monitor is per episode")
    if not image_exists(a.image):
        print("building", a.image, flush=True)
        build_image(a.image)
    scenarios = registry.all_keys() if a.scenarios == "all" else [s.strip() for s in a.scenarios.split(",") if s.strip()]
    pressures = json.loads(a.pressure_grid) if a.pressure_grid else [json.loads(a.pressure)]
    layouts = [x.strip() for x in a.layouts.split(",") if x.strip()]
    reasonings = [None if x.strip() == "none" else x.strip() for x in a.reasoning.split(",")]
    templates = None
    if a.split:
        sp = registry.template_split(a.seed)
        templates = {fam: sp[fam][a.split] for fam in sp}
    prior_results: list[dict] = []
    if a.rerun_errored:
        run_doc, jobs = errored_jobs(os.path.join(a.out, "run.json"))
        prior_results = [r for r in run_doc.get("results", []) if r.get("outcome") != "harness_error"]
        todo = []
        for j in jobs:  # an interrupted re-run leaves complete episodes behind: keep those, replace partial ones
            slug = f"{j['family']}/{j['variant']}/{j['template']}".replace("/", "_")
            d = os.path.join(a.out, slug, j["episode_id"])
            done = completed_manifest(d)
            if done is not None:
                prior_results.append(
                    {
                        "episode_id": done["episode_id"],
                        "scenario_key": done["scenario_key"],
                        "outcome": done["outcome"],
                        "n_steps": done.get("n_steps"),
                        "rerun_kept": True,
                    }
                )
                continue
            shutil.rmtree(d, ignore_errors=True)
            # the contract file is appended step by step while the episode runs (contract_out.append_step): a failed
            # attempt's rows would sit in front of the re-run's (the validator read "steps must be 0..n-1 without gaps")
            try:
                os.remove(os.path.join(a.out, "data", "episodes", j["episode_id"] + ".jsonl"))
            except FileNotFoundError:
                pass
            todo.append(j)
        kept = len(jobs) - len(todo)
        jobs = todo
        run_doc["rerun"] = {"n": len(jobs), "kept_complete": kept, "started_ts": time.time()}
        print(f"re-running {len(jobs)} errored episodes of {len(run_doc['jobs'])} ({kept} already complete, kept)", flush=True)
    else:
        jobs = plan(scenarios, a.episodes, pressures, a.seed, layouts, reasonings, templates)
        os.makedirs(a.out, exist_ok=True)
        run_doc = {
            "args": {k: v for k, v in vars(a).items() if k != "api_key_env"} | {"api_key_env": a.api_key_env},
            "image_id": image_id(a.image),
            "n_jobs": len(jobs),
            "started_ts": time.time(),
            "jobs": jobs,
        }
    json.dump(run_doc, open(os.path.join(a.out, "run.json"), "w"), indent=1)

    def make_model(job: dict, seed: int) -> ChatModel:
        return ChatModel(
            a.base_url,
            a.model,
            api_key_env=a.api_key_env,
            temperature=a.temperature,
            seed=seed,
            max_tokens=a.max_tokens,
            extra_body=json.loads(a.extra_body),
            api=a.api,
            client_id_mode=a.client_id_mode,
            reasoning_effort=job["reasoning"],
            hide_reasoning=a.hide_reasoning,
            reasoning_key=a.reasoning_key,
        )

    def one_series(job: dict) -> dict:
        """K sequential episodes on one shared volume; the job's id is the series id."""
        ms = run_series(
            job["family"],
            job["variant"],
            job["template"],
            job["pressure"],
            k=a.series,
            out_root=a.out,
            series_id=job["episode_id"],
            model_factory=lambda i, sc: make_model(job, job["seed"] + i),
            image=a.image,
            seed=job["seed"],
            layout=job["layout"],
            cmd_timeout=a.cmd_timeout,
            actions=a.actions,
            extra_manifest={"job": job},
        )
        eps = [
            {"episode_id": m["episode_id"], "scenario_key": m["scenario_key"], "outcome": m["outcome"], "n_steps": m["n_steps"]}
            for m in ms
        ]
        return {
            "episode_id": job["episode_id"],
            "series": eps,
            "scenario_key": eps[-1]["scenario_key"] if eps else None,
            "outcome": eps[-1]["outcome"] if eps else "harness_error",
            "n_steps": sum(e["n_steps"] or 0 for e in eps),
        }

    def one(job: dict) -> dict:
        if a.series:
            try:
                return one_series(job)
            except Exception as e:
                return {
                    "episode_id": job["episode_id"],
                    "outcome": "harness_error",
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                }
        sc = registry.resolve(
            job["family"], job["variant"], job["template"], job["pressure"], seed=job["seed"], layout=job["layout"]
        )
        model = make_model(job, job["seed"])
        policy = None
        if a.audit:
            from .monitor import AuditJsonlScorer, HoldPolicy

            policy = HoldPolicy(AuditJsonlScorer(a.audit, timeout_s=a.audit_timeout), hold_on=a.hold_on, shadow=a.shadow)
        try:
            m = run_episode(
                sc,
                model,
                a.out,
                episode_id=job["episode_id"],
                image=a.image,
                cmd_timeout=a.cmd_timeout,
                extra_manifest={"job": job},
                actions=a.actions,
                monitor_policy=policy,
            )
            return {
                "episode_id": m["episode_id"],
                "scenario_key": m["scenario_key"],
                "outcome": m["outcome"],
                "n_steps": m["n_steps"],
            }
        except Exception as e:
            return {"episode_id": job["episode_id"], "outcome": "harness_error", "error": f"{type(e).__name__}: {str(e)[:300]}"}

    results = list(prior_results)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {json.dumps(r)}", flush=True)
    run_doc.update(
        ended_ts=time.time(),
        results=results,
        outcomes={o: sum(1 for r in results if r.get("outcome") == o) for o in {r.get("outcome") for r in results}},
    )
    json.dump(run_doc, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    rows = label_run(a.out)
    print(json.dumps(summarize(rows), indent=1))
    try:
        from . import contract_out

        print("contract:", json.dumps(contract_out.validate(os.path.join(a.out, "data"))))
    except Exception as e:  # the validator's verdict is printed, never fatal to the run
        print("contract: FAILED", type(e).__name__, str(e)[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
