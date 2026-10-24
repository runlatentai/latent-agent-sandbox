"""The multi-agent family's unit of work: a SERIES of K sequential episodes on one template that share a docker
volume mounted at /shared. Each episode is an ordinary `run_episode` (its own directory, episode JSONL, label row)
with `series_id`, `series_index`, `series_k`, and the note history it saw (`notes_history`: the scratchpad after
each earlier episode) on its manifest. The scratchpad is seeded by the first episode's setup from the scenario's
`peer_note` (`none` = empty, `hard` = the bare note; `found` / `location` only when set and recorded as peer_hint);
the harness never writes a note that names the shortcut on its own: the first proposer is the model.

    run_series("answer_key", "directory", "rle", {"impossible": True, "shared_scratchpad": True}, k=6, out_root=...,
               series_id="...", model_factory=lambda i, scenario: ChatModel(...))

`model_factory(index, scenario)` builds the model for episode `index` (a fresh ChatModel per episode, or a
scripted policy that depends on the scenario and the notes it will find: `scenario.rules["_notes_at_start"]`).
"""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable

from .agent import run_episode
from .sandbox import IMAGE, LIFECYCLE_TIMEOUT, HarnessTimeout
from .scenarios import registry

DEFAULT_K = 6


def child_episode_id(series_id: str, index: int) -> str:
    return f"{series_id}-e{index:02d}"


def _volume(name: str) -> str:
    subprocess.run(["docker", "volume", "create", name], check=True, capture_output=True, timeout=LIFECYCLE_TIMEOUT)
    return name


def _read_notes(volume: str, image: str) -> str:
    try:
        r = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "cat",
                "-v",
                f"{volume}:/shared",
                image,
                "/shared/notes.md",
            ],
            capture_output=True,
            text=True,
            timeout=LIFECYCLE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise HarnessTimeout(f"reading the scratchpad exceeded {LIFECYCLE_TIMEOUT:.0f}s") from e
    return r.stdout if r.returncode == 0 else ""


def run_series(
    family: str,
    variant: str,
    template: str | None,
    pressure: dict,
    *,
    k: int = DEFAULT_K,
    out_root: str,
    series_id: str | None = None,
    model_factory: Callable,
    image: str = IMAGE,
    seed: int = 0,
    layout: str = "canonical",
    cmd_timeout: float = 60.0,
    actions: str = "auto",
    policy_names: list[str] | None = None,
    extra_manifest: dict | None = None,
    on_episode: Callable[[dict], None] | None = None,
) -> list[dict]:
    """K sequential episodes on one shared volume. Returns the manifests in order. `on_episode(manifest)` runs after
    each episode (the shard runner labels and records it there). A failing episode ends the series: the later
    episodes would see a scratchpad the failure left half-written."""
    series_id = series_id or time.strftime("%Y%m%dT%H%M%S") + "-s" + uuid.uuid4().hex[:6]
    pressure = dict(pressure or {}, shared_scratchpad=True)
    vol = _volume(f"las-shared-{uuid.uuid4().hex[:10]}")
    manifests: list[dict] = []
    history: list[str] = []
    try:
        for i in range(k):
            sc = registry.resolve(family, variant, template, pressure, seed=seed + i, layout=layout)
            notes = _read_notes(vol, image)
            if not notes and i == 0:  # the first episode's setup writes the seed: what the agent will find
                notes = (sc.setup.get("shared") or {}).get("seed_note") or ""
            sc.rules["_notes_at_start"] = notes
            model = model_factory(i, sc)
            m = run_episode(
                sc,
                model,
                out_root,
                episode_id=child_episode_id(series_id, i),
                image=image,
                cmd_timeout=cmd_timeout,
                actions=actions,
                volumes=[(vol, "/shared")],
                policy_name=(policy_names[i] if policy_names and i < len(policy_names) else getattr(model, "policy", None)),
                extra_manifest=dict(
                    extra_manifest or {},
                    series_id=series_id,
                    series_index=i,
                    series_k=k,
                    series_volume=vol,
                    notes_history=list(history),
                ),
            )
            manifests.append(m)
            history.append(m.get("shared_notes_at_end") or "")
            if on_episode:
                on_episode(m)
            if m.get("outcome") in ("harness_timeout", "harness_error"):
                break
    finally:
        subprocess.run(["docker", "volume", "rm", "-f", vol], capture_output=True, timeout=LIFECYCLE_TIMEOUT)
    return manifests
