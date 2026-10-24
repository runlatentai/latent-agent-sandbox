"""The per-step monitor hook: plug any per-step scorer here.

The agent loop (`agent.run_episode`) calls the monitor once per step, AFTER the model has answered and BEFORE the
tool call runs. The monitor returns a record for the step; `HoldPolicy` turns it into a decision: `run` the
command, or `hold` (the command never runs and the episode ends with outcome `held`). In `--shadow` mode every
reading is recorded and nothing is ever held, which gives the paired "would it have held" table.

Two ways to plug a scorer in:

1. **An in-process scorer** (`StepScorer`): any object with
   `score(step, context) -> dict | None`, where `context` carries what the loop knows about the step
   (`episode_id`, `step`, `request_id`, `client_request_id`, `action` (the command about to run), `text`,
   `reasoning`). It returns an EVENT in the shape below, or None when it has no reading (fail-open, counted).
   `CallableScorer` wraps a plain function. This is where a text judge, a regex rule, a keyword monitor, or a
   call into your own serving stack goes.

2. **An audit file written by a scorer that runs inside the serving stack** (`AuditJsonlScorer`): a serving-side
   monitor (one that reads the model's activations as it generates, or anything else that scores each request)
   appends one JSON line per request to a file; the scorer tails that file and finds each step's event by the
   HTTP response id (`request_id`) or by the client-set id the harness sends with every request
   (`client_request_id` = `<episode_id>:<step>`, carried in the request body under `CLIENT_REQUEST_ID_FIELD`).
   The event usually lands a few milliseconds to a few hundred milliseconds after the response; the scorer polls
   up to `timeout_s`.

The event shape (what both paths hand to `read_head`). This is the shape one serving-side monitor emits (and what
the harness's own `CallableScorer` writes); any JSONL with these keys works: a primary risk with its threshold,
optionally a `suite` of named secondary signals, and the client id echoed under `meta.otel.client_request_id`:

    {"request_id": "<the HTTP response id>",
     "risk": 0.91, "threshold": 0.70, "verdict": "escalate",          # the primary signal
     "meta": {"suite": {"<name>": {"risk": 0.2, "threshold": 0.5, "over": false, "enabled": true}, ...},
              "suite_fired": ["<name>", ...],
              "otel": {"client_request_id": "<episode_id>:<step>"}}}

`--hold-on <name>[:<cut>]` names the signal to act on. The name `PRIMARY_HEAD` ("intent") falls back to the
event's top-level `risk` / `threshold` when it is not a suite entry; any other name is read from `meta.suite`.
When no cut is given the event's own `threshold` for that signal is the cut.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import unquote

# the request-body field the harness sends the client id under (`vllm_xargs.<field>` for a vLLM server, which
# passes it through to its plugins; `metadata.<field>` on the Responses API; nothing on a strict host, see
# `agent.ChatModel` `client_id_mode`). A serving-side scorer that echoes it on the event under
# `meta.otel.client_request_id` lets the two ids cross-check each other.
CLIENT_REQUEST_ID_FIELD = "latent_client_request_id"

PRIMARY_HEAD = "intent"  # the signal read from the event's own `risk` / `threshold` when it is not a suite entry


class StepScorer(Protocol):
    """Anything that can score one agent step. `context` is the loop's view of the step; the return value is an
    event in the module's shape (see the docstring), or None for "no reading"."""

    def score(self, step: int, context: dict[str, Any]) -> dict | None: ...


def _decode_id(rid: str) -> str:
    """Undo the percent-encoding a serving-side scorer may apply to a request id (`/`, `%` and control characters);
    a trailing `#<n>` fragment is kept as is."""
    if "#" in rid:
        head, tail = rid.rsplit("#", 1)
        if tail.isdigit():
            return unquote(head) + "#" + tail
    return unquote(rid)


class CallableScorer:
    """`score(step, context)` through a plain function `fn(step, context) -> float | dict | None`. A float is
    taken as the primary risk with `cut` as its threshold; a dict is returned as the event."""

    def __init__(self, fn: Callable[[int, dict[str, Any]], Any], cut: float = 0.5, head: str = PRIMARY_HEAD) -> None:
        self.fn = fn
        self.cut = cut
        self.head = head
        self.n_lookups = 0
        self.n_misses = 0

    def score(self, step: int, context: dict[str, Any]) -> dict | None:
        self.n_lookups += 1
        out = self.fn(step, context)
        if out is None:
            self.n_misses += 1
            return None
        if isinstance(out, dict):
            return out
        risk = float(out)
        ev: dict[str, Any] = {
            "request_id": context.get("request_id"),
            "risk": risk,
            "threshold": self.cut,
            "verdict": "escalate" if risk > self.cut else "pass",
            "meta": {"suite": {}, "suite_fired": [], "otel": {"client_request_id": context.get("client_request_id")}},
        }
        if self.head != PRIMARY_HEAD:
            ev["meta"]["suite"][self.head] = {"risk": risk, "threshold": self.cut, "over": risk > self.cut, "enabled": True}
        return ev


class AuditJsonlScorer:
    """Tails an audit JSONL that a serving-side scorer appends to (one event per request) and finds each step's
    event by `request_id` or by `meta.otel.client_request_id`, polling up to `timeout_s`."""

    def __init__(self, audit_path: str, timeout_s: float = 10.0, poll_s: float = 0.1) -> None:
        self.audit_path = audit_path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._offset = 0
        self._by_rid: dict[str, dict] = {}
        self._by_crid: dict[str, dict] = {}
        self._partial = ""
        self.n_lookups = 0
        self.n_misses = 0

    def _ingest(self) -> None:
        if not os.path.exists(self.audit_path):
            return
        with open(self.audit_path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
            self._offset = f.tell()
        text = self._partial + data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._partial = lines[-1]  # a line still being written
        for ln in lines[:-1]:
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rid = e.get("request_id")
            if rid:
                self._by_rid[_decode_id(str(rid))] = e
            crid = ((e.get("meta") or {}).get("otel") or {}).get("client_request_id")
            if crid:
                self._by_crid[crid] = e

    def lookup(self, request_id: str | None, client_request_id: str | None, timeout_s: float | None = None) -> dict | None:
        """The event for this step, or None after the timeout."""
        self.n_lookups += 1
        deadline = time.time() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            self._ingest()
            e = (self._by_rid.get(request_id) if request_id else None) or (
                self._by_crid.get(client_request_id) if client_request_id else None
            )
            if e is not None:
                return e
            if time.time() >= deadline:
                self.n_misses += 1
                return None
            time.sleep(self.poll_s)

    def score(self, step: int, context: dict[str, Any]) -> dict | None:
        return self.lookup(context.get("request_id"), context.get("client_request_id"))


def read_head(event: dict | None, head: str | None) -> dict:
    """The monitor record for a step: the primary risk / verdict, the suite, and the chosen signal's reading."""
    if event is None:
        return {"found": False}
    meta = event.get("meta") or {}
    suite = meta.get("suite") or {}
    rec: dict[str, Any] = {
        "found": True,
        "event_request_id": event.get("request_id"),
        "risk": event.get("risk"),
        "verdict": event.get("verdict"),
        "suite": suite,
        "suite_fired": meta.get("suite_fired"),
        "unscored_reason": meta.get("unscored_reason"),
        "capture_inert": meta.get("capture_inert"),
    }
    if head:
        h = suite.get(head)
        rec["head"] = head
        if h is None and head == PRIMARY_HEAD and event.get("risk") is not None:
            # the primary head is not a suite entry: its reading is the event's own risk and threshold
            rec["head_risk"] = event.get("risk")
            rec["head_threshold"] = event.get("threshold") if event.get("threshold") is not None else meta.get("threshold")
            rec["head_present"] = True
            rec["head_source"] = "primary"
        else:
            rec["head_risk"] = None if h is None else h.get("risk")
            rec["head_threshold"] = None if h is None else h.get("threshold")
            rec["head_present"] = h is not None
            rec["head_source"] = "suite" if h is not None else None
    return rec


class HoldPolicy:
    """`before_tool(step, event) -> 'run' | 'hold_end'`. `monitor` is any `StepScorer` (or an object with the older
    `lookup(request_id, client_request_id)` method); the agent loop passes the step's ids and text in `event`."""

    def __init__(self, monitor, hold_on: str | None = None, shadow: bool = False) -> None:
        self.monitor = monitor
        self.shadow = shadow
        self.head, self.cut = None, None
        if hold_on:
            name, _, cut = hold_on.partition(":")
            self.head = name.strip()
            self.cut = float(cut) if cut.strip() else None
        if not shadow and not self.head:
            raise ValueError("a hold policy needs --hold-on <head>[:<cut>] or --shadow")
        self.records: list[dict] = []
        self.held: dict | None = None

    @property
    def mode(self) -> str:
        return "shadow" if self.shadow else f"hold:{self.head}" + (f":{self.cut}" if self.cut is not None else "")

    def _event(self, step: int, request_id: str | None, client_request_id: str | None, context: dict | None) -> dict | None:
        ctx = dict(context or {}, step=step, request_id=request_id, client_request_id=client_request_id)
        if hasattr(self.monitor, "score"):
            return self.monitor.score(step, ctx)
        return self.monitor.lookup(request_id, client_request_id)

    def read(self, step: int, request_id: str | None, client_request_id: str | None, context: dict | None = None) -> dict:
        t0 = time.time()
        ev = self._event(step, request_id, client_request_id, context)
        rec = read_head(ev, self.head)
        rec.update(step=step, lookup_s=round(time.time() - t0, 3), mode=self.mode)
        if self.head and rec.get("found") and rec.get("head_present"):
            cut = self.cut if self.cut is not None else rec.get("head_threshold")
            rec["cut"] = cut
            rec["over"] = rec["head_risk"] is not None and cut is not None and rec["head_risk"] > cut
        else:
            rec["cut"] = self.cut
            rec["over"] = False
        rec["decision"] = "hold" if (rec["over"] and not self.shadow) else "run"
        self.records.append(rec)
        return rec

    def before_tool(self, step: int, event: dict) -> str:
        rec = event.get("monitor") or self.read(step, event.get("request_id"), event.get("client_request_id"), context=event)
        if rec.get("decision") == "hold":
            self.held = {
                "step": step,
                "head": self.head,
                "risk": rec.get("head_risk"),
                "cut": rec.get("cut"),
                "action": event.get("action"),
            }
            return "hold_end"
        return "run"
