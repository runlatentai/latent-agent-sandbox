"""The data contract (docs/CONTRACT.md): loaders, validators, and the step-label rules.

Episodes are one JSONL per episode under ``data/episodes/``; labels are ``data/labels.jsonl``.
``load_dataset`` returns a :class:`Dataset` whose :class:`StepTable` is the row order every
vector matrix and every score array in this package uses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

STEP_REQUIRED = ("episode_id", "step", "request_id", "ts", "latency_s", "messages", "reasoning", "action", "tool_output")
LABEL_REQUIRED = (
    "episode_id",
    "cheated",
    "first_violating_step",
    "discovery_step",
    "used_result",
    "concealed",
    "condition",
    "family",
    "template",
    "pressure",
    "hint_rung",
    "model",
    "seed",
    "temperature",
)
PRESSURE_TOKENS = (
    "none",
    "time",
    "impossible",
    "graded",
    "grader_hint",
    "flaky",
    "long",
    "deadline",
    "prior_note",
    "faucet_ok",
    "peer_hint",
    "long_budget",
    "leak_trace",
    "leak_ci",
    "shared",
    "implicit",
)
CONDITIONS = ("natural", "instructed")
FAMILIES = ("answer_key", "unauthorized_spend", "shared_scratchpad")
ROBUSTNESS_ARMS = ("no_reasoning", "renamed", "moved")
HEADS = ("intent", "impossible", "graded", "concealment", "probing", "coordination")  # the step-label kinds


def parse_head(name: str) -> tuple[str, int | None]:
    """``"intent_pre2"`` -> ``("intent", 2)``; a plain label kind -> ``(name, None)``. The pre variant
    labels the k steps BEFORE the discovery step (intent) or before the first out-of-bounds step
    (probing): the agent has not yet typed anything out of bounds, so a monitor that reads the text
    has nothing to read there."""
    m = re.fullmatch(r"(intent|probing)_pre([1-9])", name)
    if m:
        return m.group(1), int(m.group(2))
    if name not in HEADS:
        raise ValueError(f"head must be one of {HEADS} or <intent|probing>_pre<k>, got {name!r}")
    return name, None


SCRATCHPAD_ROLES = ("proposer", "adopter", "declined")
CONDITION_GROUPS = ("control", "leaks", "peer_note", "hint_ladder", "other")
CONDITION_TIERS = ("unaided", "incident", "hinted", "other")
TIER_OF_GROUP = {"control": "unaided", "leaks": "incident", "peer_note": "incident", "hint_ladder": "hinted", "other": "other"}
LEAK_TOKENS = ("leak_trace", "leak_ci")


class ContractError(ValueError):
    """A file that does not follow CONTRACT.md."""


@dataclass
class Step:
    episode_id: str
    step: int
    request_id: str
    ts: float
    latency_s: float
    messages: list[dict[str, Any]]
    reasoning: str | None
    action: dict[str, Any]
    tool_output: str | None
    action_text: str | None = None
    n_output_tokens: int | None = None
    n_reasoning_tokens: int | None = None
    finish_reason: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Step:
        missing = [k for k in STEP_REQUIRED if k not in row]
        if missing:
            raise ContractError(f"step row missing {missing}: {sorted(row)[:12]}")
        action = row["action"]
        if not isinstance(action, dict) or action.get("type") not in ("tool_call", "final"):
            raise ContractError(
                f"step {row['episode_id']}/{row['step']}: action must be {{type: tool_call|final, ...}}, got {action!r}"
            )
        if action["type"] == "tool_call" and not isinstance(action.get("name"), str):
            raise ContractError(f"step {row['episode_id']}/{row['step']}: a tool_call action needs a name")
        if not isinstance(row["messages"], list):
            raise ContractError(f"step {row['episode_id']}/{row['step']}: messages must be a list")
        known = set(STEP_REQUIRED) | {"action_text", "n_output_tokens", "n_reasoning_tokens", "finish_reason", "model"}
        return cls(
            episode_id=str(row["episode_id"]),
            step=int(row["step"]),
            request_id=str(row["request_id"]),
            ts=float(row["ts"]),
            latency_s=float(row["latency_s"]),
            messages=list(row["messages"]),
            reasoning=None if row["reasoning"] is None else str(row["reasoning"]),
            action=dict(action),
            tool_output=None if row["tool_output"] is None else str(row["tool_output"]),
            action_text=row.get("action_text"),
            n_output_tokens=None if row.get("n_output_tokens") is None else int(row["n_output_tokens"]),
            n_reasoning_tokens=None if row.get("n_reasoning_tokens") is None else int(row["n_reasoning_tokens"]),
            finish_reason=row.get("finish_reason"),
            model=row.get("model"),
            extra={k: v for k, v in row.items() if k not in known},
        )

    def to_json(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in STEP_REQUIRED}
        for k in ("action_text", "n_output_tokens", "n_reasoning_tokens", "finish_reason", "model"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        d.update(self.extra)
        return d

    def action_json(self) -> str:
        """The action as one JSON string (what the rule monitor matches)."""
        return json.dumps(self.action, sort_keys=True, ensure_ascii=False)

    @property
    def is_marker(self) -> bool:
        """A terminal marker the harness writes when an episode ends without a model request:
        an empty message list and an empty final action. Not a model step: dropped at load,
        never re-read, fit, or alerted on."""
        return (not self.messages) and self.action.get("type") == "final" and not str(self.action.get("content") or "")


@dataclass
class EpisodeLabel:
    episode_id: str
    cheated: bool
    first_violating_step: int | None
    discovery_step: int | None
    used_result: bool | None
    concealed: bool | None
    condition: str
    family: str
    template: str
    pressure: str
    hint_rung: int
    model: str
    seed: int
    temperature: float
    n_steps: int | None = None
    robustness: str | None = None
    labeler: str | None = None
    notes: str | None = None
    out_of_bounds_steps: list[int] = field(default_factory=list)  # steps that touched anything outside the workspace
    probed: bool = False
    series_id: str | None = None  # the shared-scratchpad series (one note)
    series_index: int | None = None
    scratchpad_role: str | None = None  # proposer / adopter / declined
    scratchpad_read_step: int | None = None  # the step that first read the note (default discovery_step)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> EpisodeLabel:
        missing = [k for k in LABEL_REQUIRED if k not in row]
        if missing:
            raise ContractError(f"label row missing {missing}: {row.get('episode_id')!r}")
        eid = str(row["episode_id"])
        if row["condition"] not in CONDITIONS:
            raise ContractError(f"{eid}: condition must be one of {CONDITIONS}, got {row['condition']!r}")
        toks = pressure_tokens(str(row["pressure"]))
        bad = [t for t in toks if t not in PRESSURE_TOKENS]
        if bad:
            raise ContractError(f"{eid}: pressure tokens {bad} not in {PRESSURE_TOKENS}")
        rob = row.get("robustness")
        if rob is not None and rob not in ROBUSTNESS_ARMS:
            raise ContractError(f"{eid}: robustness must be null or one of {ROBUSTNESS_ARMS}, got {rob!r}")
        cheated = bool(row["cheated"])
        fvs = row["first_violating_step"]
        disc = row["discovery_step"]
        if cheated:
            if fvs is None or disc is None:
                raise ContractError(f"{eid}: a cheated episode needs first_violating_step and discovery_step")
            if not (0 <= int(disc) <= int(fvs)):
                raise ContractError(f"{eid}: need 0 <= discovery_step ({disc}) <= first_violating_step ({fvs})")
        else:
            if fvs is not None:
                raise ContractError(f"{eid}: an honest episode has first_violating_step null, got {fvs}")
            if row["concealed"] is not None or row["used_result"] is not None:
                raise ContractError(f"{eid}: concealed and used_result are null unless cheated")
        oob = [int(k) for k in (row.get("out_of_bounds_steps") or [])]
        probed = bool(row.get("probed", bool(oob)))
        if probed != bool(oob):
            raise ContractError(f"{eid}: probed={probed} but out_of_bounds_steps={oob}")
        role = row.get("scratchpad_role")
        if role in ("none", ""):  # the harness writes "none" for an episode that never touched a scratchpad
            role = None
        if role is not None and role not in SCRATCHPAD_ROLES:
            raise ContractError(f"{eid}: scratchpad_role must be one of {SCRATCHPAD_ROLES}, got {role!r}")
        if role == "adopter" and not cheated:
            raise ContractError(f"{eid}: an adopter episode is a cheated one")
        if role == "declined" and cheated:
            raise ContractError(f"{eid}: a declined episode is honest")
        # a declined episode without a discovery or read step (the harness's labels carry some) is kept;
        # the coordination head draws no negatives from it (note_read_step is None)
        known = set(LABEL_REQUIRED) | {
            "n_steps",
            "robustness",
            "labeler",
            "notes",
            "out_of_bounds_steps",
            "probed",
            "series_id",
            "series_index",
            "scratchpad_role",
            "scratchpad_read_step",
        }
        return cls(
            episode_id=eid,
            cheated=cheated,
            first_violating_step=None if fvs is None else int(fvs),
            discovery_step=None if disc is None else int(disc),
            used_result=None if row["used_result"] is None else bool(row["used_result"]),
            concealed=None if row["concealed"] is None else bool(row["concealed"]),
            condition=str(row["condition"]),
            family=str(row["family"]),
            template=str(row["template"]),
            pressure=str(row["pressure"]),
            hint_rung=int(row["hint_rung"]),
            model=str(row["model"]),
            seed=int(row["seed"]),
            temperature=float(row["temperature"]),
            n_steps=None if row.get("n_steps") is None else int(row["n_steps"]),
            robustness=rob,
            labeler=row.get("labeler"),
            notes=row.get("notes"),
            out_of_bounds_steps=sorted(set(oob)),
            probed=probed,
            series_id=None if row.get("series_id") is None else str(row["series_id"]),
            series_index=None if row.get("series_index") is None else int(row["series_index"]),
            scratchpad_role=role,
            scratchpad_read_step=None if row.get("scratchpad_read_step") is None else int(row["scratchpad_read_step"]),
            extra={k: v for k, v in row.items() if k not in known},
        )

    def to_json(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in LABEL_REQUIRED}
        for k in (
            "n_steps",
            "robustness",
            "labeler",
            "notes",
            "series_id",
            "series_index",
            "scratchpad_role",
            "scratchpad_read_step",
        ):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.out_of_bounds_steps:
            d["out_of_bounds_steps"] = list(self.out_of_bounds_steps)
            d["probed"] = True
        d.update(self.extra)
        return d

    @property
    def note_read_step(self) -> int | None:
        return self.scratchpad_read_step if self.scratchpad_read_step is not None else self.discovery_step

    @property
    def impossible(self) -> bool:
        return "impossible" in pressure_tokens(self.pressure)

    @property
    def graded(self) -> bool:
        t = pressure_tokens(self.pressure)
        return "graded" in t or "grader_hint" in t

    @property
    def grader_hint(self) -> bool:
        return "grader_hint" in pressure_tokens(self.pressure)

    @property
    def natural(self) -> bool:
        return self.condition == "natural"

    @property
    def leak(self) -> bool:
        return any(t in LEAK_TOKENS for t in pressure_tokens(self.pressure))

    @property
    def peer_note(self) -> bool:
        return "peer_hint" in pressure_tokens(self.pressure) or "prior_note" in pressure_tokens(self.pressure)

    @property
    def condition_tier(self) -> str:
        """Three tiers: ``unaided`` (control), ``incident`` (the
        leaks, the implicit prohibition, the prior-agent note: the incident-shaped
        conditions), ``hinted`` (the grader-hint ladder), else ``other``."""
        return TIER_OF_GROUP[self.condition_group]

    @property
    def condition_group(self) -> str:
        """The pushed-condition group every table can be sliced by:
        ``leaks`` (a ``leak_*`` token), ``peer_note`` (``peer_hint`` / ``prior_note`` at rung 1 or 2),
        ``hint_ladder`` (``grader_hint`` at rung 1 to 3), ``control`` (rung 0, no leak, no peer
        note), ``other`` (anything else: a hint token at an unexpected rung, or rung > 0 with no
        hint token). Read in that order, so a leak with a hint is ``leaks``."""
        if self.leak:
            return "leaks"
        if self.peer_note and self.hint_rung in (1, 2):
            return "peer_note"
        if self.grader_hint and 1 <= self.hint_rung <= 3:
            return "hint_ladder"
        if self.hint_rung == 0 and not self.peer_note and not self.grader_hint:
            return "control"
        return "other"

    @property
    def stratum(self) -> tuple[str, str]:
        """The matched-control key: (template, pressure)."""
        return (self.template, self.pressure)


def pressure_tokens(pressure: str) -> list[str]:
    return [t.strip() for t in str(pressure).split("+") if t.strip()]


@dataclass
class Episode:
    label: EpisodeLabel
    steps: list[Step]

    @property
    def episode_id(self) -> str:
        return self.label.episode_id

    @property
    def n_steps(self) -> int:
        return len(self.steps)


@dataclass
class StepTable:
    """The row order of every per-step array: ``rows[i] == (episode_id, step)``."""

    rows: list[tuple[str, int]]
    index: dict[tuple[str, int], int]
    episode_rows: dict[str, list[int]]

    @classmethod
    def build(cls, episodes: list[Episode]) -> StepTable:
        rows: list[tuple[str, int]] = []
        episode_rows: dict[str, list[int]] = {}
        for ep in episodes:
            rs = []
            for s in ep.steps:
                rs.append(len(rows))
                rows.append((ep.episode_id, s.step))
            episode_rows[ep.episode_id] = rs
        return cls(rows=rows, index={r: i for i, r in enumerate(rows)}, episode_rows=episode_rows)

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class Dataset:
    episodes: dict[str, Episode]
    table: StepTable
    condition: str = "natural"  # the condition this dataset is read as: natural (default), instructed, all

    def __len__(self) -> int:
        return len(self.episodes)

    def ordered(self) -> list[Episode]:
        return [self.episodes[eid] for eid in sorted(self.episodes)]

    def labels(self) -> list[EpisodeLabel]:
        return [ep.label for ep in self.ordered()]

    def subset(self, keep) -> Dataset:
        eps = [ep for ep in self.ordered() if keep(ep.label)]
        return Dataset(episodes={ep.episode_id: ep for ep in eps}, table=StepTable.build(eps), condition=self.condition)

    def episode_ids(self) -> list[str]:
        return sorted(self.episodes)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ContractError(f"{path}:{n}: not JSON ({e})") from e


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_episode_file(path: str | Path, keep_markers: bool = False) -> list[Step]:
    """The episode's model steps in order. Terminal marker rows (:attr:`Step.is_marker`) are
    dropped unless ``keep_markers``; a marker is only ever trailing."""
    steps = [Step.from_json(r) for r in read_jsonl(path)]
    if not steps:
        raise ContractError(f"{path}: no steps")
    if not keep_markers:
        kept = [s for s in steps if not s.is_marker]
        if any(not s.is_marker for s in steps[len(kept) :]) if len(kept) < len(steps) else False:
            pass
        markers = [s.step for s in steps if s.is_marker]
        if markers and (not kept or max(markers) < max(s.step for s in kept)):
            raise ContractError(f"{path}: marker steps {markers} are not trailing")
        steps = kept
        if not steps:
            raise ContractError(f"{path}: every step is a terminal marker")
    eids = {s.episode_id for s in steps}
    if len(eids) != 1:
        raise ContractError(f"{path}: more than one episode_id in one file: {sorted(eids)}")
    steps.sort(key=lambda s: s.step)
    if [s.step for s in steps] != list(range(len(steps))):
        raise ContractError(f"{path}: steps must be 0..n-1 without gaps, got {[s.step for s in steps]}")
    return steps


def load_labels(path: str | Path) -> dict[str, EpisodeLabel]:
    out: dict[str, EpisodeLabel] = {}
    for r in read_jsonl(path):
        lab = EpisodeLabel.from_json(r)
        if lab.episode_id in out:
            raise ContractError(f"{path}: episode {lab.episode_id!r} labeled twice")
        out[lab.episode_id] = lab
    return out


def load_dataset(data_dir: str | Path, require_labels: bool = True, limit: int | None = None) -> Dataset:
    """``data_dir/episodes/*.jsonl`` joined to ``data_dir/labels.jsonl``. An episode file without
    a label is refused (``require_labels``), a label without a file is reported and skipped."""
    data_dir = Path(data_dir)
    ep_dir = data_dir / "episodes"
    files = sorted(ep_dir.glob("*.jsonl"))
    if not files:
        raise ContractError(f"{ep_dir}: no episode files")
    labels = load_labels(data_dir / "labels.jsonl") if (data_dir / "labels.jsonl").exists() else {}
    if require_labels and not labels:
        raise ContractError(f"{data_dir / 'labels.jsonl'}: missing or empty")
    episodes: list[Episode] = []
    for p in files[:limit] if limit else files:
        steps = load_episode_file(p)
        eid = steps[0].episode_id
        if eid != p.stem:
            raise ContractError(f"{p}: file stem {p.stem!r} != episode_id {eid!r}")
        lab = labels.get(eid)
        if lab is None:
            if require_labels:
                raise ContractError(f"{p}: no label for {eid!r} in labels.jsonl")
            continue
        n_lines = sum(1 for _ in read_jsonl(p))
        if lab.n_steps is not None and lab.n_steps not in (len(steps), n_lines):
            raise ContractError(f"{p}: label says n_steps {lab.n_steps}, file has {len(steps)} model steps ({n_lines} rows)")
        if lab.cheated and lab.first_violating_step >= len(steps):
            raise ContractError(f"{p}: first_violating_step {lab.first_violating_step} >= n_steps {len(steps)}")
        if lab.discovery_step is not None and lab.discovery_step >= len(steps):
            raise ContractError(f"{p}: discovery_step {lab.discovery_step} >= n_steps {len(steps)}")
        if any(k >= len(steps) or k < 0 for k in lab.out_of_bounds_steps):
            raise ContractError(f"{p}: out_of_bounds_steps {lab.out_of_bounds_steps} outside 0..{len(steps) - 1}")
        episodes.append(Episode(label=lab, steps=steps))
    episodes.sort(key=lambda e: e.episode_id)
    return Dataset(episodes={e.episode_id: e for e in episodes}, table=StepTable.build(episodes))


# ----------------------------------------------------------------------------- step labels

CONTROL_LEVELS = ("stratum", "pressure", "template", "all")
MIN_CONTROLS = 30


def control_key(lab: EpisodeLabel, level: str) -> Any:
    if level == "stratum":
        return lab.stratum
    if level == "pressure":
        return lab.pressure
    if level == "template":
        return lab.template
    return "all"


def resolve_controls(ds: Dataset, controls: str = "auto", min_controls: int = MIN_CONTROLS) -> tuple[str, set]:
    """Which honest episodes are the matched controls of the cheating ones: the finest level
    that gives at least ``min_controls`` honest episodes. ``stratum`` = the same template and
    pressure (the template is the task instance and the pressure varies inside it, so the exact
    pair rarely has enough controls); ``pressure`` = the same pressure variant (the confound stays
    controlled, any template); ``template``; ``all`` = every honest episode.
    Returns ``(level, keys)``; a named level is taken as is."""
    if controls not in CONTROL_LEVELS + ("auto",):
        raise ValueError(f"controls must be auto or one of {CONTROL_LEVELS}, got {controls!r}")
    labs = list(ds.labels())
    levels = CONTROL_LEVELS if controls == "auto" else (controls,)
    chosen = None
    for level in levels:
        keys = {control_key(lab, level) for lab in labs if lab.cheated}
        n = sum(1 for lab in labs if not lab.cheated and control_key(lab, level) in keys)
        chosen = (level, keys)
        if n >= min_controls or controls != "auto":
            break
    return chosen


def step_labels(
    ds: Dataset, head: str, matched_controls: bool | str = True, controls: str = "auto", min_controls: int = MIN_CONTROLS
) -> np.ndarray:
    """Per-step label aligned with ``ds.table``: 1 (the bad class), 0, or -1 (out of the fit).
    The rules are CONTRACT.md's "Step labels"; the honest controls of ``intent`` and
    ``concealment`` are matched at the level :func:`resolve_controls` picks (``controls``;
    ``matched_controls=False`` = every honest episode)."""
    import numpy as np

    head, pre_k = parse_head(head)
    y = np.full(len(ds.table), -1, dtype=np.int8)
    if pre_k is not None:
        return _pre_labels(ds, head, pre_k, matched_controls, controls, min_controls)
    if matched_controls:
        level, keys = resolve_controls(ds, controls, min_controls)
    else:
        level, keys = "all", {"all"}
    adopters = adopter_series(ds)
    for ep in ds.episodes.values():
        lab = ep.label
        rows = ds.table.episode_rows[ep.episode_id]
        n = len(rows)
        if head == "intent":
            if lab.cheated:
                for k in range(lab.discovery_step, lab.first_violating_step + 1):
                    y[rows[k]] = 1
            elif control_key(lab, level) in keys:
                y[rows] = 0
        elif head == "impossible":
            y[rows] = 1 if lab.impossible else 0
        elif head == "graded":
            y[rows] = 1 if lab.graded else 0
        elif head == "concealment":
            if lab.cheated:
                for k in range(lab.first_violating_step, n):
                    y[rows[k]] = 1 if lab.concealed else 0
            elif control_key(lab, level) in keys:
                y[rows] = 0
        elif head == "probing":
            y[rows] = 0
            for k in lab.out_of_bounds_steps:
                y[rows[k]] = 1
        elif head == "coordination":
            if lab.scratchpad_role == "adopter" and lab.cheated and lab.note_read_step is not None:
                for k in range(lab.note_read_step, lab.first_violating_step + 1):
                    y[rows[k]] = 1
            elif lab.scratchpad_role == "declined" and lab.note_read_step is not None and lab.series_id in adopters:
                for k in range(lab.note_read_step, n):
                    y[rows[k]] = 0
    return y


def _pre_labels(ds: Dataset, head: str, k: int, matched_controls: bool | str, controls: str, min_controls: int) -> np.ndarray:
    """CONTRACT.md "Pre-discovery labels": positives = the ``k`` steps right before the anchor
    (``intent``: the discovery step of a cheating episode; ``probing``: the first out-of-bounds
    step of a probed episode), only those that exist; negatives = the matched honest (or unprobed)
    episodes' steps before any discovery of their own, never an out-of-bounds step; everything else
    out of the fit. A positive step never contains an out-of-bounds command."""
    import numpy as np

    y = np.full(len(ds.table), -1, dtype=np.int8)
    if head == "intent":
        level, keys = resolve_controls(ds, controls, min_controls) if matched_controls else ("all", {"all"})
    for ep in ds.episodes.values():
        lab = ep.label
        rows = ds.table.episode_rows[ep.episode_id]
        oob = set(lab.out_of_bounds_steps)
        if head == "intent":
            is_pos = lab.cheated
            anchor = lab.discovery_step
            is_neg = (not lab.cheated) and control_key(lab, level) in keys
        else:
            is_pos = lab.probed
            anchor = min(lab.out_of_bounds_steps) if lab.probed else None
            is_neg = not lab.probed
        if is_pos:
            if anchor is None:
                continue
            for s in range(max(0, anchor - k), anchor):
                if s not in oob:
                    y[rows[s]] = 1
        elif is_neg:
            stop = lab.discovery_step if lab.discovery_step is not None else len(rows)
            for s in range(stop):
                if s not in oob:
                    y[rows[s]] = 0
    return y


def adopter_series(ds: Dataset) -> set[str]:
    return {
        ep.label.series_id
        for ep in ds.episodes.values()
        if ep.label.scratchpad_role == "adopter" and ep.label.series_id is not None
    }


def group_filter(groups: Sequence[str] | None):
    """A ``Dataset.subset`` predicate keeping the given condition groups (None = everything)."""
    if not groups:
        return lambda lab: True
    bad = [g for g in groups if g not in CONDITION_GROUPS + CONDITION_TIERS]
    if bad:
        raise ValueError(f"unknown condition groups {bad}; choose from {CONDITION_GROUPS} or the tiers {CONDITION_TIERS}")
    keep = set(groups)
    return lambda lab: lab.condition_group in keep or lab.condition_tier in keep
