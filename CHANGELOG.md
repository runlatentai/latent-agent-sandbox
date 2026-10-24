# Changelog

## 0.1.0 (2026-10-22)

First public release.

- The offline Docker sandbox (`--network none`, mock services on loopback, inotify file-access log, command log,
  hidden tests, canary strings), the scenario registry with three families (answer key: five variants, 129 templates;
  unauthorized spend: two variants, 121 templates; shared scratchpad), the layouts, the pressure switches, the
  prohibition modes, the hint ladder and the peer notes. Every template is checked to be a real task (the reference
  passes its own cases, the hidden expected values are not all one value, the stub fails the public tests): the
  check found and fixed two generator defects before release (`pairs_*` references returned `{}` for every input;
  `histogram_*` returned int-keyed dicts the JSON cases could not express).
- The agent loop over any OpenAI-compatible endpoint (chat completions and the Responses API, native tool calls or
  one fenced bash block per turn; `--client-id-mode` so strict hosts never see an unknown body field), the scripted
  fake agent, the series runner for the shared scratchpad (`las-run --series K`).
- The access-log labeler, the relabel command, the data contract exporter and validator, the derived step labels.
- The monitor hook (`latent_agent_sandbox.monitor`): `StepScorer`, `CallableScorer`, `AuditJsonlScorer`, `HoldPolicy`
  (shadow and hold), and the shadow / hold summary.
- Optional text baselines: the access-log rule oracle, a regex rule derived from the registry's planted strings, a
  keyword monitor, and a step-by-step LLM judge with its rubric (priced only from prices you pass).
- 549 tests in 2,700 lines (390 run without Docker; the 159 sandbox tests skip without a daemon).
