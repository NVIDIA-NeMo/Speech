# Evaluation Fixture Data

This directory holds scenario fixtures (databases, ground-truth metadata) consumed by
the voice-agent evaluation system. Files here are loaded into `shared_state` on the
bot server via `Scenario.setup_shared_state` (see
[`evaluation/README.md`](../README.md)). The directory is resolvable via the
`EVAL_DATA_ROOT` env var (defaults to this path).

Per-domain data is grouped under a domain-prefixed subdirectory or filename so
that scenarios from different upstream libraries don't collide.

## Sources & Licenses

### eva (`eva_airline_*`)

- **Upstream**: [github.com/ServiceNow/eva](https://github.com/ServiceNow/eva)
- **Version**: `0.1.3`
- **License**: MIT
- **Contents** (verbatim copy, no local modifications):
  - `eva_airline_scenarios/` (50 files) — scenario databases (`reservations`,
    `journeys`, `disruptions`, etc.). Each file is a self-contained world state
    keyed by an eva scenario ID like `1.1.2.json`. Source: `data/airline_scenarios/`.
  - `eva_airline_dataset.jsonl` (50 lines) — scenario metadata (`user_goal`,
    `decision_tree`, `information_required`, `ground_truth.expected_scenario_db`).
    Used to translate eva scenarios into NeMo `Scenario` subclasses. Source:
    `data/airline_dataset.jsonl`.
- **Bound code**: `nemo/agents/voice_agent/evaluation/scenarios/data/eva_airline/` (package: `base.py` holds the `EvaAirlineBaseScenario` + 5 hand-authored seed scenarios; `group_Nx.py` modules carry the auto-scaffolded scenarios for each eva sub-flow)
  + `nemo/agents/voice_agent/evaluation/tools/eva_airline_tools.py` +
  `eva_airline_params.py`. Each ported code file carries an inline
  `# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3` attribution.

## Adding a new source

Add a new section above using the same fields. If the new source overlaps a
domain already present (e.g., a second airline-data library), namespace the
files with a distinct prefix (`tau_airline_*`, `cs_airline_*`, etc.) to avoid
collisions, and register a corresponding `Scenario` subclass that resolves
fixtures via `get_eval_data_root() / "<prefix>_scenarios"`.
