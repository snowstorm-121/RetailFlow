# Upstream sources

## T1: LangChain ReAct Agent template

- Source: https://github.com/langchain-ai/react-agent
- Fixed commit: `c93701b297e643c7f80926d78247e5dc06c87bfa`
- Obtained: 2026-10-20, from GitHub's commit-specific source archive; no upstream `.git` copied.
- Archive SHA-256: `5a2bf10247ae654a83bbcfabb3e2990b49b6e14efe5c8c294bb9eea115d35f43`
- Copied files: the seven files in `src/react_agent/` and `LICENSES/react-agent-MIT.txt`.
- Unchanged template files: `__init__.py`, `prompts.py`.
- T1 edits: `graph.py` passes the local model context and records non-streaming HTTP results; `context.py` holds local settings; `utils.py` constructs `ChatOllama` and logs request/response JSON; `tools.py` replaces the example Tavily search with three local, read-only smoke tools.
- The original template lists cloud model providers and Tavily. RetailFlow installs the dependencies in its own `pyproject.toml` and does not use those template integrations.
- T4 edits: `graph.py` adds a separate retail `StateGraph` around the existing model/tool loop, with a guarded tool node for one-call execution and pause states; `state.py` adds `RetailState`; `tools.py` adds nine local retail tool schemas alongside the unchanged T1 smoke tools. `src/retailflow/runtime.py` and `cli.py` provide the application-side confirmation and CLI. The upstream template graph remains available for T1 checks.
- T5 edits: the retail loop in `graph.py` uses a bounded context view, saves each model/tool step, enforces persisted episode budgets, and retries a known pre-execution read failure at most once. `tools.py` requires unique `ask_user.fields`; the same check is enforced at runtime. New `src/retailflow/context.py` and the application runtime/storage changes implement local context, session, and receipt recovery. No upstream retail business method or frozen source/data file changed.

## T2: tau2-bench retail business environment

- Source: https://github.com/sierra-research/tau2-bench at fixed commit `2174a603f6d014ef94473ffa95957f6ce27100db` (upstream package version 1.0.1).
- Obtained: 2026-10-20, twelve commit-specific raw files under `vendor/tau_retail/source/`. Their byte counts and SHA-256 values are recorded in `data/manifests/sources.json`; the copies are unmodified.
- License: repository `LICENSE` says MIT, copyright Sierra Research 2025. Keep this notice with substantial copies. The repository-wide license is present, but the downloaded data files contain no separate file-level redistribution statement; review before publishing raw database/task contents.
- RetailFlow-specific changes: `configs/evaluation.json` manually selects and scripts 13 full-goal source tasks; `src/retailflow/dataset.py` records provenance, creates private task records, and validates source/data hashes and split/state invariants. These are derivatives, not modifications to the upstream files.
- T3 runtime adaptation: `src/retailflow/upstream/tools.py` and `data_model.py` are local copies of those two fixed files. A line diff shows only import-path substitutions (and the `__main__` import path in `tools.py`); the retail business method bodies and Pydantic fields are unchanged. `compat.py` supplies the tiny generic `DB`, `ToolKitBase`, `ToolType` and `is_tool` surface needed here; `paths.py` points at the same fixed raw DB. RetailFlow only exposes the eight planned tools through `src/retailflow/tools.py`. No additional upstream files or packages were downloaded.

## T6: local evaluation adaptations

- New `evaluation.py` and `user_script.py` call the existing `run_turn`, enforce script/scorer visibility separation, register identical read-fault overlays, and record final-state/confirmation/communication checks with all assigned episodes retained. No upstream evaluator or leaderboard result is claimed.
- `context.py` and the retail graph select B0/B1/B2 message views and bounded retries; `runtime.py` publishes a common trusted application result after confirmation. `environment.py` exposes proposal display facts and merges actual normalized model changes into the original raw snapshot, avoiding default-field pollution of untouched objects. These are RetailFlow adaptations; upstream method bodies and raw files remain unchanged.
- `configs/evaluation.json` adds execution, fault, consent and review settings. `data/manifests/evaluation-t6.json` freezes exact versions and hashes. Derived task files and their T2 manifest are unchanged. New tests and reports preserve negative Dev outcomes and separate raw from reviewed results. Public evidence contains only the already-authorized simulated retail data; runtime databases and model files stay local.

## T7: frozen Test execution

- Added only a CLI preflight and `frozen.py` to enforce the registered environment before Test inference. The T7 launch manifest explicitly links the T6 parent and records this source delta; runtime, scorer, prompt, schema, script configuration and data remain byte-identical to T6.
- Four existing Test tasks were run once per variant; all results, assistant-assisted communication reviews and actual traces are retained separately. No source business methods or frozen task labels were changed after observing Test. This is local derived evaluation, not an upstream leaderboard score.
