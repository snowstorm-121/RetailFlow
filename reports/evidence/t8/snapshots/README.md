# RetailFlow

RetailFlow is a local, multi-turn retail support Agent built for an auditable engineering and interview demonstration. It adapts the fixed [LangChain ReAct agent template](https://github.com/langchain-ai/react-agent) and the retail environment from [τ2-bench](https://github.com/sierra-research/tau2-bench). The model runs through a local Ollama endpoint; the application controls business permissions, user confirmation, state changes, and receipts.

**Current state (2026-10-20):** T1–T7 engineering is complete. Real local demonstrations include order queries, confirmed cancellation across CLI restarts, and policy refusal. The frozen Dev comparison is **B0 0/9, B1 0/9, B2 0/9**: missing-identity openings ended in plain-text questions rather than `ask_user` calls. This is an unresolved model decision failure; no retry/context benefit was measured. [T6](reports/t6-dev.md) preserves raw results, semantic reviews, native-protocol diagnostics and the freeze manifest. The frozen Test comparison is **B0 0/4, B1 0/4, B2 0/4** across four tasks in three source-user groups. [T7](reports/t7-test.md) records schema-valid but blocked requests and incomplete communication. T8 demonstration and interview materials are ready; user oral acceptance is pending. The 13 derived tasks (Dev 9 / Test 4) are a starting set, **not** an official τ2-bench score or an 80-task result.

## What runs today

- Eight fixed retail business tools plus `ask_user` are offered to `ChatOllama.bind_tools`. The model chooses one next tool at a time; no reference tool chain is hard-coded into the graph.
- A session must locate one user before accessing their profile or orders. Each thread has an independent copy of the fixed retail database.
- A cancellation or return first creates a proposal without changing business state. The CLI displays the exact parameters and asks for `yes` or `no`. Only the application can accept the proposal; the model cannot pass an approval flag.
- SQLite stores business snapshots, complete conversation history, operation receipts, and episode budgets. Replaying a committed confirmation retrieves its receipt. New messages revoke uncommitted proposals, including approvals saved before an interrupted write.
- Model context retains verified state, the latest user intent, and complete recent tool blocks within an estimated budget. Only a known pre-execution read failure can retry once; failures and each attempt remain in traces.
- The CLI writes local traces under `artifacts/`. Selected real traces, including a failed model decision, are in [examples/traces](examples/traces/).

## Run locally

Requirements: Python 3.12, `uv`, and a running local Ollama service with `qwen3:4b-instruct-2507-q4_K_M` already available. No cloud API is configured. Dependency installation is explicit; this repository does not download a model.

```bash
uv sync --locked
uv run python -m retailflow.cli validate --manifest data/manifests/dataset.json
uv run python -m retailflow.cli chat --thread demo-001 --config configs/local.json
uv run python -m pytest -q
```

For a single initial message, add `--message 'I am Yara Muller, ZIP 85041. What is the status of order #W5056519?'`. The default CLI uses `runtime/business.sqlite` for local sessions. Use a new `--thread` value for an independent initial business state. `artifacts/`, `runtime/`, `.venv/`, and tool caches are ignored by Git.

To resume a pending proposal, run `chat` again with the same thread and config, without `--message`; it displays the saved operation before accepting `yes/no`. Defaults are 6 model requests per turn, 24 per episode, 10 user events including confirmations, and 900 seconds of wall time including downtime. These limits are configurable with `max_model_calls_per_turn`, `max_model_calls_per_episode`, `max_user_turns_per_episode`, and `max_episode_seconds`. Expired episodes require a new thread. Context token estimates are approximate; actual model input usage is logged separately.

## Evidence and boundaries

The [T1 feasibility report](reports/t1-feasibility.md) records local native tool-calling smoke tests. [T2](reports/t2-data.md) records fixed-source task selection and hashes. [T3](reports/t3-tools.md) tests the real business methods, permission and confirmation guard, and replay protection. [T4](reports/t4-agent.md) gives real-model message/tool/state traces. [T5](reports/t5-reliability.md) records 30 targeted and 82 total regression tests, context trimming, fault injection, and real CLI restart evidence. `answered` means a model response was produced; it does **not** mean a business task succeeded. [T6](reports/t6-dev.md) scores final state, confirmation, forbidden side effects and required communication; 13 evaluation tests, 9 freeze-entry tests and 104 total tests pass. A completed batch with exit code 0 can still have zero successful tasks.

The public repository includes the fixed upstream retail source data and all 13 derived task records for reproducibility. `data/manifests/` records source hashes. The frozen `sources.json` retains the original machine's absolute source path as provenance; validation prefers the checked-out `vendor/tau_retail/source` directory, so it also works in a clone. Original upstream files and their license notices are preserved; the adaptation inventory is in [UPSTREAM.md](UPSTREAM.md). The upstream repository carries an MIT license, while its raw data files have no separate file-level notice. No license has yet been selected for RetailFlow's original code.

## Dev evaluation

```bash
uv run python -m retailflow.cli evaluate --split dev --variant B0 --config configs/evaluation.json --output artifacts/my-dev-B0
```

Use a new output directory for each B0/B1/B2 run. This calls the local model sequentially and keeps failures in the denominator. Objective candidates require a separately recorded semantic review; the runner does not equate `answered` with success. The [frozen manifest](data/manifests/evaluation-t6.json), [portable evidence](reports/evidence/t6/) and [problem records](reports/problems.md) document the existing run. Test is now observed; future tuning cannot treat these four tasks as unseen. Frozen Test runs require `--frozen` and reject `--limit`; see the [pre-registered protocol](reports/t7-protocol.md) and [results](reports/results.json).

## Demonstration and interview guide

Start with the Chinese [three-path demonstration guide](docs/demos.md): real-model confirmed cancellation, a direct application test showing changed parameters invalidate approval, and real-model policy refusal explanation. The direct guard demo runs with:

```bash
.venv/bin/python examples/confirmation_change.py
```

It calls real business methods in a new local simulation, uses no model, and is not a benchmark result. T8 real-model traces and final-state checks are in [reports/evidence/t8](reports/evidence/t8/). A successful demonstration does not change the frozen Dev/Test failures.

Read the [two-minute introduction and code questions](docs/interview.md), [upstream/adaptation contribution table](reports/contribution.md), and [layered acceptance record](reports/acceptance.md). Implementation and debugging were assisted by Codex; these materials do not establish independent authorship or verified understanding. The resume has not been changed.

## Reference machine and operating limits

The verified reference machine is a MacBook Air, Apple M5, 16 GB unified memory, with Python 3.12.15 and Ollama 0.15.6. The cached Qwen3 4B Q4_K_M model file is approximately 2.50 GB. These are observations, not minimum hardware requirements or a measured peak RAM budget; loaded-model bytes, process RSS and unified memory must not be added as interchangeable measurements. Exact model digest and runtime are in the [T8 machine record](reports/evidence/t8/machine.json).

The existing environment needs no install for the `.venv/bin/python` commands above. On a fresh checkout, `uv sync --locked` is an explicit dependency installation step and may use the network; obtain the fixed model separately if absent. Nothing in the demo script downloads a model or changes shared Ollama settings. There is no web UI, multi-Agent system, model training, MCP, Docker deployment or production authentication claim. SQLite persistence is implemented in this project; it is not an integration with LangGraph's native SqliteSaver.

Original-code licensing remains undecided. Public GitHub access does not automatically license all RetailFlow additions under MIT; retained upstream notices apply to their respective source material.
