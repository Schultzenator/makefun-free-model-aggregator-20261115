# makefun-free-model-aggregator-20261115

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Schultzenator/makefun-free-model-aggregator-20261115/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Schultzenator/makefun-free-model-aggregator-20261115/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Deterministic Node 22 router that serves OpenAI and Anthropic APIs across ten simulated free-model providers, translates OpenAI/Anthropic/Gemini wires, and fails over without paid models.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6myn9npx&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

---

## Architecture

- **`main.ts`** — the entrypoint. Reads the brief (`$MAKEFUN_PROMPT_FILE`), probes every provider's catalog route in the provider's own wire, builds the one catalog document (printed by `node main.ts --list`, byte for byte as the `POST $MAKEFUN_ENV_URL/register` body's `catalog`), listens on `0.0.0.0` at an OS-chosen port, registers, serves the twenty requests the env's client sends one at a time, polls `GET $MAKEFUN_ENV_URL/work` every 100 ms and exits 0 the moment it says `{"done": true}`.
- **Router (quality, 60%)** — each request's requirements come from the workload manifest; every free model that meets them is ranked by the quality table's `rating_milli`, and the manifest is planned up front so the highest-rated capacity lands on the heaviest requests while no provider is ever planned beyond the smallest per-trial allowance its free tier draws from (`PLAN_CAPS`, the sealed provider table's `calls_min`) — so a first choice never hits a spent budget. `GET /v1/models` answers in the OpenAI shape for `Authorization: Bearer` and the Anthropic shape for `x-api-key`; a JSON-mode OpenAI request (its `response_format`) never routes to an anthropic-wire provider, which cannot echo it.
- **Failover (10%)** — a refusal (402/429/5xx) is a strike. A provider with no prior strikes is retried once on the very next call number (a refusal is usually one call of a sparse failure schedule); a second strike retires the provider for the trial, and the request walks down the ranked list to the next live, under-cap free model. A rescue pass may retry an already-tried candidate when nothing else remains (the vision-and-tools wall holds only three models). The smallest-budget provider is left unspent by the plan as the reserve.
- **`lib/wire.ts`** — the three providers' wires under `lib/` translations: requests pass through untouched on the same wire or are translated between OpenAI, Anthropic and Gemini shapes, preserving exactly what the provider's echo checker compares (system text, last user message's text and image bytes, tool names and order, the exact `max_tokens` or its absence, stop sequences, JSON-mode flag `response_format` ↔ `generationConfig.responseMimeType`). Answers return in the client's own wire — verbatim on the same wire, or through one neutral shape (whole text, stop family, usage) rendered to Chat Completions JSON/chunks or Messages JSON/SSE events — with stop reasons mapped (`stop`↔`end_turn`↔`STOP`) and usage carried; Chat Completions streams always carry `stream_options.include_usage`.
- **`lib/providers.ts`** — the brief's three documents (provider directory, quality table, workload manifest), the auth headers, the catalog probe.
- **`scripts/practice.mjs`** — the local practice harness: runs a practice trial end-to-end against the sealed env (`makefun-program/env/server.mjs`) and sealed scorer (`makefun-program/scorer/score.mjs`), with `--replay` for the byte-for-byte determinism check and `--list` for the catalog proof inside and outside a trial.
- **Determinism** — no wall clock, no unseeded randomness in any request path (fixed `created` stamps in the model lists), one request at a time in arrival order, every routing choice a pure function of the brief, the requests and the providers' answers. The server reaches nothing but the env its own variables name; no model call is made (the competition's `model_calls: optional`), so the makefun agent SDK is not vendored.

## Practice verification

All three practice trials, run live against the sealed kit (`node scripts/practice.mjs practice-reference/trial-<n>`):

| trial | score | quality | served | wire | failover | catalog | calls | working |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial-0 | **99** | 99 | 100 | 100 | 100 | 100 | 27 | 1 |
| trial-1 | **99** | 99 | 100 | 100 | 100 | 100 | 29 | 1 |
| trial-2 | **99** | 99 | 100 | 100 | 100 | 100 | 29 | 1 |

- Every trial: 20/20 requests answered validly; trial-0 passed 233/233 response checks.
- Trial-0 replay check (`--replay`): the second run left the same state export byte for byte (47131 bytes, both runs) — the determinism the final's replay scoring demands.
- Quality 99 is the column's ceiling: the best-rated eligible model (1475) lives on one provider whose per-trial budget cannot serve every request that wants it, so the plan spends it on the heaviest requests and the rest take the 1470 tier.
- Syntax checks: `node --check` passes on `main.ts`, `lib/wire.ts`, `lib/providers.ts`, `scripts/practice.mjs`; `node main.ts --list` outside a trial prints `{"providers":[]}` and exits 0.

## Notes for the submission (not part of the entry)

The GitHub About line is `repo_about`'s from the requirements
(shell command under `how_to_submit.repo_about.gh`, homepage
`https://make.fun/bounty/free-model-aggregator-01`, topics `makefun,ai-agents,bounties,code`).
