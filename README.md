# makefun-free-model-aggregator-20261115

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Schultzenator/makefun-free-model-aggregator-20261115/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Schultzenator/makefun-free-model-aggregator-20261115/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A deterministic Node 22 free-model aggregator: one server that serves the OpenAI Chat Completions and Anthropic Messages APIs — each in JSON and in server-sent events — over ten free-tier model providers on the OpenAI, Anthropic and Gemini wires, registering an exact free-model catalog with the trial env, routing every request to the highest-quality eligible free model, and failing over deterministically when a provider refuses.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6myn9npx&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

---

## Architecture

The whole entry is three runtime files (`main.ts`, `lib/providers.ts`, `lib/wire.ts`) on Node 22, standard library alone; the practice kit used to develop it belongs to the sponsor's sealed development kit, not this repository.

- **`main.ts`** — the entrypoint. Reads the trial brief (`$MAKEFUN_PROMPT_FILE`), probes each provider's catalog route in the provider's own wire, builds the one catalog document (printed by `node main.ts --list`, byte for byte as the `POST $MAKEFUN_ENV_URL/register` body's catalog), listens on `0.0.0.0` at an OS-chosen port, registers, serves the twenty requests the env's client sends one at a time, polls `GET $MAKEFUN_ENV_URL/work` every 100 ms and exits 0 the moment it says done.
- **Router** — each request's requirements come from the brief's workload manifest; every free model that meets them is ranked by the quality table's `rating_milli`, and the manifest is planned up front so the highest-rated capacity lands on the heaviest requests while no provider is ever planned beyond the smallest per-trial allowance its free tier draws from — so a first choice never hits a spent budget. `GET /v1/models` answers in the OpenAI shape for `Authorization: Bearer` and the Anthropic shape for `x-api-key`; a JSON-mode OpenAI request (its `response_format`) never routes to an anthropic-wire provider, which cannot echo it.
- **Failover** — a refusal (402/429/5xx) is a strike. A previously healthy provider is retried once on the very next call number (a refusal is usually one call of a sparse failure schedule); a second strike retires the provider for the trial, and the request walks down the ranked list to the next live, under-cap free model, with a rescue pass when only already-tried candidates remain (the vision-and-tools wall holds just three models). The smallest-budget provider is left unspent by the plan as the reserve.
- **`lib/wire.ts`** — all pairwise translations across the OpenAI, Anthropic and Gemini wires: requests pass through untouched on the same wire or are translated preserving exactly what the provider's echo compares (system text, the last user message's text and image bytes, tool names and order, the exact `max_tokens` or its absence, stop sequences, JSON mode via `response_format` / `generationConfig.responseMimeType`). Answers return in the client's own wire — verbatim on the same wire, or through one neutral shape (whole text, stop family, usage) rendered to Chat Completions JSON/chunks or Messages JSON/SSE events — with stop reasons mapped (`stop`↔`end_turn`↔`STOP`) and usage carried; Chat Completions streams always carry `stream_options.include_usage`.
- **`lib/providers.ts`** — the brief's three documents (provider directory, quality table, workload manifest), the auth headers, the catalog probe.
- **Determinism** — no wall clock, no unseeded randomness in any request path (fixed `created` stamps in the model lists), one request at a time in arrival order, every routing choice a pure function of the brief, the requests and the providers' answers. The server reaches nothing but the env its own variables name, and makes no model call of its own, so no SDK is vendored or needed.

## Practice verification

Validated before submission against the sponsor-provided development kit (its sealed generator, env, scorer and three practice instances), run live end-to-end, then scored by the sealed scorer:

| trial | score | quality | served | wire | failover | catalog | calls | working |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial-0 | **99** | 99 | 100 | 100 | 100 | 100 | 27 | 1 |
| trial-1 | **99** | 99 | 100 | 100 | 100 | 100 | 29 | 1 |
| trial-2 | **99** | 99 | 100 | 100 | 100 | 100 | 29 | 1 |

Every trial: 20/20 requests answered validly (trial-0 passed 233/233 response checks). The trial-0 replay check left the state export byte for byte identical on its second run (47131 bytes both runs) — the determinism the final's replay scoring demands. Quality 99 is the column's ceiling for this workload: the best-rated eligible model (1475) lives on one provider whose per-trial budget cannot serve every request that wants it, so the plan spends it on the heaviest requests and the rest take the 1470 tier.
