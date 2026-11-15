# makefun-free-model-aggregator-20261115

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Schultzenator/makefun-free-model-aggregator-20261115/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Schultzenator/makefun-free-model-aggregator-20261115/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One deterministic Node 22 server that exposes an OpenAI-compatible Chat Completions API and an
Anthropic-compatible Messages API over ten simulated free-tier providers. It reads each provider's live
catalog, routes every request to the highest-rated free model that satisfies the request, translates
between the OpenAI, Anthropic and Gemini wires, and fails over deterministically when a provider refuses.

This is the entry for make.fun's [The best free AI model aggregator](https://make.fun/bounty/free-model-aggregator-01)
bounty. It uses no paid models, no external packages and no network endpoint except the trial's
`MAKEFUN_ENV_URL`.

## Architecture

```text
client (OpenAI Chat Completions / Anthropic Messages, JSON or SSE)
        |
        v
  main.ts  --- deterministic manifest planner (quality first, budget-aware caps)
        |         |
        |         +--> GET /v1/models in the shape named by the caller's auth header
        |         +--> POST /register with the exact probed catalog
        |         +--> GET /work poll; exit 0 when the env says done
        |
        v
  lib/providers.ts  --- brief parser, catalog probes, capability/rating helpers
  lib/wire.ts       --- request/answer translation for OpenAI, Anthropic and Gemini wires
        |
        v
ten providers under MAKEFUN_ENV_URL/p/<id> (OpenAI, Anthropic and Gemini wire simulations)
```

- `main.ts` plans the 20 manifest slots in descending weight order, chooses the best-rated free model
  meeting `tools`, `vision`, `json` and `min_context`, and keeps each provider at or below the smallest
  per-trial free allowance from the sealed table. A provider refusal adds a strike; two strikes retire
  it for the trial, and the request moves to the next-best eligible candidate. No wall clock or
  unseeded randomness takes part in a routing decision.
- `lib/wire.ts` translates requests and answers through one neutral shape. The same-wire path relays the
  provider's bytes unchanged; the cross-wire path preserves system text, last-user text, tool names,
  image bytes, `max_tokens`, stop sequences, JSON mode, usage, stop family and SSE event order.
- `lib/providers.ts` parses the brief's three JSON documents (provider directory, sealed quality table,
  workload manifest), probes catalogs in each provider's own wire, and builds the catalog registered
  with the env. Catalog `base_url` values are the providers' own `/p/<id>` paths, never a live loopback
  URL, so a replay exports identical bytes.

## Run it

The trial harness provides the environment; there is nothing to install.

```sh
# Inside a make.fun trial (MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE set):
node --version           # v22 or newer
node main.ts             # serve until the env reports /work done, then exit 0

# Print the catalog document the server registers:
node main.ts --list
```

Local practice (uses only loopback and the sealed kit in this repository):

```sh
node scripts/practice.mjs practice-reference/trial-0
node scripts/practice.mjs practice-reference/trial-0 --replay
node scripts/practice.mjs practice-reference/trial-1 --replay
node scripts/practice.mjs practice-reference/trial-2 --replay
```

## Inputs

- `MAKEFUN_ENV_URL`: the trial env's origin. The server dials only providers under it.
- `MAKEFUN_PROMPT_FILE`: the brief with the provider directory, quality table and workload manifest.
- `MAKEFUN_TRIAL_SEED`: supplied by the trial; routing does not use it or any clock.

`--list` outside a trial prints `{"providers":[]}` because there is no env to probe. Inside a trial it
prints the exact catalog object that `POST /register` carries.

## Score evidence

All three local practice trials pass every per-request check and replay byte-for-byte:

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

`233/233` wire/fidelity checks pass in each trial, and each replay export matches its first export byte
for byte. The full logs are in `evidence/` and `EXECUTION-EVIDENCE.md`.

## Limitations

- Quality 99, not 100, is the practical ceiling on the practice instances: the best generic free model
  (rating 1475) has a per-trial budget, so some weighted requests must be served by the next-best free
  model (1470/1465/1452). The quality column uses the unbudgeted best as its bound.
- The server is single-process and handles client requests one at a time in arrival order. That is the
  trial's contract and it keeps routing deterministic; it is not a high-throughput production gateway.
- The catalog is whatever the providers list at probe time. A provider that fails its catalog probe is
  omitted rather than guessed; a request that no probed free model can satisfy returns a wire-correct
  503 instead of inventing a model.
- Only the ten providers in the brief are planned. The code does not hard-code their model lists; it
  reads them from the live catalogs and the brief's quality table.
- The competition's final phase replays finalists; this server intentionally avoids wall-clock and
  randomness so replay exports match. Any future feature that adds timing must keep routing decisions
  independent of it.

## Competition

- Bounty: [The best free AI model aggregator](https://make.fun/bounty/free-model-aggregator-01)
- Competition kit and local scorer: `makefun-program/`, `makefun-sdk/`, `practice-reference/`
- License: MIT (see `LICENSE`)

