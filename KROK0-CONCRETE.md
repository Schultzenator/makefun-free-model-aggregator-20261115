# KROK0-CONCRETE — make.fun free-model-aggregator-01

Goal: submit a prize-eligible entry to make.fun bounty `free-model-aggregator-01` before 2026-11-16T02:30:00Z.

Build a deterministic Node 22 server in `main.ts` that implements OpenAI Chat Completions + Anthropic Messages, JSON and SSE, discovers/registers an exact free-model catalog, translates requests/responses across OpenAI/Anthropic/Gemini provider wires, and chooses the highest-quality eligible free model with budget-aware deterministic failover.

Hard constraints from RULES.txt:
- never use paid models; never contact any network except the provided env/provider base URLs;
- serve all 20 requests correctly; maximize quality (60%), served (15%), wire (10%), failover (10%), catalog (5%);
- tie-break: fewer provider calls, so first-choice routing must be strong and failover only on provider refusal;
- deterministic live/replay state; no wall clock or unseeded randomness;
- support tools, images, JSON mode, system prompts, max_tokens, usage, stop reasons, tool results and streaming event order correctly;
- catalog must exactly reflect provider catalogs and `/v1/models` must list exactly registered free model IDs;
- process exits 0 once env `/work` reports done.

Development evidence requirements:
- use Codex CLI as the coding harness and preserve its native session logs;
- at least 5 substantive assistant turns; revise final code in at least two separate assistant edits; keep >30% counted code attributable to the recorded session;
- run all practice trials and kit checks before submission;
- independent final review must be Vibe CLI; FAIL => fix via Codex and re-review until PASS.

Do not exploit scorer/env bugs. Optimize only by implementing the intended routing/protocol behavior.
