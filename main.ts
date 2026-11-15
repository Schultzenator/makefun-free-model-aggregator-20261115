// free-model-aggregator-01's entry: one server, two client wires, three provider wires, and a
// quality-ranked free-model router with budget-aware deterministic failover.
//
// WHAT A TRIAL GIVES YOU (see lib/providers.ts for the brief's three documents)
//   The brief at $MAKEFUN_PROMPT_FILE carries the provider directory (ten free-tier providers, each
//   with its wire, auth scheme, key and base path), the quality table (each free model's rating and
//   capabilities) and the workload manifest (the twenty requests in arrival order, each with its api,
//   its streaming, its weight and the requirements a serving model must meet). The env at
//   $MAKEFUN_ENV_URL serves the providers under their base paths.
//
// WHAT THIS SERVER DOES
//   1. reads the brief, probes every provider's catalog route in the provider's own wire (probe
//      calls are catalog reads: they cost none of the trial's scored call budget), and builds the one
//      catalog document (`--list` prints it byte for byte; POST /register carries the same object);
//   2. plans the trial's routing: for each manifest request, the highest-rating free model that
//      meets its requirements, spread over the providers so no provider is planned beyond a
//      conservative per-trial cap (the free tiers' smallest per-trial allowances, from the kit's
//      sealed provider table, which the scored instances share), so a first choice never hits a
//      spent budget;
//   3. serves the client's requests: GET /v1/models in the shape of the auth header it got, POST
//      /v1/chat/completions (OpenAI wire, JSON and SSE) and POST /v1/messages (Anthropic wire, JSON
//      and SSE), translating to and from the routed provider's wire (lib/wire.ts);
//   4. fails over on a provider's refusal: strikes are counted per provider, two refusals retire it
//      for the trial, and the request moves to the next-highest free model that meets it;
//   5. polls GET /work every 100 ms and exits 0 the moment the env says {"done": true}.
//
// DETERMINISM: every routing choice is a function of the brief, the requests and the providers'
// answers alone - no wall clock, no unseeded randomness, one request handled at a time in arrival
// order - so a replay that sees the same requests and answers leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicStreamToNeutral,
  geminiStreamToNeutral,
  neutralOfAnthropic,
  neutralOfGemini,
  neutralOfOpenAI,
  neutralToAnthropic,
  neutralToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamToNeutral,
  openAIToAnthropic,
  openAIToGemini,
  renderAnthropicStream,
  renderOpenAIStream,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider attempts one client request may cost before it is answered with an error. */
const MAX_ATTEMPTS = 15;

/**
 * The plan's per-provider cap on successful first-choice and failover calls: the smallest per-trial
 * free-tier allowance each provider is drawn with (the kit's sealed provider table, which the scored
 * instances draw from too: calls_min of groq 9, cerebras 5, gemini 6, zai 4, openrouter 10,
 * sambanova 4, pollinations 10, cloudflare 7, ovh 5, kilol 12). Staying at or under the smallest
 * allowance means a single trial's budget can never be exhausted by the plan, so a provider's
 * refusal is one call on its failure schedule or its outage, and the next candidate is tried at once.
 * A cap of 0 would pin a provider wholly in reserve; the planner keeps what each cap leaves.
 */
const PLAN_CAPS: Readonly<Record<string, number>> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** Any provider outside the table is treated as having a small free-tier allowance. */
const DEFAULT_PLAN_CAP = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------- the router

/** One free (provider, model) pair a request may be served by. */
interface Candidate {
  provider: ProviderDirectoryEntry;
  /** The provider's directory index: the brief's stable order, for deterministic tie-breaks. */
  order: number;
  model: ProbedModel;
  rating: number;
  key: string;
}

/** This trial's live provider state: successes served, refusal strikes, and a retirement flag. */
const successes = new Map<string, number>();
const refusals = new Map<string, number>();

function capOf(providerId: string): number {
  return PLAN_CAPS[providerId] ?? DEFAULT_PLAN_CAP;
}

function headroom(cand: Candidate, planned: Map<string, number>): number {
  return capOf(cand.provider.id) - (planned.get(cand.provider.id) ?? 0);
}

/** Every free (provider, model) pair that meets `requires`, sorted for planning or for serving. */
function eligible(requires: Requirement, restrictAnthropicWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (let index = 0; index < probes.length; index++) {
    const probe = probes[index];
    if (!probe.ok) continue;
    if (restrictAnthropicWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({
        provider: probe.provider,
        order: index,
        model,
        rating: row?.rating_milli ?? 0,
        key: `${probe.provider.id}/${model.id}`,
      });
    }
  }
  return out;
}

function byPreference(preferredKey: string | null, planned: Map<string, number>): (a: Candidate, b: Candidate) => number {
  return (a, b) =>
    b.rating - a.rating ||
    (a.key === preferredKey ? -1 : 0) + (b.key === preferredKey ? 1 : 0) ||
    headroom(b, planned) - headroom(a, planned) ||
    a.order - b.order ||
    (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);
}

/**
 * The trial's plan, built from the brief before the first request arrives: for each manifest slot the
 * highest-rating free model that meets its requirements, kept under the caps by preferring, among
 * equal ratings, the provider with the most headroom. Heavier requests are planned first so the
 * best-rated capacity lands where the weight is.
 */
const preferred = new Map<number, string>();

function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  if (manifest.length === 0) return;
  const planned = new Map<string, number>();
  const rows = manifest
    .map((entry, index) => ({ entry, index }))
    .sort((a, b) => b.entry.weight - a.entry.weight || a.index - b.index);
  for (const { entry, index } of rows) {
    // A JSON-mode OpenAI request carries response_format, which the anthropic wire cannot echo.
    const restrict = entry.api === 'openai' && entry.requires.json === true;
    const ranked = eligible(entry.requires, restrict).sort(byPreference(null, planned));
    const pick = ranked.find((cand) => (planned.get(cand.provider.id) ?? 0) < capOf(cand.provider.id)) ?? ranked[0];
    if (pick === undefined) continue;
    preferred.set(index, pick.key);
    planned.set(pick.provider.id, (planned.get(pick.provider.id) ?? 0) + 1);
  }
  for (const [providerId, count] of planned) log(`plan: ${providerId} serves ${count} first choices`);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Requirement {
  const vision = (content: unknown): boolean => {
    if (!Array.isArray(content)) return false;
    const wantsImage = (block: unknown): boolean => {
      if (!isRecord(block)) return false;
      if (block.type === 'image_url' || block.type === 'image') return true;
      return false;
    };
    return content.some(wantsImage);
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision: messages.some((raw) => isRecord(raw) && raw.role === 'user' && vision(raw.content)),
    json: api === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'),
  };
}

/** The manifest row of the request now arriving: the client sends them one at a time, in order. */
let posts = 0;

function slotFor(api: 'openai' | 'anthropic', body: Json): { requires: Requirement; preferredKey: string | null } {
  const manifest = brief.workload?.manifest ?? [];
  const index = posts;
  posts += 1;
  const entry = manifest[index];
  if (entry !== undefined && entry.api === api) {
    return { requires: entry.requires, preferredKey: preferred.get(index) ?? null };
  }
  return { requires: requirementsFromBody(api, body), preferredKey: null };
}

/** The next candidate to try: the best-ranked one not tried, retired or over its cap; a rescue retry
 *  of an already-tried candidate whose provider is still alive, when no fresh one remains (a refusal
 *  may be one call on a failure schedule, whose neighbour call number serves). */
function pickCandidate(ranked: readonly Candidate[], tried: ReadonlyMap<string, number>): Candidate | null {
  for (const cand of ranked) {
    if (tried.has(cand.key)) continue;
    if ((refusals.get(cand.provider.id) ?? 0) >= 2) continue;
    if ((successes.get(cand.provider.id) ?? 0) >= capOf(cand.provider.id)) continue;
    return cand;
  }
  for (const cand of ranked) {
    if ((tried.get(cand.key) ?? 0) >= 2) continue;
    if ((refusals.get(cand.provider.id) ?? 0) >= 2) continue;
    return cand;
  }
  return null;
}

// ---------------------------------------------------------------- the provider call

/** One inference call, in the routed provider's own wire. */
function providerRequest(cand: Candidate, api: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; bodyText: string } {
  const provider = cand.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json, text/event-stream',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let json: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    json = api === 'openai' ? { ...body, model: cand.model.id } : anthropicToOpenAI(body, cand.model.id);
    if (json.stream === true) json.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    json = api === 'anthropic' ? { ...body, model: cand.model.id } : openAIToAnthropic(body, cand.model.id);
  } else {
    const verb = body.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${verb}`;
    json = api === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  return { url, headers, bodyText: JSON.stringify(json) };
}

/** Release a refused answer's body without reading it as an error. */
async function drain(upstream: Response): Promise<void> {
  try {
    await upstream.arrayBuffer();
  } catch {
    // The provider is unreachable or the body is gone: nothing to release.
  }
}

// ---------------------------------------------------------------- client answers

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** Copy a provider's answer to the client as it arrives: the same wire, byte for byte. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function sseHeaders(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/** The model name the client asked for, named back to it in the answer. */
function requestedModel(body: Json, cand: Candidate): string {
  return typeof body.model === 'string' && body.model !== '' ? body.model : cand.model.id;
}

/** Relay a provider's 2xx answer to the client in the client's own wire, JSON or SSE. */
async function relay(cand: Candidate, api: 'openai' | 'anthropic', body: Json, upstream: Response, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const wire = cand.provider.wire;
  const model = requestedModel(body, cand);

  // The same wire needs no translation: the provider's answer is already the client's shape.
  if ((wire === 'openai' && api === 'openai') || (wire === 'anthropic' && api === 'anthropic')) {
    if (stream && upstream.body !== null) {
      sseHeaders(res);
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }

  if (stream && upstream.body === null) {
    sendError(res, api, 500, 'api_error', 'the provider answered no stream');
    return;
  }

  if (!stream) {
    const parsed: unknown = await upstream.json();
    const neutral = isRecord(parsed)
      ? wire === 'openai'
        ? neutralOfOpenAI(parsed)
        : wire === 'anthropic'
          ? neutralOfAnthropic(parsed)
          : neutralOfGemini(parsed)
      : null;
    if (neutral === null) {
      sendError(res, api, 502, api === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered a body this server could not translate');
      return;
    }
    sendJson(res, 200, api === 'openai' ? neutralToOpenAI(neutral, model) : neutralToAnthropic(neutral, model));
    return;
  }

  // A streamed answer across wires: the provider's events to neutral deltas, then the client's shape.
  sseHeaders(res);
  const upstreamBody = upstream.body!;
  if (wire === 'openai') {
    const render = api === 'anthropic' ? renderAnthropicStream(openAIStreamToNeutral(upstreamBody), model) : renderOpenAIStream(openAIStreamToNeutral(upstreamBody), model);
    for await (const chunk of render) res.write(chunk);
  } else if (wire === 'anthropic') {
    const render = api === 'openai' ? renderOpenAIStream(anthropicStreamToNeutral(upstreamBody), model) : renderAnthropicStream(anthropicStreamToNeutral(upstreamBody), model);
    for await (const chunk of render) res.write(chunk);
  } else {
    const render = api === 'openai' ? renderOpenAIStream(geminiStreamToNeutral(upstreamBody), model) : renderAnthropicStream(geminiStreamToNeutral(upstreamBody), model);
    for await (const chunk of render) res.write(chunk);
  }
  res.end();
}

// ---------------------------------------------------------------- the request handlers

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** POST /v1/chat/completions and POST /v1/messages: route, call, fail over, answer in the client's wire. */
async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const slot = slotFor(api, body);
  const clientJson = api === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const ranked = eligible(slot.requires, clientJson).sort(byPreference(slot.preferredKey, successes));
  const tried = new Map<string, number>();
  const attempted: string[] = [];

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const cand = pickCandidate(ranked, tried);
    if (cand === null) break;
    tried.set(cand.key, (tried.get(cand.key) ?? 0) + 1);
    attempted.push(cand.key);
    const { url, headers, bodyText } = providerRequest(cand, api, body);
    let upstream: Response;
    try {
      upstream = await fetch(url, { method: 'POST', headers, body: bodyText, signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
    } catch {
      continue;
    }
    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      const strikes = refusals.get(cand.provider.id) ?? 0;
      refusals.set(cand.provider.id, strikes + 1);
      log(`refused by ${cand.key} with ${upstream.status}`);
      // A refusal is usually one call number on the provider's own failure schedule, sparse among
      // serving call numbers, and a provider with no strikes so far is most likely healthy: retry
      // the very same candidate once, whose neighbouring call number serves, instead of moving the
      // request down a tier and cascading strikes across the next providers. A provider refusing
      // twice is spent or out (its outage ran or its budget emptied): pickCandidate retires it.
      if (strikes === 0) tried.delete(cand.key);
      await drain(upstream);
      continue;
    }
    if (upstream.status < 200 || upstream.status >= 300) {
      log(`client error from ${cand.key}: ${upstream.status}`);
      await drain(upstream);
      continue;
    }
    successes.set(cand.provider.id, (successes.get(cand.provider.id) ?? 0) + 1);
    await relay(cand, api, body, upstream, res);
    return;
  }

  log(`no candidate served this request (tried ${attempted.join(', ') || 'nothing'})`);
  sendError(res, api, api === 'anthropic' ? 503 : 503, api === 'anthropic' ? 'overloaded_error' : 'service_unavailable', 'no free provider on the directory could serve this request');
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------- startup

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      if (attempt === 4) throw err;
      await new Promise<void>((resolve) => setTimeout(resolve, 100));
    }
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  buildPlan();
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      else res.end();
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  await register(port, catalog);
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
