// The brief's three documents and the catalog probe.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) carries three JSON documents, and the prose around
// them does not matter: this file finds each by its `schema` member.
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): one row a provider, with its id,
//     its wire, its auth scheme, the environment variable a real deployment would read the key from, the
//     key the simulated service takes, and the base path the entry joins to MAKEFUN_ENV_URL. A directory
//     entry may give an absolute base_url instead of a path; joinBase takes either.
//   - the quality table (`free-model-aggregator-01/ratings/v1`): each model's public rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and whether it is free. The
//     scorer ranks an answer by its rating; a paid model is never a legal pick. This baseline reads the
//     table's capability columns and free flag (the sealed truth, where a row exists) but never its
//     rating (`--list` shows why in main.ts's header, and the routing TODO there names the gap).
//   - the workload manifest (`free-model-aggregator-01/workload/v1`): the requests the client will send,
//     in order, each with its api, whether it streams, its weight and the requirements a model must meet
//     to serve it.
//
// It then probes each provider's catalog route in the provider's own wire:
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails (a refused connection, a non-2xx, a body that is not JSON) leaves its provider out of
// the catalog and gets no traffic; the trial goes on. Everything here is deterministic: no clock, no
// randomness, so `node main.ts --list` and the register body build the same document (main.ts prints or
// posts this one object).

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

type JsonList = readonly unknown[];
type DocumentPredicate = (document: Json) => boolean;

const WIRES: readonly Wire[] = ['openai', 'anthropic', 'gemini'];
const AUTH_SCHEMES: readonly AuthScheme[] = ['bearer', 'x-api-key', 'x-goog-api-key', 'none'];

function hasString(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

function textOr(value: unknown, fallback: string): string {
  return hasString(value) ? value : fallback;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function listOf(value: unknown): JsonList {
  return Array.isArray(value) ? value : [];
}

function collect<T>(values: JsonList, decode: (value: unknown) => T | null): T[] {
  const result: T[] = [];
  for (const value of values) {
    const decoded = decode(value);
    if (decoded !== null) result.push(decoded);
  }
  return result;
}

function wireOf(value: unknown): Wire | null {
  return typeof value === 'string' && (WIRES as readonly string[]).includes(value) ? value as Wire : null;
}

function authOf(value: unknown): AuthScheme {
  return typeof value === 'string' && (AUTH_SCHEMES as readonly string[]).includes(value) ? value as AuthScheme : 'bearer';
}

/** One provider as the brief lists it. `base` is the base path (or absolute URL) its routes hang off. */
export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the sealed quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as the baseline reads it. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** What a model can do, from the listing first and the quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  const headerName: Partial<Record<AuthScheme, string>> = {
    bearer: 'authorization',
    'x-api-key': 'x-api-key',
    'x-goog-api-key': 'x-goog-api-key',
  };
  const name = headerName[auth];
  if (name === undefined) return {};
  return { [name]: auth === 'bearer' ? `Bearer ${key}` : key };
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function schemaDocuments(text: string): Json[] {
  const documents: Json[] = [];
  let cursor = text.indexOf('{');
  while (cursor !== -1) {
    const end = objectEnd(text, cursor);
    if (end === -1) break;
    const candidate = text.slice(cursor, end + 1);
    try {
      const value: unknown = JSON.parse(candidate);
      if (isRecord(value) && typeof value.schema === 'string') documents.push(value);
    } catch {
      // A brace inside prose or an incomplete object is not a document; continue scanning.
    }
    cursor = text.indexOf('{', cursor + 1);
  }
  return documents;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberOrNull(value: unknown): number | null {
  return finiteNumber(value) ? value : null;
}

function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

type FieldReader<T> = (value: unknown) => T | null;

function firstOf<T>(record: Json, keys: readonly string[], read: FieldReader<T>): T | null {
  for (const key of keys) {
    const value = read(record[key]);
    if (value !== null) return value;
  }
  return null;
}

function firstText(record: Json, keys: readonly string[]): string {
  return firstOf(record, keys, (value) => hasString(value) ? value : null) ?? '';
}

function firstNumber(record: Json, keys: readonly string[]): number | null {
  return firstOf(record, keys, numberOrNull);
}

function providerOf(value: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(value)) return null;
  const id = stringOf(value.id).trim();
  const wire = wireOf(value.wire);
  if (id === '' || wire === null) return null;
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base = firstText(value, ['path', 'base_path', 'base_url', 'base']);
  return {
    id,
    name: textOr(value.name, id),
    wire,
    auth: authOf(value.auth),
    base,
    key_env: stringOf(value.key_env),
    key: stringOf(value.key),
  };
}

function ratingOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = stringOf(value.model);
  const provider = stringOf(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numberOrNull(value.rating_milli) ?? 0,
    context_length: numberOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function requirementOf(value: unknown): Requirement {
  if (!isRecord(value)) return {};
  const minContext = numberOrNull(value.min_context);
  return {
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    ...(minContext === null ? {} : { min_context: minContext }),
  };
}

function workloadRequestOf(value: unknown): WorkloadRequest | null {
  if (!isRecord(value)) return null;
  const id = stringOf(value.id);
  if (id === '') return null;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: requirementOf(value.requires),
  };
}

function documentWith(schema: string, documents: readonly Json[], predicate: DocumentPredicate): Json | null {
  return documents.find((document) => document.schema === schema && predicate(document)) ?? null;
}

function parseDirectory(document: Json): ProviderDirectoryEntry[] {
  return collect(listOf(document.providers), providerOf);
}

function bestRow(value: unknown): { id: string; rating_milli: number } | null {
  if (!isRecord(value) || !hasString(value.id)) return null;
  return { id: value.id, rating_milli: numberOrNull(value.rating_milli) ?? 0 };
}

function parseBest(value: unknown): { id: string; rating_milli: number }[] {
  return collect(listOf(value), bestRow);
}

function parseRatings(document: Json | null): RatingsFile | null {
  if (document === null) return null;
  const ratings = collect(listOf(document.ratings), ratingOf);
  const best = parseBest(document.best);
  return { schema: RATINGS_SCHEMA, ratings, best };
}

function parseWorkload(document: Json | null): WorkloadFile | null {
  if (document === null) return null;
  const manifest = collect(listOf(document.manifest), workloadRequestOf);
  return { schema: WORKLOAD_SCHEMA, manifest };
}

function emptyBrief(error: string): Brief {
  return {
    directory: { schema: PROVIDERS_SCHEMA, providers: [] },
    ratings: null,
    workload: null,
    error,
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  if (path === '') return emptyBrief('MAKEFUN_PROMPT_FILE is not set');
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return emptyBrief(`cannot read ${path}: ${(err as Error).message}`);
  }
  const docs = schemaDocuments(text);
  const directoryDoc = documentWith(PROVIDERS_SCHEMA, docs, (document) => Array.isArray(document.providers));
  if (directoryDoc === null) {
    return emptyBrief(`the brief at ${path} holds no ${PROVIDERS_SCHEMA} document`);
  }
  const ratingsDoc = documentWith(RATINGS_SCHEMA, docs, (document) => Array.isArray(document.ratings));
  const workloadDoc = documentWith(WORKLOAD_SCHEMA, docs, (document) => Array.isArray(document.manifest));
  return {
    directory: { schema: PROVIDERS_SCHEMA, providers: parseDirectory(directoryDoc) },
    ratings: parseRatings(ratingsDoc),
    workload: parseWorkload(workloadDoc),
    error: null,
  };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  for (const row of ratings.ratings) {
    if (row.provider === providerId && row.model === modelId) return row;
  }
  return null;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  const sealedAnswer = row?.free;
  if (sealedAnswer !== undefined) return sealedAnswer;
  return model.free === true;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  const chooseNumber = (sealed: number | null, listed: number | null): number | null => sealed ?? listed;
  const chooseBoolean = (sealed: boolean | null, listed: boolean | null): boolean | null => sealed ?? listed;
  return {
    context_length: chooseNumber(row?.context_length ?? null, model.context_length),
    tools: chooseBoolean(row?.tools ?? null, model.tools),
    vision: chooseBoolean(row?.vision ?? null, model.vision),
    json: chooseBoolean(row?.json ?? null, model.json),
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  const booleanRequirements: Array<[boolean | undefined, boolean | null]> = [
    [requires.tools, caps.tools],
    [requires.vision, caps.vision],
    [requires.json, caps.json],
  ];
  if (booleanRequirements.some(([needed, available]) => needed === true && available !== true)) return false;
  const minimum = requires.min_context;
  return minimum === undefined || (caps.context_length !== null && caps.context_length >= minimum);
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function numericPrice(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function zeroPriced(value: Json): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const prices = Object.values(pricing).map(numericPrice).filter((price): price is number => price !== null);
  if (prices.length === 0) return null;
  return prices.every((price) => price === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, value: unknown): ProbedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? stringOf(value.name).replace(/^models\//, '') : stringOf(value.id);
  if (id === '') return null;
  const context = firstNumber(value, ['context_length', 'context_window', 'inputTokenLimit']);
  const display = firstText(value, ['display_name', 'displayName']) || id;
  return {
    id,
    display_name: display,
    free: boolOrNull(value.free) ?? zeroPriced(value),
    context_length: context,
    tools: boolOrNull(value.tools),
    vision: boolOrNull(value.vision),
    json: boolOrNull(value.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelList(body: Json, wire: Wire): JsonList {
  return listOf(wire === 'gemini' ? body.models : body.data);
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  return collect(modelList(body, wire), (value) => modelIn(wire, value));
}

const CATALOG_PATH: Record<Wire, string> = {
  openai: '/v1/models',
  anthropic: '/v1/models',
  gemini: '/v1beta/models',
};

function catalogPath(wire: Wire): string {
  return CATALOG_PATH[wire];
}

function probeHeaders(provider: ProviderDirectoryEntry): Record<string, string> {
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

function failedProbe(provider: ProviderDirectoryEntry, status: number, error: string): Probe {
  return { provider, ok: false, status, models: [], error };
}

function stableModels(models: readonly ProbedModel[]): ProbedModel[] {
  const unique = new Map<string, ProbedModel>();
  const sorted = [...models].sort((left, right) => left.id.localeCompare(right.id));
  for (const model of sorted) {
    if (unique.has(model.id)) continue;
    unique.set(model.id, model);
  }
  return [...unique.values()];
}

interface ProbeRequest {
  url: string;
  init: RequestInit;
}

function probeRequest(provider: ProviderDirectoryEntry, envUrl: string, timeoutMs: number): ProbeRequest {
  return {
    url: `${joinBase(envUrl, provider.base)}${catalogPath(provider.wire)}`,
    init: {
      headers: probeHeaders(provider),
      signal: AbortSignal.timeout(timeoutMs),
    },
  };
}

function failureMessage(url: string, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return `GET ${url} failed: ${message}`;
}

async function probeResponse(provider: ProviderDirectoryEntry, request: ProbeRequest): Promise<Probe> {
  try {
    const response = await fetch(request.url, request.init);
    if (!response.ok) return failedProbe(provider, response.status, `GET ${request.url} answered HTTP ${response.status}`);
    const body: unknown = await response.json();
    return {
      provider,
      ok: true,
      status: response.status,
      models: stableModels(modelsIn(provider.wire, body)),
      error: null,
    };
  } catch (error) {
    return failedProbe(provider, 0, failureMessage(request.url, error));
  }
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  return probeResponse(provider, probeRequest(provider, envUrl, timeoutMs));
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it listed.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined to `envUrl`: the env's
 * origin differs between a live pass and its replay (a fresh loopback port under the process driver, a
 * fresh pod on the k8s layout), and the catalog is part of the state export a replay must reproduce byte
 * for byte. The scorer compares a reported base_url by its path suffix, so the path passes exactly as the
 * joined URL does; the joined URL reaches a provider, the path names it.
 */
export function buildCatalog(_envUrl: string, probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers = successfulProbes(probes).map((probe) => catalogProvider(probe, ratings));
  return { providers };
}

function catalogModel(providerId: string, model: ProbedModel, ratings: RatingsFile | null): CatalogModel {
  const row = ratingOfModel(ratings, providerId, model.id);
  return {
    id: model.id,
    free: modelFree(model, row),
    context_length: row?.context_length ?? model.context_length ?? null,
  };
}

function catalogProvider(probe: Probe, ratings: RatingsFile | null): CatalogProvider {
  const { provider } = probe;
  return {
    id: provider.id,
    base_url: provider.base,
    key_env: provider.key_env,
    wire: provider.wire,
    models: stableModels(probe.models).map((model) => catalogModel(provider.id, model, ratings)),
  };
}

function successfulProbes(probes: readonly Probe[]): Probe[] {
  return probes.filter((probe) => probe.ok);
}
