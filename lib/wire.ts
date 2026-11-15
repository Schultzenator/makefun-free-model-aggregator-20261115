// The client wires (OpenAI Chat Completions, Anthropic Messages) and the three provider wires
// (openai, anthropic, gemini), translated across each other through one neutral shape.
//
// A client request arrives on POST /v1/chat/completions (OpenAI) or POST /v1/messages (Anthropic). The
// routed provider speaks one of the three wires, so this file carries the six pairwise request
// translations plus the identity pass-throughs, and the answer translations back:
//
//   client api   provider wire   request                     answer
//   openai       openai          model replaced, usage kept  verbatim (JSON or SSE bytes)
//   openai       anthropic       openAIToAnthropic           neutralOfAnthropic -> neutralToOpenAI, or
//                                                            anthropicStreamToNeutral -> renderOpenAIStream
//   openai       gemini          openAIToGemini              neutralOfGemini -> neutralToOpenAI, or
//                                                            geminiStreamToNeutral -> renderOpenAIStream
//   anthropic    anthropic       model replaced               verbatim
//   anthropic    openai          anthropicToOpenAI           neutralOfOpenAI -> neutralToAnthropic, or
//                                                            openAIStreamToNeutral -> renderAnthropicStream
//   anthropic    gemini          anthropicToGemini           neutralOfGemini -> neutralToAnthropic, or
//                                                            geminiStreamToNeutral -> renderAnthropicStream
//
// Echo fidelity is the scoring wire's core. The provider echoes what it parsed of the request we sent
// and the client compares that echo with what it sent, so every translation must preserve: the system
// text (joining its parts with \n as the vendor parsers do), the last user message's text (its text
// parts joined with \n), its image bytes (base64 payload or URL, byte for byte), the tool names in
// order, the exact max_tokens number (or its absence), the stop sequences, and the JSON-mode flag
// (response_format on the openai wire, responseMimeType on the gemini wire; the anthropic wire cannot
// carry it, which the router refuses to depend on: a JSON-mode client request never routes to an
// anthropic-wire provider).
//
// Answers are translated through one neutral shape: the whole text, the stop family (0 stop/end_turn,
// 1 length/max_tokens, 2 tool_calls/tool_use, 3 content_filter/refusal) and the usage counts. Streams
// become one sequence of neutral events: text deltas, then a finish. Rendering then shapes them into
// the client's wire with its event order.
//
// Everything here is a pure function of its arguments: no clock, no randomness, fixed created stamps.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the free models this entry serves, in OpenAI's shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider, free: true })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT, free: true })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------- request-side text helpers

/** The text of a string or of a block/part array's text blocks, joined "\n": the vendors' own join. */
function textJoined(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a block array's text parts joined "" — how responses' content is read back. */
function textConcat(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && (typeof b.text === 'string' ? [b.text] : []))).join('');
}

/** One Anthropic content block as an OpenAI content part: text, or an inline image; null for the rest. */
function blockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as an Anthropic block: text, or an inline image; null for the rest. */
function partToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1]!, data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part: text, inlineData, or fileData; null for the rest. */
function partToGeminiPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1]!, data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text, inlineData, or fileData; null for the rest. */
function blockToGeminiPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textJoined(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool-call id to function-name map, for translating tool results between wires. */
function toolNameOfId(messages: readonly Json[]): Map<string, string> {
  const out = new Map<string, string>();
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && typeof call.id === 'string') {
        out.set(call.id, call.function.name);
      }
    }
  }
  return out;
}

/** An anthropic assistant tool_use id/name map, for translating tool results between wires. */
function anthropicToolNameOfId(messages: readonly Json[]): Map<string, string> {
  const out = new Map<string, string>();
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content) {
      if (isRecord(block) && block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') out.set(block.id, block.name);
    }
  }
  return out;
}

// ---------------------------------------------------------------- request normalization and translations

function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function numericField(body: Json, ...names: string[]): number | undefined {
  for (const name of names) if (typeof body[name] === 'number') return body[name] as number;
  return undefined;
}

function copyNumber(out: Json, target: string, body: Json, ...names: string[]): void {
  const value = numericField(body, ...names);
  if (value !== undefined) out[target] = value;
}

function copyArray(out: Json, target: string, body: Json, source: string, limit?: number): void {
  if (!Array.isArray(body[source]) || body[source].length === 0) return;
  out[target] = limit === undefined ? body[source] : body[source].slice(0, limit);
}

function openAIToolDeclarations(body: Json): Json[] {
  return records(body.tools).flatMap((raw) => {
    if (!isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    }];
  });
}

function anthropicToolDeclarations(body: Json): Json[] {
  return records(body.tools).flatMap((raw) => {
    if (typeof raw.name !== 'string') return [];
    return [{
      type: 'function',
      function: {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    }];
  });
}

function geminiTools(declarations: Json[]): Json[] {
  return declarations.map((declaration) => {
    const fn = isRecord(declaration.function) ? declaration.function : declaration;
    return {
      name: fn.name,
      description: fn.description,
      parameters: fn.parameters,
    };
  });
}

function withSampling(out: Json, body: Json, stopName: string, stopLimit?: number): void {
  copyNumber(out, 'temperature', body, 'temperature');
  copyNumber(out, 'top_p', body, 'top_p');
  copyArray(out, stopName, body, 'stop', stopLimit);
}

function anthropicChoice(choice: unknown): Json | string | undefined {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

function openAIChoice(choice: unknown): Json | undefined {
  if (choice === 'auto' || choice === 'none') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return undefined;
}

function openAIContentToAnthropic(content: unknown): Json[] {
  return records(content).map(partToBlock).filter((part): part is Json => part !== null);
}

function openAIContentToGemini(content: unknown): Json[] {
  return records(content).map(partToGeminiPart).filter((part): part is Json => part !== null);
}

function anthropicContentToGemini(content: unknown): Json[] {
  return records(content).flatMap((block) => {
    if (block.type === 'tool_use') {
      return [{ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } }];
    }
    if (block.type === 'tool_result') return [];
    const part = blockToGeminiPart(block);
    return part === null ? [] : [part];
  });
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. A response_format is
 * never added: the anthropic wire carries no JSON mode, so the router must not need one here.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textJoined(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of records(body.messages)) {
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls = records(content).flatMap((block) => block.type !== 'tool_use' ? [] : [{
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        }]);
      const text = textJoined(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts = records(content).flatMap((block) => {
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        return [];
      }
      const part = blockToPart(block);
      return part === null ? [] : [part];
    });
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  copyNumber(out, 'max_tokens', body, 'max_tokens');
  withSampling(out, body, 'stop');
  copyArray(out, 'stop', body, 'stop_sequences');
  const tools = anthropicToolDeclarations(body);
  if (tools.length > 0) out.tools = tools;
  const choice = anthropicChoice(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages become one system prompt, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, and the rest passes through. The anthropic wire
 * cannot carry a JSON mode: the router never sends a JSON-mode request this way.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const source = records(body.messages);
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const toolNames = toolNameOfId(source);
  for (const raw of source) {
    if (raw.role === 'system') {
      const text = textJoined(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks = openAIContentToAnthropic(content);
      for (const call of records(raw.tool_calls)) {
        if (!isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${messages.length}`,
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    const blocks = openAIContentToAnthropic(content);
    messages.push({ role: 'user', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  copyNumber(out, 'max_tokens', body, 'max_tokens', 'max_completion_tokens');
  withSampling(out, body, 'stop_sequences', 4);
  const tools = openAIToolDeclarations(body).map((tool) => {
    const fn = isRecord(tool.function) ? tool.function : {};
    return { name: fn.name, description: fn.description, input_schema: fn.parameters };
  });
  if (tools.length > 0) out.tools = tools;
  const choice = openAIChoice(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice;
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: system messages become
 * systemInstruction, message parts become contents (images inline, tool results functionResponse
 * parts), tools become functionDeclarations, and generationConfig carries the output cap, stop
 * sequences, temperature/top_p and responseMimeType for JSON mode.
 */
export function openAIToGemini(body: Json): Json {
  const source = records(body.messages);
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = toolNameOfId(source);
  for (const raw of source) {
    if (raw.role === 'system') {
      const text = textJoined(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const model = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: model ? 'model' : 'user', parts: content === '' ? [] : [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts = openAIContentToGemini(content);
    if (raw.role === 'tool') {
      const name = typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) ?? '' : '';
      parts.push({ functionResponse: { name: name === '' ? 'tool' : name, response: { result: toolResultText(raw.content) } } });
    }
    contents.push({ role: model ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = openAIToolDeclarations(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: geminiTools(declarations) }];
  const config: Json = {};
  const maxTokens = numericField(body, 'max_tokens', 'max_completion_tokens');
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  copyArray(config, 'stopSequences', body, 'stop', 5);
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt becomes
 * systemInstruction, image blocks become inlineData/fileData parts, tool_use blocks become
 * functionCall parts, tool_result blocks functionResponse parts, and generationConfig carries the
 * output cap, stop sequences, temperature/top_p.
 */
export function anthropicToGemini(body: Json): Json {
  const system = textJoined(body.system);
  const source = records(body.messages);
  const contents: Json[] = [];
  const toolNames = anthropicToolNameOfId(source);
  for (const raw of source) {
    const model = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: model ? 'model' : 'user', parts: content === '' ? [] : [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts = anthropicContentToGemini(content);
    for (const block of records(content)) {
      if (block.type === 'tool_result') {
        const name = typeof block.tool_use_id === 'string' ? toolNames.get(block.tool_use_id) ?? '' : '';
        parts.push({ functionResponse: { name: name === '' ? 'tool' : name, response: { result: toolResultText(block.content) } } });
      }
    }
    contents.push({ role: model ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolDeclarations(body).map((tool) => {
    const fn = isRecord(tool.function) ? tool.function : {};
    return { name: fn.name, description: fn.description, parameters: fn.parameters };
  });
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = numericField(body, 'max_tokens');
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  copyArray(config, 'stopSequences', body, 'stop_sequences', 5);
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------- the neutral answer

/** Stop families: 0 = clean stop, 1 = the output cap, 2 = a tool call, 3 = refusal/filter. */
export type StopFamily = 0 | 1 | 2 | 3;

/** Every stop condition a provider answers, in its own wire's spelling, as one family. */
export function stopFamilyOf(reason: unknown): StopFamily {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  return 3;
}

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'] as const;
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'] as const;

/** A provider's answer as one neutral value: the whole text, the stop family, and the usage counts. */
export interface NeutralAnswer {
  text: string;
  stop: StopFamily;
  usage: { input: number; output: number };
}

type Usage = { input: number; output: number };

function firstRecord(value: unknown): Json | null {
  return Array.isArray(value) && isRecord(value[0]) ? value[0] : null;
}

function count(value: unknown): number {
  return typeof value === 'number' ? value : 0;
}

function usageFrom(body: Json, inputKey: string, outputKey: string): Usage {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { input: count(usage[inputKey]), output: count(usage[outputKey]) };
}

function neutral(text: string, reason: unknown, usage: Usage): NeutralAnswer {
  return { text, stop: stopFamilyOf(reason), usage };
}

/** A Chat Completions JSON answer as a neutral answer. */
export function neutralOfOpenAI(body: Json): NeutralAnswer | null {
  const choice = firstRecord(body.choices);
  const message = isRecord(choice?.message) ? choice?.message : null;
  if (choice === null || message === null) return null;
  return neutral(typeof message.content === 'string' ? message.content : textConcat(message.content), choice.finish_reason, usageFrom(body, 'prompt_tokens', 'completion_tokens'));
}

/** An Anthropic Messages JSON answer as a neutral answer. */
export function neutralOfAnthropic(body: Json): NeutralAnswer | null {
  if (!Array.isArray(body.content)) return null;
  return neutral(textConcat(body.content), body.stop_reason, usageFrom(body, 'input_tokens', 'output_tokens'));
}

/** A Gemini generateContent JSON answer as a neutral answer. */
export function neutralOfGemini(body: Json): NeutralAnswer | null {
  const candidate = firstRecord(body.candidates);
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return neutral(geminiPartsText(content.parts), candidate.finishReason, {
    input: count(usage.promptTokenCount),
    output: count(usage.candidatesTokenCount),
  });
}

/** A Gemini parts array's text, joined "": how its own content reads back. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A neutral answer as an OpenAI Chat Completions object. */
export function neutralToOpenAI(answer: NeutralAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: OPENAI_STOPS[answer.stop],
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** A neutral answer as an Anthropic Messages object. */
export function neutralToAnthropic(answer: NeutralAnswer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: ANTHROPIC_STOPS[answer.stop],
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

// ---------------------------------------------------------------- streaming

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One neutral stream event: a text delta, or the finish with the stop family and the usage. */
type NeutralEvent = { kind: 'text'; text: string } | { kind: 'finish'; stop: StopFamily; usage: { input: number; output: number } };

/** Reads one SSE byte stream to the end, yielding each frame's data lines as they arrive. */
async function* sseDataLines(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (line.startsWith('data:')) yield line.slice(5).replace(/^ /, '');
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    for (const line of buffer.split('\n')) {
      if (line.startsWith('data:')) yield line.slice(5).replace(/^ /, '');
    }
  } finally {
    reader.releaseLock();
  }
}

/** Parses one SSE data line as a JSON object, or null. */
function frameJson(line: string): Json | null {
  if (line.trim() === '') return null;
  try {
    const value: unknown = JSON.parse(line);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function addText(events: NeutralEvent[], value: unknown): void {
  if (typeof value === 'string' && value !== '') events.push({ kind: 'text', text: value });
}

function finishEvent(stop: StopFamily, usage: Usage): NeutralEvent {
  return { kind: 'finish', stop, usage };
}

/** A Chat Completions SSE stream as neutral events: the text deltas, then the finish with the usage. */
export async function* openAIStreamToNeutral(upstream: ReadableStream<Uint8Array>): AsyncGenerator<NeutralEvent> {
  let stopFamily: StopFamily = 0;
  let usage: Usage | null = null;
  const pending: NeutralEvent[] = [];
  for await (const line of sseDataLines(upstream)) {
    if (line.trim() === '[DONE]') break;
    const chunk = frameJson(line);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: count(chunk.usage.prompt_tokens), output: count(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    addText(pending, delta.content);
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  for (const event of pending) yield event;
  yield finishEvent(stopFamily, usage ?? { input: 0, output: 0 });
}

/** An Anthropic Messages SSE stream as neutral events: the text deltas, then the finish with the usage. */
export async function* anthropicStreamToNeutral(upstream: ReadableStream<Uint8Array>): AsyncGenerator<NeutralEvent> {
  const pending: NeutralEvent[] = [];
  let stopFamily: StopFamily = 0;
  let input = 0;
  let output = 0;
  let finished = false;
  for await (const line of sseDataLines(upstream)) {
    const event = frameJson(line);
    if (event === null) continue;
    if (isRecord(event.message) && isRecord(event.message.usage)) {
      const inTokens = event.message.usage.input_tokens;
      if (typeof inTokens === 'number') input = inTokens;
    }
    if (isRecord(event.delta)) {
      addText(pending, event.delta.text);
      if (typeof event.delta.stop_reason === 'string') stopFamily = stopFamilyOf(event.delta.stop_reason);
    }
    if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') output = event.usage.output_tokens;
    if (isRecord(event) && event.type === 'message_stop') finished = true;
  }
  for (const event of pending) yield event;
  yield finishEvent(finished ? stopFamily : 0, { input, output });
}

/** A Gemini SSE stream as neutral events: the text deltas, then the finish with the usage. */
export async function* geminiStreamToNeutral(upstream: ReadableStream<Uint8Array>): AsyncGenerator<NeutralEvent> {
  const pending: NeutralEvent[] = [];
  let stopFamily: StopFamily = 0;
  let usage: Usage = { input: 0, output: 0 };
  for await (const line of sseDataLines(upstream)) {
    const frame = frameJson(line);
    if (frame === null) continue;
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (isRecord(frame.usageMetadata)) {
      usage = { input: count(frame.usageMetadata.promptTokenCount), output: count(frame.usageMetadata.candidatesTokenCount) };
    }
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    if (typeof content.parts !== 'undefined' && content.parts !== null) {
      // One Gemini frame may hold several parts: each text part is its own delta.
      for (const part of records(content.parts)) addText(pending, part.text);
    }
    if (typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
  }
  for (const event of pending) yield event;
  yield finishEvent(stopFamily, usage);
}

/** Neutral stream events as an OpenAI SSE stream: chunks, one finish chunk, the usage chunk, [DONE]. */
export async function* renderOpenAIStream(events: AsyncGenerator<NeutralEvent>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  yield chunk({ role: 'assistant', content: '' }, null);
  let finish = 0 as StopFamily;
  let usage = { input: 0, output: 0 };
  for (;;) {
    const next = await events.next();
    if (next.done) break;
    const event = next.value;
    if (event.kind === 'text') yield chunk({ content: event.text }, null);
    else {
      finish = event.stop;
      usage = event.usage;
    }
  }
  yield chunk({}, OPENAI_STOPS[finish]);
  yield sseEvent(null, {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  });
  yield 'data: [DONE]\n\n';
}

/** Neutral stream events as an Anthropic SSE stream: message_start, the block's events, the close. */
export async function* renderAnthropicStream(events: AsyncGenerator<NeutralEvent>, model: string): AsyncGenerator<string> {
  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });
  let blockOpen = false;
  let finish = 0 as StopFamily;
  let usage = { input: 0, output: 0 };
  for (;;) {
    const next = await events.next();
    if (next.done) break;
    const event = next.value;
    if (event.kind === 'text') {
      if (!blockOpen) {
        blockOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: event.text } });
    } else {
      finish = event.stop;
      usage = event.usage;
    }
  }
  if (blockOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  else {
    yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  }
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: ANTHROPIC_STOPS[finish], stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
