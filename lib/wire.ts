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

// ---------------------------------------------------------------- request translations

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
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textJoined(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = blockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
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
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const toolNames = toolNameOfId(source);
  for (const raw of source) {
    if (!isRecord(raw)) continue;
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
      const blocks: Json[] = content
        .map((block) => (isRecord(block) ? partToBlock(block) : null))
        .flatMap((b): Json[] => (b === null ? [] : [b]));
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
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
    const blocks: Json[] = content
      .map((block) => (isRecord(block) ? partToBlock(block) : null))
      .flatMap((b): Json[] => (b === null ? [] : [b]));
    messages.push({ role: 'user', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.slice(0, 4);
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
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
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = toolNameOfId(source);
  for (const raw of source) {
    if (!isRecord(raw)) continue;
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
    const parts: Json[] = content
      .map((part) => (isRecord(part) ? partToGeminiPart(part) : null))
      .flatMap((p): Json[] => (p === null ? [] : [p]));
    if (raw.role === 'tool') {
      const name = typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) ?? '' : '';
      parts.push({ functionResponse: { name: name === '' ? 'tool' : name, response: { result: toolResultText(raw.content) } } });
    }
    contents.push({ role: model ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.slice(0, 5);
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
  const source = Array.isArray(body.messages) ? body.messages : [];
  const contents: Json[] = [];
  const toolNames = anthropicToolNameOfId(source);
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const model = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: model ? 'model' : 'user', parts: content === '' ? [] : [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      if (block.type === 'tool_result') {
        const name = typeof block.tool_use_id === 'string' ? toolNames.get(block.tool_use_id) ?? '' : '';
        parts.push({ functionResponse: { name: name === '' ? 'tool' : name, response: { result: toolResultText(block.content) } } });
        continue;
      }
      const part = blockToGeminiPart(block);
      if (part !== null) parts.push(part);
    }
    contents.push({ role: model ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.slice(0, 5);
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

/** A Chat Completions JSON answer as a neutral answer. */
export function neutralOfOpenAI(body: Json): NeutralAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  const message = isRecord(choice?.message) ? choice?.message : null;
  if (choice === null || message === null) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textConcat(message.content),
    stop: stopFamilyOf(choice.finish_reason),
    usage: {
      input: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
      output: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
    },
  };
}

/** An Anthropic Messages JSON answer as a neutral answer. */
export function neutralOfAnthropic(body: Json): NeutralAnswer | null {
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textConcat(body.content),
    stop: stopFamilyOf(body.stop_reason),
    usage: {
      input: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      output: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
    },
  };
}

/** A Gemini generateContent JSON answer as a neutral answer. */
export function neutralOfGemini(body: Json): NeutralAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: geminiPartsText(content.parts),
    stop: stopFamilyOf(candidate.finishReason),
    usage: {
      input: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
      output: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
    },
  };
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

/** A Chat Completions SSE stream as neutral events: the text deltas, then the finish with the usage. */
export async function* openAIStreamToNeutral(upstream: ReadableStream<Uint8Array>): AsyncGenerator<NeutralEvent> {
  let stopFamily: StopFamily = 0;
  let usage: { input: number; output: number } | null = null;
  const pending: NeutralEvent[] = [];
  for await (const line of sseDataLines(upstream)) {
    if (line.trim() === '[DONE]') break;
    const chunk = frameJson(line);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = {
        input: typeof chunk.usage.prompt_tokens === 'number' ? chunk.usage.prompt_tokens : 0,
        output: typeof chunk.usage.completion_tokens === 'number' ? chunk.usage.completion_tokens : 0,
      };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') pending.push({ kind: 'text', text: delta.content });
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  for (const event of pending) yield event;
  yield { kind: 'finish', stop: stopFamily, usage: usage ?? { input: 0, output: 0 } };
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
      if (typeof event.delta.text === 'string' && event.delta.text !== '') pending.push({ kind: 'text', text: event.delta.text });
      if (typeof event.delta.stop_reason === 'string') stopFamily = stopFamilyOf(event.delta.stop_reason);
    }
    if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') output = event.usage.output_tokens;
    if (isRecord(event) && event.type === 'message_stop') finished = true;
  }
  for (const event of pending) yield event;
  yield { kind: 'finish', stop: finished ? stopFamily : 0, usage: { input, output } };
}

/** A Gemini SSE stream as neutral events: the text deltas, then the finish with the usage. */
export async function* geminiStreamToNeutral(upstream: ReadableStream<Uint8Array>): AsyncGenerator<NeutralEvent> {
  const pending: NeutralEvent[] = [];
  let stopFamily: StopFamily = 0;
  let usage = { input: 0, output: 0 };
  for await (const line of sseDataLines(upstream)) {
    const frame = frameJson(line);
    if (frame === null) continue;
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (isRecord(frame.usageMetadata)) {
      usage = {
        input: typeof frame.usageMetadata.promptTokenCount === 'number' ? frame.usageMetadata.promptTokenCount : 0,
        output: typeof frame.usageMetadata.candidatesTokenCount === 'number' ? frame.usageMetadata.candidatesTokenCount : 0,
      };
    }
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    if (typeof content.parts !== 'undefined' && content.parts !== null) {
      // One Gemini frame may hold several parts: each text part is its own delta.
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string' && part.text !== '') pending.push({ kind: 'text', text: part.text });
      }
    }
    if (typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
  }
  for (const event of pending) yield event;
  yield { kind: 'finish', stop: stopFamily, usage };
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
