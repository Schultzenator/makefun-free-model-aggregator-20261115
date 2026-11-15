// The local practice harness: runs one practice trial exactly as the harness does, then scores it.
//
//   node scripts/practice.mjs practice-reference/trial-0
//   node scripts/practice.mjs practice-reference/trial-0 --replay   # twice, export must match byte for byte
//   node scripts/practice.mjs --list                                  # --list inside and outside a trial
//
// One trial: the sealed env (makefun-program/env/server.mjs) seeds itself from the instance's
// input/env (providers.json, workload.json), the entry (main.ts) registers with it, the env's client
// sends the 20 requests, the entry exits 0 when /work says done, the env is SIGTERMed and exports its
// state, and the sealed scorer (makefun-program/scorer/score.mjs) scores the export against the
// instance's reference/. Nothing here contacts any network but the loopback.

import { spawn, execSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, join, resolve } from 'node:path';

const ROOT = resolve(dirname(new URL(import.meta.url).pathname), '..');

function freePort() {
  return new Promise((ok, fail) => {
    const server = createServer();
    server.unref();
    server.once('error', fail);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => ok(port));
    });
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function waitFile(file, timeoutMs) {
  const end = Date.now() + timeoutMs;
  return new Promise((ok, fail) => {
    const tick = () => {
      if (existsSync(file) && statSync(file).size > 0) return ok();
      if (Date.now() > end) return fail(new Error(`waited for ${file}`));
      setTimeout(tick, 50);
    };
    tick();
  });
}

function child(name, command, args, env) {
  const proc = spawn(command, args, { env: { ...process.env, ...env }, cwd: ROOT, stdio: ['ignore', 'inherit', 'inherit'] });
  let settled = false;
  const done = new Promise((ok, fail) => {
    proc.once('exit', (code, signal) => {
      settled = true;
      code === 0 ? ok(code) : fail(new Error(`${name} exited ${code ?? signal}`));
    });
    proc.once('error', fail);
  });
  return { done, kill: () => { if (!settled) { try { proc.kill('SIGTERM'); } catch {} } } };
}

async function runOnce(trialDir, stateDir) {
  rmSync(stateDir, { recursive: true, force: true });
  mkdirSync(stateDir, { recursive: true });
  const envPort = await freePort();
  const envUrl = `http://127.0.0.1:${envPort}`;
  const env = child('env', process.execPath, ['makefun-program/env/server.mjs'], {
    MAKEFUN_ENV_DATA_DIR: join(trialDir, 'input/env'),
    MAKEFUN_STATE_DIR: stateDir,
    MAKEFUN_ENV_HOST: '127.0.0.1',
    MAKEFUN_ENV_PORT: String(envPort),
  });
  try {
    await waitFile(join(stateDir, 'ready'), 15000);
    const agent = child('main.ts', process.execPath, ['main.ts'], {
      MAKEFUN_ENV_URL: envUrl,
      MAKEFUN_PROMPT_FILE: join(trialDir, 'input/prompt.txt'),
      MAKEFUN_TRIAL_SEED: 'deadbeef'.repeat(8),
      MAKEFUN_TRIAL_NOW: '2026-11-15T00:00:00Z',
    });
    try {
      await agent.done;
    } finally {
      agent.kill();
    }
    env.kill();
    await Promise.race([env.done, sleep(8000).then(() => { throw new Error('env did not exit'); })]);
    await waitFile(join(stateDir, 'export', 'state.json'), 15000);
    return readFileSync(join(stateDir, 'export', 'state.json'), 'utf8');
  } finally {
    env.kill();
  }
}

async function score(trialDir, stateDir) {
  const scorer = spawn(process.execPath, ['makefun-program/scorer/score.mjs'], {
    cwd: ROOT,
    env: {
      ...process.env,
      MAKEFUN_OUTPUT_DIR: join(stateDir, 'export'),
      MAKEFUN_REFERENCE_DIR: join(trialDir, 'reference'),
      MAKEFUN_SCORES_DIR: join(stateDir, 'scores'),
    },
    stdio: ['ignore', 'inherit', 'pipe'],
  });
  let stderr = '';
  scorer.stderr.on('data', (d) => { stderr += d.toString(); });
  const code = await new Promise((r) => scorer.once('exit', (c) => r(c ?? 1)));
  if (code !== 0) throw new Error(`the scorer failed: ${stderr}`);
  return JSON.parse(readFileSync(join(stateDir, 'scores', 'scores.json'), 'utf8'));
}

const args = process.argv.slice(2);
if (args[0] === '--list') {
  process.stdout.write('# outside a trial (no MAKEFUN_* variables set):\n');
  process.stdout.write(execSync('env -u MAKEFUN_ENV_URL -u MAKEFUN_PROMPT_FILE node main.ts --list').toString());
  process.stdout.write('# inside trial-0 (catalog probed from the env):\n');
  const trialDir = resolve(ROOT, 'practice-reference/trial-0');
  const stateDir = join(ROOT, '.practice', 'state-list');
  mkdirSync(stateDir, { recursive: true });
  const envPort = await freePort();
  const env = spawn(process.execPath, ['makefun-program/env/server.mjs'], {
    cwd: ROOT,
    env: {
      ...process.env,
      MAKEFUN_ENV_DATA_DIR: join(trialDir, 'input/env'),
      MAKEFUN_STATE_DIR: stateDir,
      MAKEFUN_ENV_HOST: '127.0.0.1',
      MAKEFUN_ENV_PORT: String(envPort),
    },
    stdio: 'ignore',
  });
  await waitFile(join(stateDir, 'ready'), 15000);
  try {
    const out = execSync('node main.ts --list', {
      cwd: ROOT,
      env: {
        ...process.env,
        MAKEFUN_ENV_URL: `http://127.0.0.1:${envPort}`,
        MAKEFUN_PROMPT_FILE: join(trialDir, 'input/prompt.txt'),
        MAKEFUN_TRIAL_SEED: '0'.repeat(64),
      },
    });
    process.stdout.write(out.toString());
  } finally {
    env.kill();
  }
  process.exit(0);
}

const trialDir = resolve(ROOT, args[0] ?? 'practice-reference/trial-0');
const replay = args.includes('--replay');
const stateDir = join(ROOT, '.practice', trialDir.split('/').pop());

const first = await runOnce(trialDir, join(stateDir, 'run-1'));
const scores = await score(trialDir, join(stateDir, 'run-1'));
process.stdout.write(`scores ${JSON.stringify(scores)}\n`);
if (replay) {
  const second = await runOnce(trialDir, join(stateDir, 'run-2'));
  process.stdout.write(`export bytes: first ${Buffer.byteLength(first)}, replay ${Buffer.byteLength(second)}, ${first === second ? 'MATCH' : 'MISMATCH'}\n`);
  if (first !== second) process.exit(1);
}
