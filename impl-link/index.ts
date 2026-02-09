import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

export type FormalWork = {
  formalWorkDir: string;
  tla2toolsJar: string;
  javaBin: string;
};

function getJavaBin(formalWorkDir: string) {
  const candidates = [
    path.join(
      formalWorkDir,
      'jdk',
      'Contents',
      'Home',
      'bin',
      process.platform === 'win32' ? 'java.exe' : 'java'
    ),
    path.join(formalWorkDir, 'jdk', 'bin', process.platform === 'win32' ? 'java.exe' : 'java'),
  ];
  return candidates.find((p) => fs.existsSync(p)) || 'java';
}

/** Locate the FormalAnswer toolchain (the `work/` folder inside the FormalAnswer repo). */
export function findFormalWork(formalanswerRoot: string): FormalWork | null {
  const formalWorkDir = path.join(formalanswerRoot, 'work');
  const tla2toolsJar = path.join(formalWorkDir, 'tla2tools.jar');
  if (!fs.existsSync(tla2toolsJar)) return null;
  return { formalWorkDir, tla2toolsJar, javaBin: getJavaBin(formalWorkDir) };
}

export function runTlc(
  formalWork: FormalWork,
  opts: { tlaFile: string; cfgFile?: string; cwd?: string; args?: string[] }
): { stdout: string; stderr: string } {
  const args = ['-jar', formalWork.tla2toolsJar];
  if (opts.cfgFile) args.push('-config', opts.cfgFile);
  if (opts.args?.length) args.push(...opts.args);
  args.push(opts.tlaFile);

  const res = spawnSync(formalWork.javaBin, args, {
    cwd: opts.cwd,
    encoding: 'utf8',
  });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    throw new Error(`TLC failed (status ${res.status}):\n${res.stdout}\n${res.stderr}`);
  }
  return { stdout: String(res.stdout || ''), stderr: String(res.stderr || '') };
}

/**
 * Extracts a TLC-printed TLA+ set from a single output line containing `${key}=`.
 * Returns sorted, unquoted string elements.
 */
export function extractPrintedTlaStringSet(stdout: string, key: string): string[] {
  const s = String(stdout || '');
  const line = s.split(/\r?\n/).find((l) => l.includes(`${key}=`));
  if (!line) throw new Error(`missing ${key}= line in TLC output`);

  const setMatch = line.match(/\{[\s\S]*\}/);
  if (!setMatch) throw new Error(`missing set literal in: ${line}`);

  const raw = setMatch[0].trim();
  if (raw === '{}' || raw === '{ }') return [];

  const m = raw.match(/^\{([\s\S]*)\}$/);
  if (!m) throw new Error(`unexpected set format: ${raw}`);

  return m[1]
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => x.replace(/^"|"$/g, ''))
    .sort();
}

/** Write a fixture directory tree from `relPath -> fileContent`. */
export function writeFixtureTree(root: string, files: Record<string, string>) {
  for (const [rel, content] of Object.entries(files)) {
    const abs = path.join(root, rel);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    fs.writeFileSync(abs, content);
  }
}

/** Run a node script that prints JSON on stdout and parse it. */
export function runNodeJson(
  nodeArgs: string[],
  opts?: { cwd?: string }
): { json: any; stdout: string; stderr: string; status: number } {
  const res = spawnSync(process.execPath, nodeArgs, {
    cwd: opts?.cwd,
    encoding: 'utf8',
  });
  if (res.error) throw res.error;
  const stdout = String(res.stdout || '');
  const stderr = String(res.stderr || '');

  let json: any;
  try {
    json = JSON.parse(stdout || '{}');
  } catch {
    throw new Error(`failed to parse JSON stdout:\n${stdout}\n--- stderr ---\n${stderr}`);
  }

  return { json, stdout, stderr, status: res.status ?? 0 };
}

export function diffStringSets(expected: string[], actual: string[]) {
  const e = new Set(expected);
  const a = new Set(actual);
  return {
    missing: expected.filter((x) => !a.has(x)),
    extra: actual.filter((x) => !e.has(x)),
  };
}
