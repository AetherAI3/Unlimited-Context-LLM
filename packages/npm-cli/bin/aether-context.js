#!/usr/bin/env node
// aether-context (Unlimited Context)
// Copyright (c) 2026 Aether AI
// SPDX-License-Identifier: Apache-2.0
//
// npm launcher for the `aether-context` Python package.
//
// The engine is Python (numpy-only core), so this package does not reimplement it — it makes
// `npx aether-context` work for people whose muscle memory is npm. On first run it builds a
// private virtualenv under the OS cache directory, installs the matching `aether-context`
// release from PyPI into it, and from then on forwards straight through. Nothing is written to
// the user's global site-packages and nothing needs sudo.
//
// Zero dependencies on purpose: a launcher that has to resolve a dependency tree before it can
// tell you that Python is missing is a worse launcher.

"use strict";

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const PACKAGE_VERSION = require("../package.json").version;
/** Minimum interpreter the Python package declares (`requires-python = ">=3.10"`). */
const MIN_PYTHON = [3, 10];

/**
 * The `aether-context` release to install.
 *
 * Pinned to this launcher's own version: the two are cut from one commit in one repo, and a
 * release-parity test fails the build if they ever drift. `AETHER_CONTEXT_VERSION` overrides it
 * (use `latest` for the newest release) for anyone testing a pre-release.
 */
function targetVersion() {
  const override = process.env.AETHER_CONTEXT_VERSION;
  if (!override) return PACKAGE_VERSION;
  return override.toLowerCase() === "latest" ? null : override;
}

function requirement() {
  const version = targetVersion();
  return version === null ? "aether-context" : `aether-context==${version}`;
}

// --- interpreter discovery -----------------------------------------------------------------

/** Candidate interpreter commands, best first. `AETHER_CONTEXT_PYTHON` short-circuits the search. */
function pythonCandidates() {
  const explicit = process.env.AETHER_CONTEXT_PYTHON;
  if (explicit) return [[explicit, []]];
  const candidates = [
    ["python3", []],
    ["python", []],
  ];
  // The Windows launcher resolves a real interpreter even when `python` is the Store alias stub.
  if (process.platform === "win32") candidates.unshift(["py", ["-3"]]);
  return candidates;
}

/** `[major, minor]` for an interpreter, or null if it cannot run / is not a real Python. */
function pythonVersion(command, args) {
  const probe = spawnSync(
    command,
    [...args, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
    { encoding: "utf8", windowsHide: true },
  );
  if (probe.status !== 0 || !probe.stdout) return null;
  const match = /^(\d+)\.(\d+)/.exec(probe.stdout.trim());
  return match ? [Number(match[1]), Number(match[2])] : null;
}

function meetsMinimum(version) {
  return (
    version[0] > MIN_PYTHON[0] ||
    (version[0] === MIN_PYTHON[0] && version[1] >= MIN_PYTHON[1])
  );
}

/**
 * The first interpreter on PATH new enough to run the package.
 *
 * Reports the two failure modes separately, because the fix differs: no Python at all versus a
 * Python that is too old. On Windows the App Execution Alias makes `python` exist while failing
 * to run, which the probe treats as "not a real Python" rather than crashing.
 */
function findPython() {
  let bestFound = null;
  for (const [command, args] of pythonCandidates()) {
    const version = pythonVersion(command, args);
    if (!version) continue;
    if (meetsMinimum(version)) return { command, args, version };
    if (!bestFound || version > bestFound.version) bestFound = { command, args, version };
  }
  if (bestFound) {
    fail(
      `found Python ${bestFound.version.join(".")}, but aether-context needs ` +
        `${MIN_PYTHON.join(".")} or newer.`,
      "install a newer Python from https://python.org, or point AETHER_CONTEXT_PYTHON at one",
    );
  }
  fail(
    "no Python interpreter found on PATH.",
    "install Python " +
      MIN_PYTHON.join(".") +
      "+ from https://python.org, or set AETHER_CONTEXT_PYTHON=/path/to/python",
  );
}

// --- managed environment --------------------------------------------------------------------

/** Where the private virtualenv lives — per-user cache, never the repo or global site-packages. */
function venvRoot() {
  if (process.env.AETHER_CONTEXT_HOME) return process.env.AETHER_CONTEXT_HOME;
  const home = os.homedir();
  if (process.platform === "win32") {
    return path.join(process.env.LOCALAPPDATA || path.join(home, "AppData", "Local"), "aether-context", "npm-venv");
  }
  if (process.platform === "darwin") {
    return path.join(home, "Library", "Caches", "aether-context", "npm-venv");
  }
  const base = process.env.XDG_CACHE_HOME || path.join(home, ".cache");
  return path.join(base, "aether-context", "npm-venv");
}

function venvPython(root) {
  return process.platform === "win32"
    ? path.join(root, "Scripts", "python.exe")
    : path.join(root, "bin", "python");
}

/** The installed `aether_context.__version__` inside the venv, or null if it is not importable. */
function installedVersion(python) {
  if (!fs.existsSync(python)) return null;
  const probe = spawnSync(
    python,
    ["-c", "import aether_context; print(aether_context.__version__)"],
    { encoding: "utf8", windowsHide: true },
  );
  return probe.status === 0 && probe.stdout ? probe.stdout.trim() : null;
}

function run(command, args, label) {
  const result = spawnSync(command, args, { stdio: "inherit", windowsHide: true });
  if (result.error) fail(`${label} failed to start: ${result.error.message}`);
  if (result.status !== 0) fail(`${label} failed (exit ${result.status}).`);
}

/**
 * Ensure the managed venv exists with the right release installed, and return its interpreter.
 *
 * The fast path — already installed at the pinned version — does one short probe and no network
 * access at all, so the launcher does not add a package-manager round trip to every invocation.
 */
function ensureEnvironment() {
  const root = venvRoot();
  const python = venvPython(root);
  const wanted = targetVersion();
  const have = installedVersion(python);

  if (have !== null && (wanted === null || have === wanted)) return python;

  const interpreter = findPython();
  if (!fs.existsSync(python)) {
    console.error(`aether-context: preparing a private Python environment in ${root}`);
    fs.mkdirSync(path.dirname(root), { recursive: true });
    run(interpreter.command, [...interpreter.args, "-m", "venv", root], "creating the virtualenv");
  }
  console.error(`aether-context: installing ${requirement()} from PyPI (first run only)`);
  run(python, ["-m", "pip", "install", "--quiet", "--upgrade", "pip"], "upgrading pip");
  run(python, ["-m", "pip", "install", "--quiet", requirement()], "installing aether-context");
  return python;
}

// --- entry point -------------------------------------------------------------------------------

function fail(message, hint) {
  console.error(`aether-context: ${message}`);
  if (hint) console.error(`  fix: ${hint}`);
  process.exit(1);
}

function main() {
  const python = ensureEnvironment();
  // `-m` rather than the console script: the module path is stable across platforms and does not
  // depend on where the venv put its Scripts/bin shims.
  const result = spawnSync(python, ["-m", "aether_context.cli", ...process.argv.slice(2)], {
    stdio: "inherit",
    windowsHide: true,
  });
  if (result.error) fail(`could not run aether-context: ${result.error.message}`);
  process.exit(result.status === null ? 1 : result.status);
}

main();
