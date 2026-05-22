#!/usr/bin/env node

/**
 * Persistent Pokemon Showdown team validator.
 *
 * Protocol:
 * - Read JSON lines on stdin: {"format": "gen1ou", "team": "<team string>"}
 * - Write JSON lines on stdout: {"ok": true/false, "errors": ["..."]}
 */

const readline = require("readline");

let TeamValidator;
let Teams;

function loadFromDist(distRoot) {
  ({ TeamValidator } = require(`${distRoot}/sim/team-validator`));
  const teamsModule = require(`${distRoot}/sim/teams`);
  Teams = teamsModule.Teams || teamsModule;
}

function loadFromPackage() {
  try {
    ({ TeamValidator, Teams } = require("pokemon-showdown"));
  } catch (errPrimary) {
    try {
      ({ TeamValidator } = require("pokemon-showdown/dist/sim/team-validator"));
    } catch (errSecondary) {
      const message =
        "Unable to load pokemon-showdown. Set SHOWDOWN_DIST to a built checkout " +
        "or ensure a Showdown server is running / installed on this machine.";
      console.error(message);
      console.error(String(errPrimary));
      console.error(String(errSecondary));
      process.exit(1);
    }
  }

  if (!Teams) {
    try {
      const teamsModule = require("pokemon-showdown/dist/sim/teams");
      Teams = teamsModule.Teams || teamsModule;
    } catch (err) {
      console.error("Unable to load pokemon-showdown Teams module.");
      console.error(String(err));
      process.exit(1);
    }
  }
}

const showdownDist = process.env.SHOWDOWN_DIST;
if (showdownDist) {
  try {
    loadFromDist(showdownDist);
  } catch (err) {
    console.error(`Unable to load pokemon-showdown from SHOWDOWN_DIST=${showdownDist}`);
    console.error(String(err));
    process.exit(1);
  }
} else {
  loadFromPackage();
}

const validatorsByFormat = new Map();

function getValidator(format) {
  if (!validatorsByFormat.has(format)) {
    validatorsByFormat.set(format, TeamValidator.get(format));
  }
  return validatorsByFormat.get(format);
}

function normalizeErrors(result) {
  if (!result) return [];
  if (Array.isArray(result)) return result.map((e) => String(e));
  return [String(result)];
}

function respond(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;

  let req;
  try {
    req = JSON.parse(trimmed);
  } catch (err) {
    respond({ ok: false, errors: ["Invalid JSON input."] });
    return;
  }

  const format = req.format;
  const team = req.team;

  if (!format || typeof team !== "string") {
    respond({ ok: false, errors: ["Input must include format and team string."] });
    return;
  }

  let validator;
  try {
    validator = getValidator(format);
  } catch (err) {
    respond({ ok: false, errors: [String(err)] });
    return;
  }

  try {
    const parsedTeam = Teams.import(team);
    if (!parsedTeam) {
      respond({ ok: false, errors: ["Invalid team data"] });
      return;
    }
    const result = validator.validateTeam(parsedTeam);
    const errors = normalizeErrors(result);
    respond({ ok: errors.length === 0, errors });
  } catch (err) {
    respond({ ok: false, errors: [String(err)] });
  }
});

rl.on("close", () => {
  process.exit(0);
});
