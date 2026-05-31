"use strict";

/*
 * Vectorized Showdown battle host.
 *
 * Runs N independent Pokemon Showdown battles inside a single Node process and
 * multiplexes them over a JSON-lines protocol on stdin/stdout. The Python side
 * (`metamon.env.vectorized.sim_process.ShowdownSimProcess`) drives the lanes and
 * batches neural-network inference across them.
 *
 * Protocol (one JSON object per line):
 *
 *   Python -> host (stdin):
 *     {"cmd": "start",  "lane": K, "formatid": "gen9ou", "seed": "...",
 *      "p1": {"name": "p1", "team": "<packed>"},
 *      "p2": {"name": "p2", "team": "<packed>"}}
 *     {"cmd": "choose", "lane": K, "side": "p1", "choice": "move 3"}
 *     {"cmd": "reset",  "lane": K}      // tear a lane down (a fresh start re-creates it)
 *     {"cmd": "ping"}                    // host replies {"event": "pong"}
 *     {"cmd": "close"}                   // host exits
 *
 *   host -> Python (stdout):
 *     {"lane": K, "epoch": E, "stream": "p1"|"p2"|"omniscient", "data": "<text>"}
 *     {"lane": K, "epoch": E, "stream": "error", "data": "<message>"}
 *     {"event": "ready"}                                   // emitted once at startup
 *     {"event": "pong"}
 *
 * The `pokemon-showdown` package is resolved normally; for development a dist
 * path can be supplied via METAMON_SHOWDOWN_DIST (we never modify that source).
 */

const readline = require("readline");

function loadShowdown() {
  try {
    return require("pokemon-showdown");
  } catch (errPkg) {
    const devPath = process.env.METAMON_SHOWDOWN_DIST;
    if (devPath) {
      try {
        return require(devPath);
      } catch (errDev) {
        throw new Error(
          `Could not require 'pokemon-showdown' (${errPkg.message}) nor ` +
            `METAMON_SHOWDOWN_DIST='${devPath}' (${errDev.message})`
        );
      }
    }
    throw new Error(
      "Could not require 'pokemon-showdown'. Install it in this package " +
        "(npm install) or set METAMON_SHOWDOWN_DIST to a built sim dist. " +
        `Underlying error: ${errPkg.message}`
    );
  }
}

const Showdown = loadShowdown();
const { BattleStream, getPlayerStreams } = Showdown;

// Single shared writable stream to stdout. Each emit is exactly one line.
function emit(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

class Lane {
  constructor(id) {
    this.id = id;
    this.battleStream = null;
    this.streams = null;
    this.closed = false;
    // Monotonic battle counter. Every emitted message carries the epoch of the
    // battle that produced it so Python can drop stale chunks from a previous
    // (destroyed) battle that may still be draining when a new one starts.
    this.epoch = 0;
  }

  start(spec, p1spec, p2spec, epoch) {
    // A fresh BattleStream per battle keeps lanes fully isolated.
    this.battleStream = new BattleStream();
    this.streams = getPlayerStreams(this.battleStream);
    this.closed = false;
    this.epoch = epoch;

    // Capture the epoch in each pump closure so trailing chunks from an old
    // battle keep their original epoch even after `this.epoch` advances.
    const epochNow = epoch;
    // Pump each player's channel-filtered protocol + private requests. Each
    // side becomes its own point-of-view that Python parses into a battle.
    //
    // NOTE: we must NOT also iterate the raw `battleStream` here.
    // `getPlayerStreams` already attaches a consumer to it; a second consumer
    // would split (steal) chunks and silently drop messages such as p2's
    // request. Battle end is detected Python-side from `|win|`/`|tie|`.
    this._pump(this.streams.p1, "p1", epochNow);
    this._pump(this.streams.p2, "p2", epochNow);
    // Omniscient stream: full log, used by Python for win/tie detection and
    // optional replay logging. Not required for per-POV parsing.
    this._pump(this.streams.omniscient, "omniscient", epochNow);

    const initMessage =
      `>start ${JSON.stringify(spec)}\n` +
      `>player p1 ${JSON.stringify(p1spec)}\n` +
      `>player p2 ${JSON.stringify(p2spec)}`;
    void this.streams.omniscient.write(initMessage);
  }

  async _pump(stream, name, epoch) {
    const id = this.id;
    try {
      for await (const chunk of stream) {
        if (chunk) emit({ lane: id, epoch, stream: name, data: chunk });
      }
    } catch (err) {
      emit({ lane: id, epoch, stream: "error", data: `${name}: ${err.message}` });
    }
  }

  choose(side, choice, epoch) {
    if (!this.streams) {
      emit({ lane: this.id, epoch: this.epoch, stream: "error", data: "choose before start" });
      return;
    }
    if (epoch !== undefined && epoch !== this.epoch) {
      // Stale choice aimed at a previous battle; ignore.
      return;
    }
    const stream = side === "p1" ? this.streams.p1 : this.streams.p2;
    void stream.write(String(choice));
  }

  destroy() {
    this.closed = true;
    if (this.battleStream) {
      try {
        void this.battleStream.destroy();
      } catch (err) {
        /* already torn down */
      }
    }
    this.battleStream = null;
    this.streams = null;
  }
}

const lanes = new Map();

function getLane(id) {
  let lane = lanes.get(id);
  if (!lane) {
    lane = new Lane(id);
    lanes.set(id, lane);
  }
  return lane;
}

function handleCommand(msg) {
  switch (msg.cmd) {
    case "start": {
      const lane = getLane(msg.lane);
      lane.destroy();
      const spec = { formatid: msg.formatid };
      if (msg.seed !== undefined && msg.seed !== null) spec.seed = msg.seed;
      const epoch = msg.epoch !== undefined ? msg.epoch : lane.epoch + 1;
      lane.start(spec, msg.p1 || { name: "p1" }, msg.p2 || { name: "p2" }, epoch);
      break;
    }
    case "choose": {
      getLane(msg.lane).choose(msg.side, msg.choice, msg.epoch);
      break;
    }
    case "reset": {
      const lane = lanes.get(msg.lane);
      if (lane) lane.destroy();
      break;
    }
    case "ping": {
      emit({ event: "pong" });
      break;
    }
    case "close": {
      for (const lane of lanes.values()) lane.destroy();
      process.exit(0);
      break;
    }
    default:
      emit({ stream: "error", data: `unknown cmd: ${JSON.stringify(msg)}` });
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch (err) {
    emit({ stream: "error", data: `bad json: ${err.message}` });
    return;
  }
  try {
    handleCommand(msg);
  } catch (err) {
    emit({
      lane: msg && msg.lane,
      stream: "error",
      data: `cmd ${msg && msg.cmd}: ${err.message}`,
    });
  }
});
rl.on("close", () => process.exit(0));

emit({ event: "ready" });
