"use strict";

/*
 * Vectorized Showdown battle host.
 *
 * Runs N independent Pokemon Showdown battles inside a single Node process and
 * multiplexes them over stdin (JSON-lines commands) and stdout (binary frames).
 * The Python side (`metamon.env.vectorized.sim_process.ShowdownSimProcess`) drives
 * the lanes and batches neural-network inference across them.
 *
 * Protocol:
 *
 *   Python -> host (stdin, JSON-lines):
 *     {"cmd": "start",  "lane": K, "formatid": "gen9ou", "seed": "...",
 *      "p1": {"name": "p1", "team": "<packed>"},
 *      "p2": {"name": "p2", "team": "<packed>"}}
 *     {"cmd": "choose", "lane": K, "side": "p1", "choice": "move 3"}
 *     {"cmd": "choose_batch", "choices": [{"lane": K, "epoch": E, "side": "p1", "choice": "move 3"}, ...]}
 *     {"cmd": "reset",  "lane": K}
 *     {"cmd": "ping"}
 *     {"cmd": "close"}
 *
 *   host -> Python (stdout, little-endian binary frames):
 *     type 0 ready
 *     type 1 chunk: u8 type, u32 lane, u32 epoch, u8 stream, u32 data_len, data[bytes]
 *         stream 0=p1, 1=p2, 2=error
 *     type 2 host_error: u8 type, u32 data_len, data[bytes]
 *     type 3 lane_error: u8 type, u32 lane, u32 epoch, u32 data_len, data[bytes]
 *     type 4 pong
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
const { Battle, BattleStream, getPlayerStreams } = Showdown;

/*
 * Test-time search snapshot/fork support.
 *
 * Snapshots are stored host-side as JSON *strings* (not objects) because
 * `State.serializeBattle` sets `state.log = battle.log` by reference — a fork
 * deserialized from an object snapshot would share the trunk's log array and
 * corrupt it when the fork appends new log lines. `Battle.fromJSON(string)`
 * runs `JSON.parse`, producing fresh arrays and breaking the alias.
 *
 * A fork replays its entire log (sentLogPos = 0) to a fresh Python lane so the
 * lane rebuilds full parsed state from scratch (exactly the normal ingestion
 * path), then re-emits each side's activeRequest so the lane parks at the same
 * settled decision point as the trunk.
 */
const snapshots = new Map(); // snapshot_id (int) -> JSON string

const MSG = { READY: 0, CHUNK: 1, HOST_ERROR: 2, LANE_ERROR: 3, PONG: 4 };
const STREAM = { p1: 0, p2: 1, error: 2 };

function emitReady() {
  const buf = Buffer.alloc(1);
  buf.writeUInt8(MSG.READY, 0);
  process.stdout.write(buf);
}

function emitPong() {
  const buf = Buffer.alloc(1);
  buf.writeUInt8(MSG.PONG, 0);
  process.stdout.write(buf);
}

function emitHostError(message) {
  const payload = Buffer.from(String(message), "utf8");
  const frame = Buffer.alloc(5 + payload.length);
  frame.writeUInt8(MSG.HOST_ERROR, 0);
  frame.writeUInt32LE(payload.length, 1);
  payload.copy(frame, 5);
  process.stdout.write(frame);
}

function emitLaneError(lane, epoch, message) {
  const payload = Buffer.from(String(message), "utf8");
  const frame = Buffer.alloc(13 + payload.length);
  frame.writeUInt8(MSG.LANE_ERROR, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt32LE(payload.length, 9);
  payload.copy(frame, 13);
  process.stdout.write(frame);
}

function emitChunk(lane, epoch, streamName, data) {
  const payload = Buffer.from(String(data), "utf8");
  const streamId = STREAM[streamName] !== undefined ? STREAM[streamName] : STREAM.error;
  const frame = Buffer.alloc(14 + payload.length);
  frame.writeUInt8(MSG.CHUNK, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt8(streamId, 9);
  frame.writeUInt32LE(payload.length, 10);
  payload.copy(frame, 14);
  process.stdout.write(frame);
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
    // Win/tie is detected on player streams in Python; we do not pump
    // omniscient (avoids ~33% duplicate IPC + parsing at high lane counts).

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
        if (chunk) emitChunk(id, epoch, name, chunk);
      }
    } catch (err) {
      emitLaneError(id, epoch, `${name}: ${err.message}`);
    }
  }

  choose(side, choice, epoch) {
    if (!this.streams) {
      emitLaneError(this.id, this.epoch, "choose before start");
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

  snapshot() {
    // Serialize the current battle to a JSON string (deep copy via stringify).
    if (!this.battleStream || !this.battleStream.battle) {
      throw new Error("cannot snapshot a lane with no active battle");
    }
    return JSON.stringify(this.battleStream.battle.toJSON());
  }

  startFromSnapshot(snapshotStr, epoch, replayLog = true, seed = null) {
    // Rebuild this lane from a serialized battle. Mirrors `start` but injects
    // a pre-deserialized Battle instead of running `>start`.
    //
    // ``seed`` (optional, test-time search): a branch-only PRNG seed. When
    // provided, the fork's inherited trunk future-RNG stream is replaced with
    // an independent branch-only PRNG so candidate actions are evaluated over
    // resampled future chance, not the trunk's actual hidden RNG realization
    // (skill §7 future-chance RNG leakage). ``setSeed`` is silent, unlike
    // ``Battle.resetRNG`` which emits a ``"message"`` log line that would
    // pollute branch logs. Accepts a 4x uint16 array (``[s0,s1,s2,s3]``) or a
    // ``"s0,s1,s2,s3"`` string (Gen5 LCG, the form metamon battles use).
    //
    // When ``replayLog`` is true (default, for a *fresh* Python lane that needs
    // to rebuild parsed state from scratch), we set ``sentLogPos = 0`` and flush
    // the entire log so the lane rebuilds, then re-emit each side's current
    // request so it parks at the trunk's settled decision point.
    //
    // When ``replayLog`` is false, the Python lane is a deep copy of the trunk's
    // parsed state (already populated, already holding the current request), so
    // we skip the log replay and request re-emit entirely — the fork only emits
    // *new* log entries from subsequent choices. This avoids the request-
    // regeneration quirk where ``deserializeBattle``'s ``getRequests`` produces a
    // request with slightly different field ordering/PP than the live emission.
    this.battleStream = new BattleStream();
    this.streams = getPlayerStreams(this.battleStream);
    this.closed = false;
    this.epoch = epoch;
    const battle = Battle.fromJSON(snapshotStr);
    // Optional branch-only PRNG reseed (applied before any choose so the first
    // branch transition draws from the branch seed, not the trunk's stream).
    if (seed != null) {
      battle.prng.setSeed(Array.isArray(seed) ? seed.join(",") : String(seed));
    }
    if (replayLog) {
      battle.sentLogPos = 0;
    } // else: keep serialized sentLogPos so only new log entries are sent
    // Wire `send` exactly like BattleStream's `>start` handler does.
    battle.restart((t, data) => {
      if (Array.isArray(data)) data = data.join("\n");
      this.battleStream.pushMessage(t, data);
      if (t === "end" && !this.battleStream.keepAlive) this.battleStream.pushEnd();
    });
    this.battleStream.battle = battle;
    const epochNow = epoch;
    this._pump(this.streams.p1, "p1", epochNow);
    this._pump(this.streams.p2, "p2", epochNow);
    if (replayLog) {
      // Flush the full log replay to the player streams...
      battle.sendUpdates();
      // ...then re-emit each side's current request so the lane sees a fresh
      // |request| and parks at the same settled decision point as the trunk.
      for (const side of battle.sides) {
        if (side.activeRequest) side.emitRequest(side.activeRequest);
      }
    }
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
    case "choose_batch": {
      for (const c of msg.choices || []) {
        getLane(c.lane).choose(c.side, c.choice, c.epoch);
      }
      break;
    }
    case "reset": {
      const lane = lanes.get(msg.lane);
      if (lane) {
        lane.destroy();
        // Remove the lane shell from the map so it does not accumulate:
        // test-time search allocates ever-incrementing fork-lane ids (never
        // recycled), so without this delete the map grows by ~A*K entries per
        // searched root per config and the host degrades / stalls after a few
        // hundred roots. getLane() re-creates a fresh Lane on demand.
        lanes.delete(msg.lane);
      }
      break;
    }
    case "snapshot": {
      // Store a serialized snapshot of `msg.lane` under `msg.snapshot_id`.
      const lane = getLane(msg.lane);
      snapshots.set(int(msg.snapshot_id), lane.snapshot());
      break;
    }
    case "fork": {
      // Create/replace lane `msg.to_lane` from a stored snapshot.
      const snap = snapshots.get(int(msg.snapshot_id));
      if (snap === undefined) {
        emitLaneError(msg.to_lane, 0, `fork: unknown snapshot_id ${msg.snapshot_id}`);
        break;
      }
      const lane = getLane(msg.to_lane);
      lane.destroy();
      const epoch = msg.epoch !== undefined ? msg.epoch : lane.epoch + 1;
      const replayLog = msg.replay_log !== undefined ? !!msg.replay_log : true;
      lane.startFromSnapshot(snap, epoch, replayLog, msg.seed);
      break;
    }
    case "fork_batch": {
      // Create many lanes from one stored snapshot in a single command.
      const snap = snapshots.get(int(msg.snapshot_id));
      if (snap === undefined) {
        emitHostError(`fork_batch: unknown snapshot_id ${msg.snapshot_id}`);
        break;
      }
      const replayLog = msg.replay_log !== undefined ? !!msg.replay_log : true;
      for (const toLane of (msg.to_lanes || [])) {
        const lane = getLane(int(toLane.lane));
        lane.destroy();
        lane.startFromSnapshot(snap, int(toLane.epoch), replayLog, toLane.seed);
      }
      break;
    }
    case "restore": {
      // Replace `msg.lane`'s battle with a stored snapshot (advances epoch so
      // any in-flight chunks from the old battle are dropped by Python).
      const snap = snapshots.get(int(msg.snapshot_id));
      if (snap === undefined) {
        emitLaneError(msg.lane, 0, `restore: unknown snapshot_id ${msg.snapshot_id}`);
        break;
      }
      const lane = getLane(msg.lane);
      lane.destroy();
      const epoch = msg.epoch !== undefined ? msg.epoch : lane.epoch + 1;
      const replayLog = msg.replay_log !== undefined ? !!msg.replay_log : true;
      lane.startFromSnapshot(snap, epoch, replayLog, msg.seed);
      break;
    }
    case "release_snapshot": {
      snapshots.delete(int(msg.snapshot_id));
      break;
    }
    case "ping": {
      emitPong();
      break;
    }
    case "close": {
      for (const lane of lanes.values()) lane.destroy();
      process.exit(0);
      break;
    }
    default:
      emitHostError(`unknown cmd: ${JSON.stringify(msg)}`);
  }
}

// Coerce a value that may arrive from JSON as a number or string to an int.
function int(x) {
  const n = Number(x);
  if (!Number.isInteger(n)) throw new Error(`expected int, got ${JSON.stringify(x)}`);
  return n;
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch (err) {
    emitHostError(`bad json: ${err.message}`);
    return;
  }
  try {
    handleCommand(msg);
  } catch (err) {
    emitLaneError(
      msg && msg.lane !== undefined ? msg.lane : 0,
      msg && msg.epoch !== undefined ? msg.epoch : 0,
      `cmd ${msg && msg.cmd}: ${err.message}`
    );
  }
});
rl.on("close", () => process.exit(0));

emitReady();
