"""Python transport for the vectorized Showdown Node host.

Spawns ``battle_host.js`` (a single Node process running N in-process Showdown
``BattleStream``s) and multiplexes a binary frame protocol on stdout plus
JSON-lines commands on stdin. A background thread reads the host's stdout into a
thread-safe queue; the env thread dispatches those chunks to per-lane handlers
and blocks until every lane that owes a decision has produced a fresh
``|request|`` (or the battle ended).

This module is intentionally decoupled from battle parsing: it only moves bytes
and routes them. The per-lane parsing/observation logic lives in
:mod:`metamon.env.vectorized.lane`.
"""

from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


HOST_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battle_host.js")

# Binary stdout frame types (must match battle_host.js).
MSG_READY = 0
MSG_CHUNK = 1
MSG_HOST_ERROR = 2
MSG_LANE_ERROR = 3
MSG_PONG = 4

_STREAM_BY_ID = ("p1", "p2", "error")
_CHUNK_BODY = struct.Struct("<BIIBI")  # type, lane, epoch, stream_id, data_len
_HOST_ERROR_BODY = struct.Struct("<BI")  # type, data_len
_LANE_ERROR_BODY = struct.Struct("<BIII")  # type, lane, epoch, data_len


class LaneHandler(Protocol):
    """Anything that can consume a host chunk for a single lane."""

    def handle_chunk(self, stream: str, data: str) -> None: ...


class ShowdownSimProcessError(RuntimeError):
    pass


class ShowdownSimProcess:
    """Owns the Node host subprocess and the host transport.

    Args:
        node_path: Node executable (defaults to ``node`` on PATH).
        host_script: Path to ``battle_host.js`` (defaults to the bundled copy).
        showdown_dist: Optional path to a built ``pokemon-showdown`` sim dist.
            Forwarded to the host as ``METAMON_SHOWDOWN_DIST`` so development
            setups without an installed package can still run. In production the
            host resolves the installed ``pokemon-showdown`` package directly.
        ready_timeout: Seconds to wait for the host's ready frame.
    """

    def __init__(
        self,
        node_path: str = "node",
        host_script: str = HOST_SCRIPT,
        showdown_dist: Optional[str] = None,
        ready_timeout: float = 30.0,
    ):
        if not os.path.exists(host_script):
            raise FileNotFoundError(f"battle host script not found: {host_script}")
        env = dict(os.environ)
        if showdown_dist is not None:
            env["METAMON_SHOWDOWN_DIST"] = str(showdown_dist)

        self._proc = subprocess.Popen(
            [node_path, host_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self._handlers: Dict[int, LaneHandler] = {}
        # Per-lane battle epoch; chunks tagged with a stale epoch are dropped so
        # a previous (destroyed) battle that is still draining cannot corrupt the
        # state of the new battle that reused the same lane id.
        self._epoch: Dict[int, int] = {}
        self._inbox: "queue.Queue[dict]" = queue.Queue()
        self._closed = False
        self._write_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._stdout_buf = bytearray()

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

        self._await_ready(ready_timeout)

    # ----- subprocess IO ---------------------------------------------------

    def _read_exact(self, n: int) -> Optional[bytes]:
        assert self._proc.stdout is not None
        while len(self._stdout_buf) < n:
            chunk = self._proc.stdout.read(max(65536, n - len(self._stdout_buf)))
            if not chunk:
                return None
            self._stdout_buf.extend(chunk)
        out = bytes(self._stdout_buf[:n])
        del self._stdout_buf[:n]
        return out

    def _read_stdout(self) -> None:
        while True:
            type_b = self._read_exact(1)
            if type_b is None:
                break
            msg_type = type_b[0]
            if msg_type == MSG_READY:
                self._inbox.put({"event": "ready"})
                continue
            if msg_type == MSG_PONG:
                self._inbox.put({"event": "pong"})
                continue
            if msg_type == MSG_CHUNK:
                body = self._read_exact(_CHUNK_BODY.size - 1)
                if body is None:
                    break
                lane, epoch, stream_id, data_len = _CHUNK_BODY.unpack(type_b + body)[1:]
                data_b = self._read_exact(data_len)
                if data_b is None:
                    break
                stream = (
                    _STREAM_BY_ID[stream_id]
                    if 0 <= stream_id < len(_STREAM_BY_ID)
                    else "error"
                )
                self._inbox.put(
                    {
                        "lane": lane,
                        "epoch": epoch,
                        "stream": stream,
                        "data": data_b.decode("utf-8"),
                    }
                )
                continue
            if msg_type == MSG_HOST_ERROR:
                body = self._read_exact(_HOST_ERROR_BODY.size - 1)
                if body is None:
                    break
                (data_len,) = _HOST_ERROR_BODY.unpack(type_b + body)[1:]
                data_b = self._read_exact(data_len)
                if data_b is None:
                    break
                self._inbox.put({"stream": "error", "data": data_b.decode("utf-8")})
                continue
            if msg_type == MSG_LANE_ERROR:
                body = self._read_exact(_LANE_ERROR_BODY.size - 1)
                if body is None:
                    break
                _, lane, epoch, data_len = _LANE_ERROR_BODY.unpack(type_b + body)
                data_b = self._read_exact(data_len)
                if data_b is None:
                    break
                self._inbox.put(
                    {
                        "lane": lane,
                        "epoch": epoch,
                        "stream": "error",
                        "data": data_b.decode("utf-8"),
                    }
                )
                continue
            self._inbox.put(
                {
                    "stream": "error",
                    "data": f"unknown host frame type {msg_type}",
                }
            )
            break
        self._inbox.put({"event": "_eof"})

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if text:
                self._stderr_lines.append(text)

    def _await_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self._inbox.get(timeout=deadline - time.monotonic())
            except queue.Empty:
                break
            if msg.get("event") == "ready":
                return
            if msg.get("event") == "_eof":
                break
            # Stash any pre-ready lane traffic (shouldn't happen) back for later.
            self._inbox.put(msg)
            time.sleep(0.001)
        raise ShowdownSimProcessError(
            "Showdown host did not become ready. stderr:\n"
            + "\n".join(self._stderr_lines[-20:])
        )

    def _send(self, msg: dict) -> None:
        if self._closed:
            raise ShowdownSimProcessError("send on closed ShowdownSimProcess")
        if self._proc.poll() is not None:
            raise ShowdownSimProcessError(
                "Showdown host exited (code "
                f"{self._proc.returncode}). stderr:\n"
                + "\n".join(self._stderr_lines[-20:])
            )
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            assert self._proc.stdin is not None
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    # ----- public commands -------------------------------------------------

    def register_lane(self, lane_id: int, handler: LaneHandler) -> None:
        self._handlers[int(lane_id)] = handler
        self._epoch.setdefault(int(lane_id), 0)

    def start_battle(
        self,
        lane: int,
        formatid: str,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        seed: Optional[Any] = None,
    ) -> None:
        lane = int(lane)
        self._epoch[lane] = self._epoch.get(lane, 0) + 1
        msg = {
            "cmd": "start",
            "lane": lane,
            "epoch": self._epoch[lane],
            "formatid": formatid,
            "p1": p1,
            "p2": p2,
        }
        if seed is not None:
            msg["seed"] = seed
        self._send(msg)

    def choose(self, lane: int, side: str, choice: str) -> None:
        lane = int(lane)
        self._send(
            {
                "cmd": "choose",
                "lane": lane,
                "epoch": self._epoch.get(lane, 0),
                "side": side,
                "choice": choice,
            }
        )

    def choose_batch(self, entries: List[Tuple[int, str, str]]) -> None:
        """Send many lane choices in one stdin write (one flush)."""
        if not entries:
            return
        if len(entries) == 1:
            lane, side, choice = entries[0]
            self.choose(lane, side, choice)
            return
        choices = [
            {
                "lane": int(lane),
                "epoch": self._epoch.get(int(lane), 0),
                "side": side,
                "choice": choice,
            }
            for lane, side, choice in entries
        ]
        self._send({"cmd": "choose_batch", "choices": choices})

    def reset(self, lane: int) -> None:
        self._send({"cmd": "reset", "lane": int(lane)})

    def ping(self) -> None:
        self._send({"cmd": "ping"})

    # ----- pumping ---------------------------------------------------------

    def _dispatch(self, msg: dict) -> None:
        event = msg.get("event")
        if event is not None:
            if event == "_eof":
                raise ShowdownSimProcessError(
                    "Showdown host stdout closed unexpectedly. stderr:\n"
                    + self.stderr_tail
                )
            return
        lane_id = msg.get("lane")
        stream = msg.get("stream")
        data = msg.get("data", "")
        if lane_id is not None:
            epoch = msg.get("epoch")
            if epoch is not None and epoch != self._epoch.get(int(lane_id)):
                return
        if stream == "error":
            raise ShowdownSimProcessError(f"host error (lane {lane_id}): {data}")
        if lane_id is None:
            return
        handler = self._handlers.get(int(lane_id))
        if handler is not None:
            handler.handle_chunk(stream, data)

    def pump_until(
        self,
        predicate: Callable[[], bool],
        timeout: float = 90.0,
        idle_timeout: float = 45.0,
    ) -> None:
        """Dispatch host chunks until ``predicate()`` is True."""
        if predicate():
            return
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ShowdownSimProcessError(
                    f"pump_until timed out after {timeout}s. stderr:\n"
                    + self.stderr_tail
                )
            try:
                msg = self._inbox.get(timeout=min(idle_timeout, remaining))
            except queue.Empty:
                raise ShowdownSimProcessError(
                    f"pump_until idle for {idle_timeout}s (host produced no "
                    "output). stderr:\n" + self.stderr_tail
                )
            self._dispatch(msg)
            while True:
                try:
                    msg = self._inbox.get_nowait()
                except queue.Empty:
                    break
                self._dispatch(msg)
            if predicate():
                return

    # ----- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                self._send_close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        # Let the reader threads observe EOF and exit before we close the pipes,
        # otherwise closing a pipe out from under a blocked daemon reader can
        # abort the interpreter during shutdown.
        for thread in (
            getattr(self, "_reader", None),
            getattr(self, "_stderr_reader", None),
        ):
            if thread is not None:
                thread.join(timeout=2.0)
        for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass

    def _send_close(self) -> None:
        line = (json.dumps({"cmd": "close"}) + "\n").encode("utf-8")
        with self._write_lock:
            if self._proc.stdin is not None:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines[-20:])


class ShardedShowdownSimProcess:
    """Fan-out/fan-in coordinator over several :class:`ShowdownSimProcess` workers.

    Global lane ids seen by the env are mapped to contiguous local lane ids on each
    worker. Opponent/eval NN batching stays at the env level (``batched_envs`` lanes);
    only Showdown simulation and IPC are sharded across Node processes.
    """

    def __init__(
        self,
        num_lanes: int,
        n_workers: int,
        node_path: str = "node",
        host_script: str = HOST_SCRIPT,
        showdown_dist: Optional[str] = None,
        ready_timeout: float = 30.0,
    ):
        self._num_lanes = int(num_lanes)
        self._n_workers = min(max(1, int(n_workers)), self._num_lanes)
        self._handlers: Dict[int, LaneHandler] = {}
        self._epoch: Dict[int, int] = {}
        self._inbox: "queue.Queue[dict]" = queue.Queue()
        self._closed = False

        self._worker_bases: List[int] = []
        self._worker_counts: List[int] = []
        offset = 0
        base = self._num_lanes // self._n_workers
        rem = self._num_lanes % self._n_workers
        for i in range(self._n_workers):
            count = base + (1 if i < rem else 0)
            self._worker_bases.append(offset)
            self._worker_counts.append(count)
            offset += count

        worker_kwargs = dict(
            node_path=node_path,
            host_script=host_script,
            showdown_dist=showdown_dist,
            ready_timeout=ready_timeout,
        )
        self._workers = [
            ShowdownSimProcess(**worker_kwargs) for _ in range(self._n_workers)
        ]
        self._relay_threads = [
            threading.Thread(
                target=self._relay_worker, args=(worker_id, proc), daemon=True
            )
            for worker_id, proc in enumerate(self._workers)
        ]
        for thread in self._relay_threads:
            thread.start()

    def _global_lane(self, worker_id: int, local_lane: int) -> int:
        return self._worker_bases[worker_id] + int(local_lane)

    def _local_lane(self, global_lane: int) -> Tuple[int, int]:
        global_lane = int(global_lane)
        for worker_id in range(self._n_workers):
            base = self._worker_bases[worker_id]
            count = self._worker_counts[worker_id]
            if base <= global_lane < base + count:
                return worker_id, global_lane - base
        raise ShowdownSimProcessError(f"invalid global lane id {global_lane}")

    def _relay_worker(self, worker_id: int, proc: ShowdownSimProcess) -> None:
        while not self._closed:
            try:
                msg = proc._inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            event = msg.get("event")
            if event == "_eof":
                self._inbox.put({"event": "_eof", "worker": worker_id})
                return
            if event in ("ready", "pong"):
                continue
            local_lane = msg.get("lane")
            if local_lane is not None:
                msg = dict(msg)
                msg["lane"] = self._global_lane(worker_id, int(local_lane))
            self._inbox.put(msg)

    @property
    def stderr_tail(self) -> str:
        parts = []
        for i, proc in enumerate(self._workers):
            tail = proc.stderr_tail
            if tail:
                parts.append(f"[worker {i}]\n{tail}")
        return "\n".join(parts)

    def register_lane(self, lane_id: int, handler: LaneHandler) -> None:
        self._handlers[int(lane_id)] = handler
        self._epoch.setdefault(int(lane_id), 0)

    def start_battle(
        self,
        lane: int,
        formatid: str,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        seed: Optional[Any] = None,
    ) -> None:
        lane = int(lane)
        worker_id, local_lane = self._local_lane(lane)
        self._epoch[lane] = self._epoch.get(lane, 0) + 1
        self._workers[worker_id].start_battle(local_lane, formatid, p1, p2, seed=seed)

    def choose(self, lane: int, side: str, choice: str) -> None:
        worker_id, local_lane = self._local_lane(int(lane))
        self._workers[worker_id].choose(local_lane, side, choice)

    def choose_batch(self, entries: List[Tuple[int, str, str]]) -> None:
        if not entries:
            return
        by_worker: Dict[int, List[Tuple[int, str, str]]] = {}
        for lane, side, choice in entries:
            worker_id, local_lane = self._local_lane(int(lane))
            by_worker.setdefault(worker_id, []).append((local_lane, side, choice))
        for worker_id, batch in by_worker.items():
            self._workers[worker_id].choose_batch(batch)

    def reset(self, lane: int) -> None:
        worker_id, local_lane = self._local_lane(int(lane))
        self._workers[worker_id].reset(local_lane)

    def ping(self) -> None:
        for proc in self._workers:
            proc.ping()

    def _dispatch(self, msg: dict) -> None:
        event = msg.get("event")
        if event is not None:
            if event == "_eof":
                worker = msg.get("worker")
                raise ShowdownSimProcessError(
                    f"Showdown host worker {worker} stdout closed unexpectedly. "
                    f"stderr:\n{self.stderr_tail}"
                )
            return
        lane_id = msg.get("lane")
        stream = msg.get("stream")
        data = msg.get("data", "")
        if lane_id is not None:
            epoch = msg.get("epoch")
            if epoch is not None and epoch != self._epoch.get(int(lane_id)):
                return
        if stream == "error":
            raise ShowdownSimProcessError(f"host error (lane {lane_id}): {data}")
        if lane_id is None:
            return
        handler = self._handlers.get(int(lane_id))
        if handler is not None:
            handler.handle_chunk(stream, data)

    def pump_until(
        self,
        predicate: Callable[[], bool],
        timeout: float = 90.0,
        idle_timeout: float = 45.0,
    ) -> None:
        if predicate():
            return
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ShowdownSimProcessError(
                    f"pump_until timed out after {timeout}s. stderr:\n"
                    + self.stderr_tail
                )
            try:
                msg = self._inbox.get(timeout=min(idle_timeout, remaining))
            except queue.Empty:
                raise ShowdownSimProcessError(
                    f"pump_until idle for {idle_timeout}s (host produced no "
                    "output). stderr:\n" + self.stderr_tail
                )
            self._dispatch(msg)
            while True:
                try:
                    msg = self._inbox.get_nowait()
                except queue.Empty:
                    break
                self._dispatch(msg)
            if predicate():
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for proc in self._workers:
            try:
                proc.close()
            except Exception:
                pass
        for thread in self._relay_threads:
            thread.join(timeout=2.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_sim_process(
    num_lanes: int,
    n_workers: int = 1,
    node_path: str = "node",
    host_script: str = HOST_SCRIPT,
    showdown_dist: Optional[str] = None,
    ready_timeout: float = 30.0,
):
    """Return a single- or multi-Node Showdown transport for ``num_lanes`` battles."""
    n_workers = int(n_workers)
    if n_workers <= 1:
        return ShowdownSimProcess(
            node_path=node_path,
            host_script=host_script,
            showdown_dist=showdown_dist,
            ready_timeout=ready_timeout,
        )
    return ShardedShowdownSimProcess(
        num_lanes=int(num_lanes),
        n_workers=n_workers,
        node_path=node_path,
        host_script=host_script,
        showdown_dist=showdown_dist,
        ready_timeout=ready_timeout,
    )
