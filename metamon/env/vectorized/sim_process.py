"""Python transport for the vectorized Showdown Node host.

Spawns ``battle_host.js`` (a single Node process running N in-process Showdown
``BattleStream``s) and multiplexes the JSON-lines protocol over stdin/stdout. A
background thread reads the host's stdout into a thread-safe queue; the env
thread dispatches those chunks to per-lane handlers and blocks until every lane
that owes a decision has produced a fresh ``|request|`` (or the battle ended).

This module is intentionally decoupled from battle parsing: it only moves bytes
and routes them. The per-lane parsing/observation logic lives in
:mod:`metamon.env.vectorized.lane`.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol


HOST_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battle_host.js")


class LaneHandler(Protocol):
    """Anything that can consume a host chunk for a single lane."""

    def handle_chunk(self, stream: str, data: str) -> None: ...


class ShowdownSimProcessError(RuntimeError):
    pass


class ShowdownSimProcess:
    """Owns the Node host subprocess and the JSON-lines transport.

    Args:
        node_path: Node executable (defaults to ``node`` on PATH).
        host_script: Path to ``battle_host.js`` (defaults to the bundled copy).
        showdown_dist: Optional path to a built ``pokemon-showdown`` sim dist.
            Forwarded to the host as ``METAMON_SHOWDOWN_DIST`` so development
            setups without an installed package can still run. In production the
            host resolves the installed ``pokemon-showdown`` package directly.
        ready_timeout: Seconds to wait for the host's ``{"event":"ready"}``.
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
            bufsize=1,
            text=True,
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

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

        self._await_ready(ready_timeout)

    # ----- subprocess IO ---------------------------------------------------

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._inbox.put(
                    {"stream": "error", "data": f"bad host line: {line[:200]}"}
                )
                continue
            self._inbox.put(msg)
        self._inbox.put({"event": "_eof"})

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            line = line.rstrip("\n")
            if line:
                self._stderr_lines.append(line)

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
        line = json.dumps(msg, separators=(",", ":")) + "\n"
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
                    + "\n".join(self._stderr_lines[-20:])
                )
            return
        lane_id = msg.get("lane")
        stream = msg.get("stream")
        data = msg.get("data", "")
        if lane_id is not None:
            # Drop chunks from a superseded battle on this lane.
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
        timeout: float = 30.0,
        idle_timeout: float = 10.0,
    ) -> None:
        """Dispatch host chunks until ``predicate()`` is True.

        Args:
            predicate: Returns True once the caller has received everything it
                needs (e.g. every pending lane has a fresh request or ended).
            timeout: Hard cap on total wait time.
            idle_timeout: Max time to wait for any single new chunk before
                declaring the host stuck.
        """
        if predicate():
            return
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ShowdownSimProcessError(
                    f"pump_until timed out after {timeout}s. stderr:\n"
                    + "\n".join(self._stderr_lines[-20:])
                )
            try:
                msg = self._inbox.get(timeout=min(idle_timeout, remaining))
            except queue.Empty:
                raise ShowdownSimProcessError(
                    f"pump_until idle for {idle_timeout}s (host produced no "
                    "output). stderr:\n" + "\n".join(self._stderr_lines[-20:])
                )
            self._dispatch(msg)
            # Drain anything else already queued before re-checking the predicate
            # (keeps us from checking the predicate on every single chunk).
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
        line = json.dumps({"cmd": "close"}) + "\n"
        with self._write_lock:
            if self._proc.stdin is not None:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
