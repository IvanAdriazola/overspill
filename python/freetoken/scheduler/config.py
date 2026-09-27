from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def zmq_local_addr(channel: int, unique_suffix: str) -> str:
    """Process-local ZMQ endpoint for internal channel ``channel``.

    Unix: an ipc:// socket under /tmp. Windows' libzmq has no ipc:// transport, so there it is a
    loopback TCP port derived from the launching process's pid (the config carrying the suffix
    is shared with every child process, so all sides compute the same port)."""
    import os
    import zlib

    if os.name != "nt":
        return f"ipc:///tmp/freetoken_{channel}{unique_suffix}"
    port = 20000 + (zlib.crc32(unique_suffix.encode()) % 4000) * 10 + channel
    return f"tcp://127.0.0.1:{port}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return zmq_local_addr(0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return zmq_local_addr(1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return zmq_local_addr(2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
