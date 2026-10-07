"""Which server each role talks to, in one place.

Resolved per call, first match wins: ``QWEN_<ROLE>_URL`` (comma-separated for replicas), then
the ``roles:`` table of the servers file (``$STORM_SERVERS``, else ``configs/servers.yaml``;
``configs/servers.example.yaml`` is the layout this project was measured on), then
:data:`ROLES`. ``QWEN_<ROLE>_IMAGE_SIDE`` overrides a role's picture size (:data:`IMAGE_SIDES`).

A role may be spread over several identical servers -- :func:`client_for` returns a
:class:`ReplicatedClient` the moment a role resolves to more than one URL.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
from typing import Dict, List, Sequence

from .qwen import QwenClient, VlmReply

#: role -> the servers it uses when neither the environment nor the servers file names one:
#: the executor on one server, everything else sharing a second.
ROLES: Dict[str, List[str]] = {
    "executor": ["http://127.0.0.1:8082/v1"],
    "planner": ["http://127.0.0.1:8081/v1"],
    "monitor": ["http://127.0.0.1:8081/v1"],
    "memory": ["http://127.0.0.1:8081/v1"],
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def servers_file() -> Dict[str, List[str]]:
    """The ``roles:`` table of the servers file, or ``{}`` when there is none."""
    path = os.environ.get("STORM_SERVERS") or os.path.join(ROOT, "configs", "servers.yaml")
    if not os.path.exists(path):
        return {}
    import yaml
    roles = (yaml.safe_load(open(path)) or {}).get("roles") or {}
    return {role: [urls] if isinstance(urls, str) else list(urls) for role, urls in roles.items()}

#: role -> how big a picture it sends, on the longest side.
IMAGE_SIDES: Dict[str, int] = {
    "executor": 512,
    "planner": 768,     # the one role above 512
    "monitor": 512,
    "memory": 512,
}


#: role -> how much context its server is started with (configs/servers.example.yaml: 32k each).
CONTEXT_TOKENS: Dict[str, int] = {
    "executor": 32768,
    "planner": 32768,
    "monitor": 32768,
    "memory": 32768,
}

#: How many tokens of TEXT each prompt carries, and how many pictures.
PROMPT_SHAPES = {
    # Measured 2026-09-19 over the 580 propose calls and 339 supervise calls of the round-11
    # regression sweep.
    "executor": {"propose": (5331, 3), "supervise": (1590, 6)},
    "planner": {"plan": (8630, 3), "verify": (4617, 6)},
}


def image_tokens(side: int) -> int:
    """What one picture costs this server, measured: ``(side / 32)^2 + 2``."""
    return (int(side) // 32) ** 2 + 2


def env_var(role: str) -> str:
    """The environment variable that overrides ``role``'s URL (or its comma-separated URLs)."""
    return "QWEN_{}_URL".format(role.upper())


def image_side_var(role: str) -> str:
    """The environment variable that overrides ``role``'s picture size."""
    return "QWEN_{}_IMAGE_SIDE".format(role.upper())


def image_side_for(role: str) -> int:
    """How big a picture ``role`` sends, the environment winning over the table above."""
    if role not in IMAGE_SIDES:
        raise KeyError("no such role {!r}; this project has {}"
                       .format(role, ", ".join(sorted(IMAGE_SIDES))))
    raw = os.environ.get(image_side_var(role))
    if raw and raw.strip():
        try:
            return int(raw.strip())
        except ValueError:
            raise ValueError("{} is {!r}, which is not a number of pixels"
                             .format(image_side_var(role), raw))
    return IMAGE_SIDES[role]


def urls_for(role: str) -> List[str]:
    """Every server ``role`` may talk to, the environment winning over the table above."""
    if role not in ROLES:
        raise KeyError("no such role {!r}; this project has {}"
                       .format(role, ", ".join(sorted(ROLES))))
    override = os.environ.get(env_var(role)) or ""
    listed = [part.strip() for part in override.split(",") if part.strip()]
    return listed or servers_file().get(role) or list(ROLES[role])


def url_for(role: str) -> str:
    """The FIRST server ``role`` talks to -- what the page shows as this role's model."""
    return urls_for(role)[0]


# --------------------------------------------------------- spreading a role over replicas
#
# The counter is module-level and keyed by URL, not per client, because one process builds
# several executor clients (run_sim.py gives the Proposer and the MotionSupervisor one each)
# and two clients each counting only their own calls would both see an idle server and both
# pick it. The round-robin turn starts at this process's pid so that ten sweep_missions
# SUBPROCESSES -- they are processes, not threads, so they cannot see each other's in-flight
# counts -- do not all open on the same replica.
_PICK_LOCK = threading.Lock()
_IN_FLIGHT: Dict[str, int] = {}
_TURN = itertools.count(os.getpid())


@contextlib.contextmanager
def _busy(url: str):
    with _PICK_LOCK:
        _IN_FLIGHT[url] = _IN_FLIGHT.get(url, 0) + 1
    try:
        yield
    finally:
        with _PICK_LOCK:
            _IN_FLIGHT[url] = max(0, _IN_FLIGHT.get(url, 1) - 1)


def _least_busy_first(clients: Sequence[QwenClient]) -> List[QwenClient]:
    """The order to try the replicas in: fewest calls in flight from this process, then turn."""
    with _PICK_LOCK:
        turn = next(_TURN)
    ranked = sorted(enumerate(clients),
                    key=lambda pair: (_IN_FLIGHT.get(pair[1].url, 0),
                                      (pair[0] - turn) % len(clients)))
    return [client for _, client in ranked]


def _unreachable(reply: VlmReply) -> bool:
    """True when the server was never reached -- the one error worth trying a replica for."""
    return bool(reply is not None and reply.error.startswith("cannot reach the model")
                and "timed out" not in reply.error)


class ReplicatedClient:
    """A role served by several identical servers, used as if it were one client."""

    def __init__(self, clients: Sequence[QwenClient]):
        self.clients: List[QwenClient] = list(clients)

    def __getattr__(self, name: str):
        if name == "clients":                       # before __init__, or after an unpickle
            raise AttributeError(name)
        return getattr(self.clients[0], name)

    def ask_json(self, *args, **kwargs) -> VlmReply:
        """Ask the least busy replica, and on a transport failure ask the next one -- once."""
        reply = None
        for client in _least_busy_first(self.clients):
            with _busy(client.url):
                reply = client.ask_json(*args, **kwargs)
            if not _unreachable(reply):
                break
        return reply

    def health(self):
        """Reachable if ANY replica is: one server down is capacity lost, not a dead role."""
        checked = [client.health() for client in self.clients]
        return any(ok for ok, _ in checked), "; ".join(detail for _, detail in checked)


def _named(role: str, client: QwenClient) -> QwenClient:
    """Tag the client with whose it is, so a recorded call says which role made it."""
    client.role = role
    return client


def client_for(role: str, **kwargs):
    """A client pointed at ``role``'s server(s), sending ``role``'s picture size, asking for
    ``QWEN_<ROLE>_MODEL`` (or ``VLM_MODEL``) when one is set."""
    kwargs.setdefault("image_side", image_side_for(role))
    model = os.environ.get("QWEN_{}_MODEL".format(role.upper())) or os.environ.get("VLM_MODEL")
    if model:
        kwargs.setdefault("model", model)
    if "url" in kwargs:
        return _named(role, QwenClient(**kwargs))
    urls = urls_for(role)
    if len(urls) == 1:
        return _named(role, QwenClient(url=urls[0], **kwargs))
    return ReplicatedClient([_named(role, QwenClient(url=url, **kwargs)) for url in urls])
