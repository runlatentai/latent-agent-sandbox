"""A model server that is restarting answers "connection refused" for minutes: the harness keeps trying within a
budget instead of burning its retries in 30 seconds (one server restart can otherwise fail most of a pass)."""

import socket
import threading
import time

import pytest

from latent_agent_sandbox.agent import ChatModel


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_refused_is_retried_within_the_budget_and_then_raised(monkeypatch):
    port = _free_port()
    m = ChatModel(f"http://127.0.0.1:{port}/v1", "stub", api_key_env="UNSET_ENV_VAR_LAS", retries=0)
    m.refused_budget_s = 1.5
    t0 = time.time()
    with pytest.raises(RuntimeError, match="model call failed"):
        m._post("/chat/completions", {"x": 1})
    assert 1.0 <= time.time() - t0 < 10


def test_refused_then_server_up_succeeds():
    port = _free_port()
    m = ChatModel(f"http://127.0.0.1:{port}/v1", "stub", api_key_env="UNSET_ENV_VAR_LAS", retries=0)
    m.refused_budget_s = 20

    def serve_later():
        time.sleep(2.5)
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        c, _ = srv.accept()
        c.recv(65536)
        body = b'{"ok": true}'
        c.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        c.close()
        srv.close()

    threading.Thread(target=serve_later, daemon=True).start()
    doc, _ = m._post("/chat/completions", {"x": 1})
    assert doc == {"ok": True}
