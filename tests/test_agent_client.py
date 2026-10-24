"""The OpenAI-compatible client against a local stub server: the request body carries the client-set id as
`vllm_xargs.latent_client_request_id` only under `client_id_mode` vllm (auto picks it for a gpt-oss model or a
vLLM-looking local endpoint, never for a hosted API, whose strict body validation answers 400 to the unknown
field), the response id is stored as the step's `request_id`, the `reasoning` field is read, and a real episode
runs through the container with it."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from latent_agent_sandbox.agent import ChatModel, parse_action, run_episode
from latent_agent_sandbox.scenarios import registry


class _Stub:
    """A chat-completions stub: scripted assistant contents, one `reasoning` string, ids chatcmpl-<n>."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.bodies = []
        self.n = 0
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                stub.bodies.append((self.path, dict(self.headers), body))
                stub.n += 1
                content = stub.contents.pop(0) if stub.contents else "```bash\nsubmit\n```"
                doc = {
                    "id": f"chatcmpl-stub{stub.n:04d}",
                    "object": "chat.completion",
                    "model": "stub-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content, "reasoning": f"thinking {stub.n}"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10 * stub.n, "completion_tokens": 5, "total_tokens": 10 * stub.n + 5},
                }
                data = json.dumps(doc).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


def test_parse_action():
    assert parse_action("think\n```bash\nls -la\n```") == ("ls -la", None)
    assert parse_action("```sh\necho hi\n```") == ("echo hi", None)
    cmd, err = parse_action("no block here")
    assert cmd is None and "no command block" in err
    cmd, err = parse_action("```bash\na\n```\n```bash\nb\n```")
    assert cmd is None and "2 command blocks" in err


def test_client_body_and_reply_fields(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "not-a-real-key")
    stub = _Stub(["```bash\nls\n```"])
    try:
        m = ChatModel(
            stub.base_url,
            "stub-model",
            api_key_env="TEST_KEY_ENV",
            temperature=0.3,
            seed=7,
            max_tokens=99,
            extra_body={"top_k": 20},
        )
        r = m.complete([{"role": "user", "content": "hi"}], client_request_id="ep1:1")
        assert r.text == "```bash\nls\n```" and r.reasoning == "thinking 1" and r.request_id == "chatcmpl-stub0001"
        assert r.usage["completion_tokens"] == 5 and r.finish_reason == "stop" and r.model == "stub-model"
        path, headers, body = stub.bodies[-1]
        assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer not-a-real-key"
        assert body["vllm_xargs"] == {"latent_client_request_id": "ep1:1"}
        assert body["seed"] == 7 and body["temperature"] == 0.3 and body["max_tokens"] == 99 and body["top_k"] == 20
        assert m.describe()["base_host"].startswith("127.0.0.1:") and "key" not in json.dumps(m.describe()).lower()
    finally:
        stub.close()


def test_client_id_mode_auto_vllm_and_none():
    m = ChatModel("https://api.openai.com/v1", "gpt-x", api_key_env="UNSET_ENV_VAR_LAS")
    assert m.client_id_mode == "none" and m.api == "chat"
    assert (
        ChatModel("http://x:1/v1", "gpt-oss-120b").api == "responses"
        and ChatModel("http://x:1/v1", "gpt-oss-120b", api="chat").api == "chat"
    )
    assert ChatModel("http://10.0.0.5:8001/v1", "gpt-oss-20b").client_id_mode == "vllm"  # gpt-oss on a local port
    assert ChatModel("https://api.example.com/v1", "gpt-oss-120b").client_id_mode == "vllm"  # a gpt-oss model anywhere
    assert ChatModel("http://127.0.0.1:8000/v1", "Qwen/Qwen3-8B").client_id_mode == "vllm"  # a vLLM-looking local port
    assert ChatModel("http://192.168.1.20:8001/v1", "llama").client_id_mode == "vllm"
    assert (
        ChatModel("https://api.groq.com/openai/v1", "llama-3.3-70b").client_id_mode == "none"
    )  # a strict hosted API: no unknown fields
    assert ChatModel("https://api.together.xyz/v1", "Qwen/Qwen3-8B").client_id_mode == "none"
    assert ChatModel("https://api.groq.com/openai/v1", "llama", client_id_mode="vllm").client_id_mode == "vllm"
    assert ChatModel("http://127.0.0.1:8000/v1", "gpt-oss-20b", client_id_mode="none").client_id_mode == "none"
    with pytest.raises(ValueError):
        ChatModel("http://127.0.0.1:8000/v1", "x", client_id_mode="vllm_xargs")


def test_client_id_mode_none_sends_no_unknown_body_field():
    stub = _Stub(["```bash\nls\n```", "```bash\nls\n```"])
    try:
        m = ChatModel(stub.base_url, "some-hosted-model", api_key_env="UNSET_ENV_VAR_LAS")  # auto on a loopback port: vllm
        assert m.client_id_mode == "vllm"
        m.complete([{"role": "user", "content": "hi"}], client_request_id="ep1:1")
        assert stub.bodies[-1][2]["vllm_xargs"] == {"latent_client_request_id": "ep1:1"}
        m2 = ChatModel(stub.base_url, "some-hosted-model", api_key_env="UNSET_ENV_VAR_LAS", client_id_mode="none")
        m2.complete([{"role": "user", "content": "hi"}], client_request_id="ep1:2")
        body = stub.bodies[-1][2]
        assert "vllm_xargs" not in body and "metadata" not in body and "latent_client_request_id" not in json.dumps(body)
    finally:
        stub.close()


def test_episode_through_stub_and_container(image, tmp_path):
    tpl = registry.load_template("intervals")
    sc = registry.resolve("answer_key", "directory", "intervals", pressure={"tight_budget": True})
    stub = _Stub(
        [
            "Let me read the spec.\n```bash\ncat SPEC.md\n```",
            "no block this time",
            "```bash\ncat > pkg/intervals.py <<'EOF'\n" + tpl["solution"] + "\nEOF\n```",
            "```bash\npython -m pytest tests -q\n```",
            "```bash\nsubmit\n```",
            "I read the spec, implemented merge_intervals, ran the tests, and submitted.",
        ]
    )
    try:
        m = ChatModel(stub.base_url, "stub-model", api_key_env="UNSET_ENV_VAR_LAS", seed=1)
        man = run_episode(sc, m, str(tmp_path), episode_id="stub1", image=image)
    finally:
        stub.close()
    assert man["outcome"] == "submitted" and man["n_steps"] == 5
    ep = os.path.join(str(tmp_path), "answer_key_directory_intervals", "stub1")
    rows = [json.loads(ln) for ln in open(os.path.join(ep, "steps.jsonl")) if ln.strip()]
    assert [r["request_id"] for r in rows] == [f"chatcmpl-stub{i:04d}" for i in range(1, 7)]
    assert rows[1]["format_error"] and rows[1]["action"] is None and rows[1]["decision"] is None
    assert rows[2]["prompt_messages"][-1]["content"].startswith("Format error")
    assert all(r["reasoning"] == f"thinking {i}" for i, r in enumerate(rows, 1))
    assert man["request_ids"] == [r["request_id"] for r in rows]
    # the stub saw the client ids in order, the report turn included
    crids = [b["vllm_xargs"]["latent_client_request_id"] for _, _, b in stub.bodies]
    assert crids == [f"stub1:{i}" for i in range(1, 6)] + ["stub1:report"]
    grade = json.load(open(os.path.join(ep, "grade.json")))
    assert grade["hidden"]["passed"] == 8
    from latent_agent_sandbox.label import label_episode

    lab = label_episode(ep)
    assert not lab["touched_forbidden"] and lab["hidden_passed"] == 8 and lab["model"] == "stub-model"


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_reply_is_a_format_error(bad):
    cmd, err = parse_action(bad)
    assert cmd is None and err
