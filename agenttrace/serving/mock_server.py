"""CPU-only OpenAI-compatible server for integration tests and demonstrations."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class MockServerConfig:
    host: str = "127.0.0.1"
    port: int = 8010
    first_token_delay_seconds: float = 0.01
    inter_chunk_delay_seconds: float = 0.005
    chunks: int = 3
    fail_every: int = 0
    model: str = "agenttrace-mock"


class MockState:
    def __init__(self) -> None:
        self.requests = 0
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()


def create_server(config: MockServerConfig) -> ThreadingHTTPServer:
    state = MockState()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._json(HTTPStatus.OK, {"object": "list", "data": [{"id": config.model}]})
            elif self.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/metrics":
                payload = (
                    "# TYPE vllm:num_requests_running gauge\n"
                    f"vllm:num_requests_running {state.active}\n"
                    "# TYPE vllm:num_requests_waiting gauge\n"
                    "vllm:num_requests_waiting 0\n"
                ).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                return
            with state.lock:
                state.requests += 1
                number = state.requests
                state.active += 1
                state.maximum_active = max(state.maximum_active, state.active)
            try:
                if config.fail_every and number % config.fail_every == 0:
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "simulated failure"})
                    return
                content = _response_content(body)
                if body.get("stream"):
                    self._stream(body, content)
                else:
                    self._completion(body, content)
            finally:
                with state.lock:
                    state.active -= 1

        def _stream(self, body: dict[str, Any], content: str) -> None:
            request_id = f"chatcmpl-mock-{uuid4().hex}"
            pieces = _split(content, config.chunks)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            time.sleep(config.first_token_delay_seconds)
            for piece in pieces:
                event = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": piece}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                time.sleep(config.inter_chunk_delay_seconds)
            usage = _usage(body, content)
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()

        def _completion(self, body: dict[str, Any], content: str) -> None:
            time.sleep(config.first_token_delay_seconds)
            self._json(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-mock-{uuid4().hex}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage(body, content),
                },
            )

        def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((config.host, config.port), Handler)
    server.mock_state = state  # type: ignore[attr-defined]
    return server


def serve_mock(config: MockServerConfig) -> None:
    server = create_server(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _response_content(body: dict[str, Any]) -> str:
    messages = body.get("messages", [])
    system = " ".join(
        str(message.get("content", "")) for message in messages if message.get("role") == "system"
    )
    if "bounded coding agent" in system.lower():
        assistant_count = sum(message.get("role") == "assistant" for message in messages)
        actions = [
            {"action": "tool", "tool": "read_file", "arguments": {"path": "calc.py"}},
            {
                "action": "tool",
                "tool": "replace_text",
                "arguments": {
                    "path": "calc.py",
                    "old": "return left - right",
                    "new": "return left + right",
                },
            },
            {"action": "tool", "tool": "run_command", "arguments": {"argv": ["pytest", "-q"]}},
            {"action": "finish", "summary": "fixed add and validated the test suite"},
        ]
        return json.dumps(actions[min(assistant_count, len(actions) - 1)], separators=(",", ":"))
    max_tokens = max(1, min(int(body.get("max_tokens", 8)), 128))
    return " ".join(f"mock{i}" for i in range(max_tokens))


def _usage(body: dict[str, Any], content: str) -> dict[str, int]:
    prompt = " ".join(str(message.get("content", "")) for message in body.get("messages", []))
    return {"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())}


def _split(content: str, chunks: int) -> list[str]:
    chunks = max(1, chunks)
    size = max(1, (len(content) + chunks - 1) // chunks)
    return [content[index : index + size] for index in range(0, len(content), size)]
