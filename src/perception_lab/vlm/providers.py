"""Local VLM runtimes behind one small interface, using only the standard library.

* :class:`OllamaProvider`: Ollama's native ``/api/chat`` with base64 ``images`` and a JSON
  schema in ``format``.
* :class:`OpenAICompatProvider`: any OpenAI-compatible ``/v1/chat/completions`` endpoint
  (for example llama.cpp ``llama-server``) with ``image_url`` data URIs and
  ``response_format``.

Whether a given runtime, version and model accept several images per request or honour the
schema must be checked on the target machine (``lab bench vlm``); nothing here assumes it.
Endpoints must be loopback addresses unless ``allow_remote`` is set.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import cv2
from numpy.typing import NDArray

from perception_lab.core.config import VLMConfig

PostJSON = Callable[[str, dict[str, Any], float], dict[str, Any]]


class ProviderError(RuntimeError):
    """Raised when the runtime cannot be reached or returns an unexpected response."""


@dataclass
class ProviderReply:
    text: str
    latency_s: float
    meta: dict[str, Any] = field(default_factory=dict)


class VLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self, prompt: str, images_jpeg: Sequence[bytes], schema: dict[str, Any]
    ) -> ProviderReply: ...


def require_loopback(url: str, allow_remote: bool = False) -> None:
    host = urlparse(url).hostname or ""
    if allow_remote:
        return
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ProviderError(
        f"refusing non-loopback endpoint {url!r}; set allow_remote only for a machine you control"
    )


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response from {url}") from exc


def encode_jpeg(image: NDArray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image as JPEG")
    return buffer.tobytes()


class OllamaProvider:
    name = "ollama"

    def __init__(self, config: VLMConfig, post: PostJSON = post_json) -> None:
        require_loopback(config.base_url, config.allow_remote)
        self.config = config
        self.model = config.model
        self._post = post

    def complete(self, prompt, images_jpeg, schema):
        cfg = self.config
        payload = {
            "model": cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(b).decode() for b in images_jpeg],
                }
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": cfg.temperature},
            "keep_alive": cfg.keep_alive,
        }
        start = time.perf_counter()
        data = self._post(cfg.base_url.rstrip("/") + "/api/chat", payload, cfg.timeout_s)
        latency = time.perf_counter() - start
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected Ollama response: {str(data)[:200]}") from exc
        meta = {
            k: data[k]
            for k in ("total_duration", "load_duration", "prompt_eval_count", "eval_count")
            if k in data
        }
        return ProviderReply(text, latency, meta)


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, config: VLMConfig, post: PostJSON = post_json) -> None:
        require_loopback(config.base_url, config.allow_remote)
        self.config = config
        self.model = config.model
        self._post = post

    def complete(self, prompt, images_jpeg, schema):
        cfg = self.config
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b in images_jpeg:
            url = "data:image/jpeg;base64," + base64.b64encode(b).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": cfg.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "pet_activity", "schema": schema},
            },
        }
        base = cfg.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        start = time.perf_counter()
        data = self._post(url, payload, cfg.timeout_s)
        latency = time.perf_counter() - start
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response: {str(data)[:200]}") from exc
        return ProviderReply(text, latency, {"usage": data.get("usage", {})})


def create_provider(config: VLMConfig, post: PostJSON = post_json) -> VLMProvider:
    if config.provider == "ollama":
        return OllamaProvider(config, post)
    return OpenAICompatProvider(config, post)
