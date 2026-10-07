"""Talking to a Qwen3.8-Next-Flash server, and nothing else."""

from __future__ import annotations

import base64
import io
import json
import re
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests

DEFAULT_URL = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "qwen"
#: A hosted, OpenAI-style endpoint is reached with a key, read from the file this names (never
#: from the repository). With no key the server is taken to be a local SGLang one.
KEY_FILE_VAR = "VLM_API_KEY_FILE"
#: A reasoning model spends part of its output budget thinking; this much more is allowed.
REASONING_TOKENS = int(os.environ.get("VLM_REASONING_TOKENS", "4000"))
REASONING_EFFORT = os.environ.get("VLM_REASONING_EFFORT", "low")
MIN_OUTPUT_TOKENS = 1500


@dataclass
class VlmReply:
    """One answer, with enough about how it arrived to spot a degradation."""

    data: Optional[Dict[str, Any]]
    text: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    salvaged_from: str = ""          # "" | "fenced" | "embedded" | "reasoning_content"
    error: str = ""
    finish_reason: str = ""          # "length" is an answer cut off by the output budget

    @property
    def ok(self) -> bool:
        return self.data is not None

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "latency_s": round(self.latency_s, 2),
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "salvaged_from": self.salvaged_from, "error": self.error,
                "finish_reason": self.finish_reason, "text": self.text[:400]}


def encode_image_bytes(rgb: np.ndarray, side: int = 512, quality: int = 90) -> bytes:
    """RGB array -> the JPEG bytes that go on the wire, DOWNSCALED to ``side`` on its longest
    edge first. This, not the render, is what the model sees."""
    from PIL import Image
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if side and max(image.size) > side:
        scale = side / float(max(image.size))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                             Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


#: Set to ``recorder.on_call`` to have every call of every role written out whole -- prompt,
#: reply, tokens, seconds and the pictures above. One process is one mission (run_mission.py
#: is a subprocess per mission), so this being module-wide is what makes it catch every role.
#: Called AFTER the reply is back, so it cannot add a millisecond to the answer.
RECORDER = None


# Sampling temperature for every call.
DEFAULT_TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0"))


# How often one ``ask_json`` may put the SAME request on the wire, and how long it waits in
# between.
MAX_TRIES = 3
RETRY_BACKOFF_S = (2.0, 4.0, 8.0)


def _worth_retrying(exc: Optional[BaseException], status: Optional[int]) -> str:
    """Why this attempt should be put on the wire again, or "" for leave it alone."""
    if exc is not None:
        return "{}: {}".format(type(exc).__name__, exc)
    if status is not None and status >= 500:
        return "HTTP {}".format(status)
    return ""


class QwenClient:
    #: Every ask is a request of its own, so several may be out at once.
    concurrent = True

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 timeout_s: float = 60.0, image_side: int = 512,   # see endpoints.IMAGE_SIDES
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tries: int = MAX_TRIES,
                 backoff_s: Sequence[float] = RETRY_BACKOFF_S,
                 post=None, sleep=None):
        self.url = url.rstrip("/")
        self.model = model
        key_file = os.environ.get(KEY_FILE_VAR, "")
        self._headers = ({"Authorization": "Bearer " + open(key_file).read().strip()}
                         if key_file else {})
        self.timeout_s = timeout_s if not self._headers else max(timeout_s, 180.0)
        self.image_side = image_side
        self.temperature = temperature
        self.max_tries = max(1, int(max_tries))
        self.backoff_s = tuple(float(s) for s in backoff_s)
        # Injected so a test can drive the retry without a server and without reaching into
        # `requests`; nothing in the code base passes them.
        self._post = post or (lambda url, **kwargs: requests.post(url, **kwargs))
        self._sleep = sleep or time.sleep
        #: Which role's client this is, for the recording. Set by :func:`endpoints.client_for`.
        self.role = ""
        self.calls = 0
        self.failures = 0
        self.retries = 0                 # transport failures put back on the wire

    # ------------------------------------------------------------------ health

    def health(self):
        try:
            reply = requests.get(self.url + "/models", headers=self._headers, timeout=10)
            reply.raise_for_status()
            names = [m.get("id") for m in reply.json().get("data", [])]
        except Exception as exc:
            return False, "cannot reach {}: {}".format(self.url, exc)
        if self.model not in names:
            return False, "{} does not serve {!r}; it has {}".format(self.url, self.model, names)
        return True, "{} serving {}".format(self.url, self.model)

    # ------------------------------------------------------------------ the wire

    def _post_with_retries(self, payload: Dict[str, Any]):
        """One request, up to :attr:`max_tries` times."""
        last_error = "the request was never sent"
        for attempt in range(self.max_tries):
            response, exc, status = None, None, None
            try:
                response = self._post(self.url + "/chat/completions", json=payload,
                                      headers=self._headers, timeout=self.timeout_s)
                status = response.status_code
            except requests.RequestException as problem:   # aborted, refused, timed out
                exc = problem
            except Exception as problem:                   # not the transport: do not repeat
                return None, "{}: {}".format(type(problem).__name__, problem)
            why = _worth_retrying(exc, status)
            if not why:
                return response, ""
            last_error = why
            if attempt + 1 >= self.max_tries:
                break
            wait = self.backoff_s[min(attempt, len(self.backoff_s) - 1)] if self.backoff_s \
                else 0.0
            self.retries += 1
            print("qwen retry {}/{} to {} after {} -- waiting {:.0f} s".format(
                attempt + 1, self.max_tries - 1, self.url, why, wait),
                file=sys.stderr, flush=True)
            self._sleep(wait)
        if response is not None:
            return response, ""          # a 5xx that outlasted the retries is still an answer
        return None, "{} (gave up after {} tries)".format(last_error, self.max_tries)

    # ------------------------------------------------------------------ the call

    def ask_json(self, prompt: str, images: Sequence[np.ndarray] = (),
                 system: Optional[str] = None, max_tokens: int = MIN_OUTPUT_TOKENS,
                 temperature: Optional[float] = None) -> VlmReply:
        """Ask for one JSON object. Never raises on a bad answer -- it reports one."""
        blobs = [encode_image_bytes(im, self.image_side) for im in images]
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64,"
                                          + base64.b64encode(blob).decode("ascii")}}
                    for blob in blobs]
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": content}]
        payload = {"model": self.model, "messages": messages,
                   "response_format": {"type": "json_object"}}
        if self._headers:        # a hosted reasoning model: its own budget names, no sampling knobs
            payload.update(max_completion_tokens=max(MIN_OUTPUT_TOKENS, int(max_tokens))
                           + REASONING_TOKENS, reasoning_effort=REASONING_EFFORT)
        else:                    # local SGLang: thinking on means prose and no JSON (measured)
            payload.update(max_tokens=max(MIN_OUTPUT_TOKENS, int(max_tokens)),
                           temperature=self.temperature if temperature is None else temperature,
                           chat_template_kwargs={"enable_thinking": False})

        started = time.monotonic()
        self.calls += 1
        response, transport = self._post_with_retries(payload)
        latency = time.monotonic() - started
        if response is None:
            self.failures += 1
            return self._recorded(VlmReply(
                data=None, latency_s=latency,
                error="cannot reach the model: {}".format(transport)), prompt, system, blobs)
        if response.status_code != 200:
            self.failures += 1
            return self._recorded(VlmReply(
                data=None, latency_s=latency,
                error="HTTP {}: {}".format(response.status_code, response.text[:300])),
                prompt, system, blobs)

        body = response.json()
        first = (body.get("choices") or [{}])[0]
        choice = first.get("message", {})
        usage = body.get("usage") or {}
        text = choice.get("content") or ""
        source = ""
        if not text.strip():
            # the JSON sometimes arrives here instead, with content empty
            text = choice.get("reasoning_content") or ""
            source = "reasoning_content" if text.strip() else ""

        data, how = _parse_json(text)
        if data is None:
            self.failures += 1
        return self._recorded(VlmReply(
            data=data, text=text, latency_s=latency,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            # SGLang reports it at the top of usage, OpenAI under completion_tokens_details.
            reasoning_tokens=int(usage.get("reasoning_tokens") or (usage.get(
                "completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            salvaged_from=source or how, finish_reason=str(first.get("finish_reason") or ""),
            error="" if data is not None else "no JSON object in the reply"),
            prompt, system, blobs)

    def _recorded(self, reply: VlmReply, prompt: str, system: Optional[str],
                  blobs: Sequence[bytes]) -> VlmReply:
        """Hand the call to :data:`RECORDER` if one is set, and return the reply unchanged.
        A recorder that throws is a lost record, never a lost answer."""
        if RECORDER is not None:
            try:
                RECORDER(self.role, self.url, prompt, system, blobs, reply, reply.latency_s)
            except Exception as exc:
                print("recording a {} call failed: {}".format(self.role or "?", exc),
                      file=sys.stderr, flush=True)
        return reply


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _parse_json(text: str):
    """Straight parse, then a fenced block, then the first balanced object in the prose."""
    text = (text or "").strip()
    if not text:
        return None, ""
    try:
        parsed = json.loads(text)
        return (parsed, "") if isinstance(parsed, dict) else (None, "")
    except ValueError:
        pass
    fenced = _FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "fenced"
        except ValueError:
            pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1]), "embedded"
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None, ""
