"""
2_mock_groq.py  —  local mock of Groq's /openai/v1/chat/completions
────────────────────────────────────────────────────────────────────
Mimics the real endpoint's SSE streaming format, x-ratelimit-* /
retry-after headers, and 429 error bodies — calibrated from the live
probe numbers:

    TTFT   192 / 390 / 1050 ms   (min / avg / max)
    total  322 / 600 / 1079 ms
    limits  30 RPM  |  8 000 TPM  |  1 000 RPD  |  200 000 TPD
    burst   1 concurrent before 429 (retry-after: 2 or 8)

Admin endpoints  (hit while the load runner is working):
    GET  /config                     read current config
    POST /config          {partial}  patch any field live
    POST /config/scenario/{name}     apply a preset
    GET  /stats                      counters + window state
    POST /reset                      clear windows & counters

Scenario presets:  default | rpm_halved | daily_exhausted | high_load | flaky

"""

import asyncio
import json
import random
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

# ── fake token vocabulary (plausible ML text, cycles for any length) ──────────

_VOCAB: list[str] = (
    "The transformer architecture uses self-attention to process sequences "
    "in parallel rather than sequentially enabling efficient training "
    "Multi-head attention allows the model to attend to different "
    "representation subspaces simultaneously capturing diverse relationships "
    "Positional encodings inject order information using sinusoidal functions "
    "Feed-forward sublayers apply two linear transformations with ReLU "
    "Residual connections and layer normalization stabilize deep training "
    "Dynamic chunking learns content-dependent segmentation end-to-end "
    "replacing handcrafted tokenization heuristics with learned boundaries "
    "Full-duplex dialogue enables simultaneous speaking and listening "
    "Inner monologue predicts text tokens before each audio token improving "
    "linguistic quality and streaming speech recognition as a byproduct "
    "Hierarchical networks model multiple abstraction levels improving "
    "data efficiency by nearly four times on DNA sequences and code "
    "Rate limiting uses sliding windows to enforce per-minute quotas "
    "Token buckets refill linearly at the configured tokens-per-second rate "
).split()


# ── config ────────────────────────────────────────────────────────────────────

class MockConfig(BaseModel):
    # rate limits — calibrated from live probe
    rpm_limit: int   = 30
    tpm_limit: int   = 8_000
    rpd_limit: int   = 1_000
    tpd_limit: int   = 200_000
    max_concurrent: int = 1          # burst: requests allowed simultaneously

    # latency (ms) — from probe data
    ttft_base_ms:    float = 390.0
    ttft_min_ms:     float = 192.0
    ttft_jitter_ms:  float = 180.0   # gaussian std
    total_base_ms:   float = 600.0
    total_min_ms:    float = 322.0
    total_jitter_ms: float = 150.0

    # nonlinear penalty per extra in-flight request beyond 1
    concurrency_ttft_penalty_ms:  float = 300.0
    concurrency_total_penalty_ms: float = 150.0

    # fault injection (all default off)
    slow_call_p:       float = 0.0   # P(3–6 s stall before first token)
    slow_stall_min_ms: float = 3_000.0
    slow_stall_max_ms: float = 6_000.0
    error_503_p:       float = 0.0   # P(immediate 503)
    hard_timeout_p:    float = 0.0   # P(stream starts, then stalls → drops)
    hard_timeout_ms:   float = 5_000.0

    # token estimation
    prompt_chars_per_token:  float = 4.0    # ~4 chars per BPE token
    output_tokens_min_frac:  float = 0.35   # fraction of max_tokens produced
    output_tokens_max_frac:  float = 0.90

    # retry-after (seconds) returned in 429 headers
    retry_after_concurrent_s: int = 8
    retry_after_rpm_s:        int = 2
    retry_after_tpm_s:        int = 1

    model: str = "qwen/qwen3.8-27b"


# ── scenario presets ──────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "default":         {},
    "rpm_halved":      {"rpm_limit": 15},
    "daily_exhausted": {"rpd_limit": 10, "tpd_limit": 5_000},   # near-daily-limit
    "high_load":       {"concurrency_ttft_penalty_ms": 500.0,
                        "concurrency_total_penalty_ms": 300.0},
    "flaky":           {"slow_call_p": 0.20, "error_503_p": 0.10,
                        "hard_timeout_p": 0.05},
}


# ── sliding-window rate limiters ──────────────────────────────────────────────

class _ReqWindow:
    """Sliding window that counts requests (each counts as 1)."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit  = limit
        self._window = window_s
        self._ts: deque[float] = deque()
        self._lock   = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()

    async def try_consume(self) -> tuple[bool, float]:
        """Returns (allowed, reset_seconds_until_a_slot_opens)."""
        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._ts) >= self._limit:
                return False, max(0.0, self._ts[0] + self._window - now)
            self._ts.append(now)
            return True, 0.0

    @property
    def remaining(self) -> int:
        self._prune(time.monotonic())
        return max(0, self._limit - len(self._ts))

    @property
    def reset_s(self) -> float:
        now = time.monotonic()
        self._prune(now)
        return max(0.0, self._ts[0] + self._window - now) if self._ts else 0.0

    @property
    def limit(self) -> int:
        return self._limit

    def set_limit(self, v: int) -> None:
        self._limit = v

    def clear(self) -> None:
        self._ts.clear()

    def prefill(self, n: int) -> None:
        """Pre-consume n slots, spread across the middle 40% of the window."""
        n   = min(n, self._limit)
        now = time.monotonic()
        self._ts.clear()
        for i in range(n):
            # oldest at –0.5 * window, newest at –0.1 * window
            frac = 0.1 + 0.4 * (1 - i / max(n - 1, 1))
            self._ts.append(now - self._window * frac)
        self._ts = deque(sorted(self._ts))


class _TokWindow:
    """Sliding window that counts tokens (weighted entries)."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit   = limit
        self._window  = window_s
        self._entries: deque[tuple[float, int]] = deque()  # (ts, tokens)
        self._lock    = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    async def try_consume(self, tokens: int) -> tuple[bool, float]:
        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            used = sum(e[1] for e in self._entries)
            if used + tokens > self._limit:
                reset_s = (
                    self._entries[0][0] + self._window - now
                    if self._entries else 0.0
                )
                return False, max(0.0, reset_s)
            self._entries.append((now, tokens))
            return True, 0.0

    @property
    def remaining(self) -> int:
        self._prune(time.monotonic())
        return max(0, self._limit - sum(e[1] for e in self._entries))

    @property
    def reset_s(self) -> float:
        now = time.monotonic()
        self._prune(now)
        return (
            max(0.0, self._entries[0][0] + self._window - now)
            if self._entries else 0.0
        )

    @property
    def limit(self) -> int:
        return self._limit

    def set_limit(self, v: int) -> None:
        self._limit = v

    def clear(self) -> None:
        self._entries.clear()

    def prefill_tokens(self, tokens: int) -> None:
        """Pre-consume tokens in a single entry at –0.1 * window."""
        now = time.monotonic()
        self._entries.clear()
        self._entries.append((now - self._window * 0.1, tokens))


# ── app state (one instance, lives for the process) ───────────────────────────

class AppState:
    def __init__(self) -> None:
        self.cfg       = MockConfig()
        self.rpm       = _ReqWindow(self.cfg.rpm_limit,  60.0)
        self.rpd       = _ReqWindow(self.cfg.rpd_limit,  86_400.0)
        self.tpm       = _TokWindow(self.cfg.tpm_limit,  60.0)
        self.tpd       = _TokWindow(self.cfg.tpd_limit,  86_400.0)
        self.in_flight = 0
        self._inf_lock = asyncio.Lock()
        self.stats: dict[str, int] = {
            "total_requests":   0,
            "ok_requests":      0,
            "rate_limited":     0,
            "injected_503":     0,
            "injected_timeout": 0,
            "injected_slow":    0,
        }

    # ── window management ─────────────────────────────────────────────────────

    def _rebuild_windows(self) -> None:
        """Propagate limit changes to existing window objects."""
        self.rpm.set_limit(self.cfg.rpm_limit)
        self.rpd.set_limit(self.cfg.rpd_limit)
        self.tpm.set_limit(self.cfg.tpm_limit)
        self.tpd.set_limit(self.cfg.tpd_limit)

    def apply_scenario(self, name: str) -> str:
        overrides = SCENARIOS.get(name)
        if overrides is None:
            raise KeyError(f"unknown scenario '{name}'. "
                           f"Available: {', '.join(SCENARIOS)}")
        self.cfg = MockConfig(**{**self.cfg.model_dump(), **overrides})
        self._rebuild_windows()
        if name == "daily_exhausted":
            self.rpd.prefill(self.cfg.rpd_limit - 2)
            self.tpd.prefill_tokens(self.cfg.tpd_limit - 500)
        return f"applied scenario '{name}'"

    def reset_all(self) -> None:
        self.rpm.clear(); self.rpd.clear()
        self.tpm.clear(); self.tpd.clear()
        self.in_flight = 0
        for k in self.stats:
            self.stats[k] = 0


# ── pure helpers ──────────────────────────────────────────────────────────────

def _fmt_reset(s: float) -> str:
    """Format seconds to Groq header style: '52.019s' or '1m26.4s'."""
    if s <= 0:
        return "0s"
    if s < 60:
        return f"{s:.3f}s"
    m = int(s // 60)
    return f"{m}m{s % 60:.1f}s"


def _estimate_prompt_tokens(body: dict, cfg: MockConfig) -> int:
    total_chars = sum(
        len(m.get("content") or "")
        for m in body.get("messages", [])
    )
    return max(1, int(total_chars / cfg.prompt_chars_per_token))


def _gen_words(n: int) -> list[str]:
    """Return n words from the cyclic vocabulary (each ≈ 1.3 BPE tokens)."""
    return [_VOCAB[i % len(_VOCAB)] for i in range(n)]


def _classify_call(body: dict) -> str:
    """
    Inspect message text to decide which pipeline step this is.
    Returns: 'router' | 'fix' | 'draft'
    """
    text = " ".join(
        (m.get("content") or "") for m in body.get("messages", [])
    ).lower()
    # _ROUTER_SYS contains "scope checker" and "in_scope / out_scope"
    if "scope checker" in text or "in_scope" in text or "out_scope" in text:
        return "router"
    # fix loop injects "[REVISION REQUIRED]" into the user message
    if "revision required" in text:
        return "fix"
    return "draft"


def _rl_headers(st: AppState) -> dict[str, str]:
    return {
        "x-ratelimit-limit-requests":     str(st.rpd.limit),
        "x-ratelimit-limit-tokens":       str(st.tpm.limit),
        "x-ratelimit-remaining-requests": str(st.rpd.remaining),
        "x-ratelimit-remaining-tokens":   str(st.tpm.remaining),
        "x-ratelimit-reset-requests":     _fmt_reset(st.rpd.reset_s),
        "x-ratelimit-reset-tokens":       _fmt_reset(st.tpm.reset_s),
    }


def _429_response(
    st: AppState, retry_after: int, kind: str,
) -> JSONResponse:
    cfg = st.cfg
    messages = {
        "concurrent": (
            f"Rate limit reached for model `{cfg.model}`: concurrent request "
            f"limit exceeded. Please try again in {retry_after}s."
        ),
        "rpm": (
            f"Rate limit reached for model `{cfg.model}`: {cfg.rpm_limit} requests "
            f"per minute. Please try again in {retry_after}s."
        ),
        "tpm": (
            f"Rate limit reached for model `{cfg.model}` on tokens per minute "
            f"(TPM): Limit {cfg.tpm_limit}. Please try again in {retry_after}s."
        ),
        "rpd": (
            f"Rate limit reached for model `{cfg.model}`: daily request limit "
            f"({cfg.rpd_limit}) reached."
        ),
        "tpd": (
            f"Rate limit reached for model `{cfg.model}`: daily token limit "
            f"({cfg.tpd_limit}) reached."
        ),
    }
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": messages.get(kind, "Rate limit exceeded."),
                "type":    kind,
                "code":    "rate_limit_exceeded",
            }
        },
        headers={**_rl_headers(st), "retry-after": str(retry_after)},
    )


# ── SSE stream generator ──────────────────────────────────────────────────────

async def _stream_sse(
    st:            AppState,
    body:          dict,
    req_id:        str,
    prompt_tokens: int,
    output_tokens: int,
    ttft_ms:       float,
    total_ms:      float,
    is_timeout:    bool,
) -> AsyncIterator[str]:
    """Yield SSE lines that look exactly like Groq's streaming format."""
    cfg     = st.cfg
    model   = body.get("model") or cfg.model
    created = int(time.time())
    include_usage = (body.get("stream_options") or {}).get("include_usage", False)

    def _chunk(delta: dict, finish_reason=None, usage=None) -> str:
        obj: dict = {
            "id":      req_id,
            "object":  "chat.completion.chunk",
            "created": created,
            "model":   model,
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                if (delta or finish_reason is not None)
                else []
            ),
        }
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n"

    # ── wait TTFT ─────────────────────────────────────────────────────────────
    await asyncio.sleep(ttft_ms / 1000)

    # ── hard-timeout injection ────────────────────────────────────────────────
    # Stream a few tokens, then stall — the client's read timeout fires.
    if is_timeout:
        yield _chunk({"role": "assistant", "content": ""})
        for word in _gen_words(4):
            await asyncio.sleep(0.08)
            yield _chunk({"content": word + " "})
        await asyncio.sleep(cfg.hard_timeout_ms / 1000)
        return   # generator exhausts → connection closes without [DONE]

    # ── normal stream ─────────────────────────────────────────────────────────
    yield _chunk({"role": "assistant", "content": ""})   # role chunk

    call_kind = _classify_call(body)
    if call_kind == "router":
        words = ["IN_SCOPE"]
    elif call_kind == "fix":
        # Return a corrected draft with a citation the verifier will accept.
        # The page number here is 4 — always present in the Transformer PDF.
        words = ["The", "Transformer", "uses", "self-attention",
                 "[1706.03762v7,", "p.4]", "for", "sequence", "modeling."]
    else:
        # Normal draft: use the vocab but inject a citation early so the
        # programmatic verifier always finds at least one grounded citation.
        draft_words = _gen_words(max(1, output_tokens - 6))
        words = (draft_words[:4]
                 + ["[1706.03762v7,", "p.4]"]
                 + draft_words[4:])
    remaining_ms   = max(20.0, total_ms - ttft_ms)
    inter_delay_ms = remaining_ms / len(words)

    for word in words:
        await asyncio.sleep(inter_delay_ms / 1000)
        yield _chunk({"content": word + " "})

    yield _chunk({}, finish_reason="stop")               # finish chunk

    if include_usage:                                    # usage chunk
        yield _chunk(
            {},
            usage={
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens":      prompt_tokens + output_tokens,
            },
        )

    yield "data: [DONE]\n\n"


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    st = AppState()
    app.state.st = st
    cfg = st.cfg
    print(
        f"[mock] ready  model={cfg.model}  "
        f"rpm={cfg.rpm_limit}  tpm={cfg.tpm_limit}  "
        f"rpd={cfg.rpd_limit}  max_concurrent={cfg.max_concurrent}"
    )
    yield


app = FastAPI(title="Mock Groq Server", lifespan=lifespan)


def _st(req: Request) -> AppState:
    return req.app.state.st


# ── /openai/v1/chat/completions ───────────────────────────────────────────────

@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    st  = _st(request)
    cfg = st.cfg
    st.stats["total_requests"] += 1

    # ── immediate 503 injection ───────────────────────────────────────────────
    if cfg.error_503_p > 0 and random.random() < cfg.error_503_p:
        st.stats["injected_503"] += 1
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": "Service temporarily unavailable (injected fault).",
                "code":    "service_unavailable",
            }},
        )

    # ── parse body ────────────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body.", "code": "invalid_request"}},
        )

    # ── concurrency gate (burst limit) ───────────────────────────────────────
    async with st._inf_lock:
        if st.in_flight >= cfg.max_concurrent:
            st.stats["rate_limited"] += 1
            return _429_response(st, cfg.retry_after_concurrent_s, "concurrent")
        st.in_flight += 1

    async def _release() -> None:
        async with st._inf_lock:
            st.in_flight -= 1

    # Every code path beyond here must call _release() exactly once.
    try:
        # ── token estimation ──────────────────────────────────────────────────
        prompt_tokens = _estimate_prompt_tokens(body, cfg)
        max_tokens    = int(body.get("max_tokens") or 256)
        output_tokens = random.randint(
            max(1, int(max_tokens * cfg.output_tokens_min_frac)),
            max(1, int(max_tokens * cfg.output_tokens_max_frac)),
        )
        total_tokens = prompt_tokens + output_tokens

        # ── sliding-window checks (RPM → TPM → RPD → TPD) ────────────────────
        ok, retry = await st.rpm.try_consume()
        if not ok:
            await _release(); st.stats["rate_limited"] += 1
            return _429_response(st, max(cfg.retry_after_rpm_s, int(retry) + 1), "rpm")

        ok, retry = await st.tpm.try_consume(total_tokens)
        if not ok:
            await _release(); st.stats["rate_limited"] += 1
            return _429_response(st, max(cfg.retry_after_tpm_s, int(retry) + 1), "tpm")

        ok, retry = await st.rpd.try_consume()
        if not ok:
            await _release(); st.stats["rate_limited"] += 1
            return _429_response(st, int(retry) + 1, "rpd")

        ok, retry = await st.tpd.try_consume(total_tokens)
        if not ok:
            await _release(); st.stats["rate_limited"] += 1
            return _429_response(st, int(retry) + 1, "tpd")

        # ── timing: apply nonlinear concurrency penalty ───────────────────────
        # in_flight already includes this request; extra = how many others
        extra = max(0, st.in_flight - 1)

        is_slow    = cfg.slow_call_p    > 0 and random.random() < cfg.slow_call_p
        is_timeout = cfg.hard_timeout_p > 0 and random.random() < cfg.hard_timeout_p

        if is_slow:
            st.stats["injected_slow"] += 1
            ttft_ms = random.uniform(cfg.slow_stall_min_ms, cfg.slow_stall_max_ms)
        else:
            ttft_ms = max(
                cfg.ttft_min_ms,
                random.gauss(
                    cfg.ttft_base_ms + extra * cfg.concurrency_ttft_penalty_ms,
                    cfg.ttft_jitter_ms,
                ),
            )

        total_ms = max(
            ttft_ms + 20.0,
            random.gauss(
                cfg.total_base_ms + extra * cfg.concurrency_total_penalty_ms,
                cfg.total_jitter_ms,
            ),
        )

        if is_timeout:
            st.stats["injected_timeout"] += 1

        req_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        hdrs   = _rl_headers(st)

        # ── streaming ─────────────────────────────────────────────────────────
        if body.get("stream", False):
            async def _streamer():
                try:
                    async for line in _stream_sse(
                        st, body, req_id,
                        prompt_tokens, output_tokens,
                        ttft_ms, total_ms, is_timeout,
                    ):
                        yield line
                finally:
                    await _release()   # always fires: normal end or disconnect

            st.stats["ok_requests"] += 1
            return StreamingResponse(
                _streamer(), media_type="text/event-stream", headers=hdrs,
            )

        # ── non-streaming ─────────────────────────────────────────────────────
        await asyncio.sleep(total_ms / 1000)
        await _release()
        st.stats["ok_requests"] += 1
        call_kind = _classify_call(body)
        if call_kind == "router":
            content = "IN_SCOPE"
        elif call_kind == "fix":
            content = "The Transformer uses self-attention [1706.03762v7, p.4] to process sequences in parallel."
        else:
            draft = _gen_words(max(1, output_tokens - 6))
            content = " ".join(draft[:4] + ["[1706.03762v7, p.4]"] + draft[4:])
        return JSONResponse(
            content={
                "id":      req_id,
                "object":  "chat.completion",
                "created": int(time.time()),
                "model":   body.get("model") or cfg.model,
                "choices": [{
                    "index":         0,
                    "message":       {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens":      total_tokens,
                },
            },
            headers=hdrs,
        )

    except Exception:
        await _release()
        raise


# ── admin: config ─────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config(request: Request):
    """Return the current mock configuration."""
    return _st(request).cfg.model_dump()


@app.post("/config")
async def patch_config(request: Request):
    """
    Partially update config fields.  Send only the keys you want to change:

        curl -X POST http://localhost:8001/config \\
             -H 'Content-Type: application/json' \\
             -d '{"rpm_limit": 15, "slow_call_p": 0.2}'
    """
    st = _st(request)
    try:
        patch = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    st.cfg = MockConfig(**{**st.cfg.model_dump(), **patch})
    st._rebuild_windows()
    return {"ok": True, "config": st.cfg.model_dump()}


@app.post("/config/scenario/{name}")
async def apply_scenario(name: str, request: Request):
    """
    Apply a named scenario preset.

    Available presets:
        default         — base calibrated values
        rpm_halved      — RPM cut to 15 (tests retry/backoff logic)
        daily_exhausted — RPD/TPD near zero (tests graceful degradation)
        high_load       — heavy concurrency penalties on TTFT/total
        flaky           — 20% slow calls, 10% 503s, 5% hard timeouts
    """
    st = _st(request)
    try:
        msg = st.apply_scenario(name)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    return {"ok": True, "message": msg, "config": st.cfg.model_dump()}


# ── admin: stats & reset ──────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats(request: Request):
    """Return request counters and current rate-limit window state."""
    st = _st(request)
    return {
        **st.stats,
        "in_flight":        st.in_flight,
        "rpm_remaining":    st.rpm.remaining,
        "tpm_remaining":    st.tpm.remaining,
        "rpd_remaining":    st.rpd.remaining,
        "tpd_remaining":    st.tpd.remaining,
        "rpm_reset_s":      round(st.rpm.reset_s, 3),
        "tpm_reset_s":      round(st.tpm.reset_s, 3),
        "rpd_reset_s":      round(st.rpd.reset_s, 3),
        "tpd_reset_s":      round(st.tpd.reset_s, 3),
    }


@app.post("/reset")
async def reset_state(request: Request):
    """Clear all rate-limit windows and stats counters (keeps config)."""
    _st(request).reset_all()
    return {"ok": True, "message": "rate-limit windows and stats cleared"}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("MOCK_PORT", 8001))
    print(f"[mock] starting on http://0.0.0.0:{port}")
    uvicorn.run("2_mock_groq:app", host="0.0.0.0", port=port, reload=False)