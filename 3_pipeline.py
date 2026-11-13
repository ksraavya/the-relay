#!/usr/bin/env python3
"""
3_pipeline.py  —  The Relay: 4-step async RAG pipeline
════════════════════════════════════════════════════════════════════════════════
Steps per request
─────────────────
  0  pre-flight   quota check → load-shed decision
  1  router       LLM scope check (skipped when load-shedding)
  2  retrieve     BM25 over corpus.json (3 configurable settings)
  3  draft        streaming LLM with [paper_id, p.N] citations, TTFT measured
  4  verify/fix   programmatic citation check → one loop-back if bad + patience

Rate-limit handling
───────────────────
  • Primary model qwen/qwen3.8-27b  → fallback openai/gpt-oss-20b on first 429
    (gpt-oss-20b prefixes reasoning tokens in delta.reasoning; TTFT is measured
     from the first delta.content token, not from delta.reasoning)
  • Retry policies: "smart" (exponential backoff + jitter, honours Retry-After)
                   "aggressive" (immediate retry — tests spiral degradation)
                   "none" (fail on first error — raw failure-rate baseline)
  • Load-shed trigger: remaining RPD or TPM < shed_threshold (default 20%)
    → skip router + verifier, shrink retrieval to k=2

Usage
─────
  python 3_pipeline.py --question "What BLEU score did the Transformer achieve?"
  python 3_pipeline.py --all                        # all 18 from questions.json
  python 3_pipeline.py --all --source mock          # point at local mock server
  python 3_pipeline.py --all --retry aggressive --scenario burst_te
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

# ─────────────────────────────── constants ────────────────────────────────────

PRIMARY_MODEL  = "qwen/qwen3.8-27b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

# minimum patience (ms) required to even attempt each step
_BUDGET_ROUTER = 1_800
_BUDGET_DRAFT  = 2_500
_BUDGET_VERIFY = 2_500   # spec: skip verifier if < 2.5 s left
_BUDGET_FIX    = 2_500   # spec: skip fix loop  if < 2.5 s left

_ROUTER_MAX_TOKENS = 10
_DRAFT_MAX_TOKENS  = 512
_FIX_MAX_TOKENS    = 512

CITATION_RE = re.compile(r"\[([^,\]]+),\s*p\.(\d+)\]")

# columns written to results.csv (in order)
CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
    # timing breakdown
    "queue_wait_ms", "retry_wait_ms", "pre_ttft_ms", "post_ttft_ms",
    # extra context columns (useful for slow-request analysis)
    "steps", "load_shed", "fallback_used", "refusal_reason",
]


# ─────────────────────────────── config ───────────────────────────────────────

@dataclass
class PipelineConfig:
    """All pipeline knobs in one place.  __post_init__ fills blanks from env."""

    base_url: str = ""          # resolved from GROQ_BASE_URL or real Groq
    api_key:  str = ""          # resolved from GROQ_API_KEY

    primary_model:  str = PRIMARY_MODEL
    fallback_model: str = FALLBACK_MODEL

    # one of: small_k3_norerank | med_k5_norerank | med_k8_rerank3
    retrieval_setting: str = "med_k5_norerank"

    patience_ms: float = 8_000.0   # hard per-request wall-clock budget

    # "smart"      — exponential backoff with jitter; respects Retry-After
    # "aggressive" — immediate retry (tests spiral degradation)
    # "none"       — fail on first error (raw failure-rate baseline)
    retry_policy:        str   = "smart"
    max_retries:         int   = 3
    backoff_base_ms:     float = 500.0
    backoff_max_ms:      float = 5_000.0
    backoff_jitter_frac: float = 0.25

    # shed load when any remaining-* fraction drops below this
    shed_threshold: float = 0.20

    corpus_path: str = "corpus.json"
    results_csv:  str = "results.csv"

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get(
                "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
            ).rstrip("/")
        if not self.api_key:
            self.api_key = os.environ.get("GROQ_API_KEY", "")

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def source(self) -> str:
        url = self.base_url.lower()
        return "mock" if ("localhost" in url or "127.0.0.1" in url) else "groq"


# ─────────────────────────────── exceptions ───────────────────────────────────

class RateLimitError(Exception):
    pass

class DailyQuotaError(Exception):
    pass

class PipelineAPIError(Exception):
    pass


# ───────────────────────── rate-limit state ───────────────────────────────────

@dataclass
class RLState:
    """
    Accumulates x-ratelimit-* header values across LLM calls.

    Instantiate once per session and pass to every run_question call so
    quota info (especially RPD) improves load-shedding decisions over time.
    Per-request counters (retries, wait time) are reset each call.
    """
    # from x-ratelimit-* response headers
    limit_requests:     int | None = None   # RPD limit  (1 000)
    remaining_requests: int | None = None   # RPD remaining
    limit_tokens:       int | None = None   # TPM limit  (8 000)
    remaining_tokens:   int | None = None   # TPM remaining

    retry_after_s:  float = 0.0
    daily_exhausted: bool = False

    # per-request accumulators — reset by reset_per_request()
    total_retry_wait_ms: float = 0.0
    total_retries:       int   = 0

    def update(self, headers: httpx.Headers) -> None:
        def _int(name: str) -> int | None:
            v = headers.get(name)
            return int(v) if v is not None else None

        lr = _int("x-ratelimit-limit-requests")
        rr = _int("x-ratelimit-remaining-requests")
        lt = _int("x-ratelimit-limit-tokens")
        rt = _int("x-ratelimit-remaining-tokens")
        ra = headers.get("retry-after")

        if lr is not None: self.limit_requests     = lr
        if rr is not None: self.remaining_requests = rr
        if lt is not None: self.limit_tokens        = lt
        if rt is not None: self.remaining_tokens    = rt
        if ra is not None: self.retry_after_s       = float(ra)

        if self.remaining_requests == 0:
            self.daily_exhausted = True

    def should_shed_load(self, threshold: float) -> bool:
        if (self.remaining_requests is not None and self.limit_requests
                and self.remaining_requests / self.limit_requests < threshold):
            return True
        if (self.remaining_tokens is not None and self.limit_tokens
                and self.remaining_tokens / self.limit_tokens < threshold):
            return True
        return False

    def reset_per_request(self) -> None:
        self.total_retry_wait_ms = 0.0
        self.total_retries       = 0


# ─────────────────────────────── BM25 index ───────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _chunk_page(
    paper_id: str, page: int, text: str,
    chunk_w: int, overlap_w: int,
) -> list[dict]:
    words = text.split()
    step  = max(1, chunk_w - overlap_w)
    out: list[dict] = []
    start = 0
    while start < len(words):
        w = words[start: start + chunk_w]
        if w:
            out.append({"paper_id": paper_id, "page": page,
                        "text": " ".join(w)})
        if start + chunk_w >= len(words):
            break
        start += step
    return out


class BM25Index:
    """Lazy-builds per-setting chunk indices from corpus.json."""

    # (chunk_words, overlap_words, fetch_k, rerank_to)  rerank_to=0 → no rerank
    _SETTINGS: dict[str, tuple[int, int, int, int]] = {
        "small_k3_norerank": (200,  50, 3, 0),
        "med_k5_norerank":   (450, 100, 5, 0),
        "med_k8_rerank3":    (450, 100, 8, 3),
        "_shed_k2":          (200,  50, 2, 0),   # internal: load-shed mode
    }

    def __init__(self, corpus_path: str = "corpus.json") -> None:
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(
                f"corpus.json not found at '{path}' — "
                "run 1_probe_and_ingest.py first."
            )
        self._pages: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        self._cache: dict[str, tuple[list[dict], BM25Okapi]] = {}
        print(f"[bm25] loaded {len(self._pages)} pages from {corpus_path}")

    def _build(self, setting: str) -> tuple[list[dict], BM25Okapi]:
        if setting in self._cache:
            return self._cache[setting]
        chunk_w, overlap_w, _, _ = self._SETTINGS[setting]
        chunks: list[dict] = []
        for p in self._pages:
            chunks.extend(
                _chunk_page(p["paper_id"], p["page"], p["text"],
                            chunk_w, overlap_w)
            )
        index = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        self._cache[setting] = (chunks, index)
        print(f"[bm25] built index '{setting}': {len(chunks)} chunks")
        return chunks, index

    def retrieve(
        self, query: str, setting: str, *, shed: bool = False,
    ) -> list[dict]:
        s = "_shed_k2" if shed else setting
        chunks, index = self._build(s)
        _, _, fetch_k, rerank_to = self._SETTINGS[s]

        scores  = index.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:fetch_k]
        top     = [chunks[i] for i in top_idx]

        if rerank_to:
            top = _rerank(query, top, rerank_to)
        return top


def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    q_tok = _tokenize(query)
    q_uni = set(q_tok)
    q_bi  = _bigrams(q_tok)

    def score(c: dict) -> float:
        c_tok = _tokenize(c["text"])
        uni   = len(q_uni & set(c_tok)) / max(len(q_uni), 1)
        bi    = (len(q_bi & _bigrams(c_tok)) / max(len(q_bi), 1)) if q_bi else 0.0
        return 0.6 * uni + 0.4 * bi

    return sorted(chunks, key=score, reverse=True)[:top_k]


# ─────────────────────────────── prompts ──────────────────────────────────────

_ROUTER_SYS = """\
Scope checker for a 3-paper knowledge base:
  1706.03762v7  "Attention Is All You Need" — Transformer architecture (2017)
  2410.00037v1  "Moshi" — real-time full-duplex speech-text LLM (2024)
  2507.07955v2  "H-Net / Dynamic Chunking" — end-to-end sequence modeling (2025)

Reply with exactly one of:  IN_SCOPE  OUT_SCOPE
No other text."""

_DRAFTER_SYS = """\
You are a precise research assistant. Answer using ONLY the context chunks provided.
1. Cite every factual claim with [paper_id, p.N]  e.g. [1706.03762v7, p.4]
2. If context is insufficient, reply exactly: CANNOT_ANSWER
3. Be concise. Do not add information not present in the context."""


def _drafter_user(
    question: str, chunks: list[dict], *, fix_hint: str = "",
) -> str:
    ctx = "\n\n---\n\n".join(
        f"[{c['paper_id']}, p.{c['page']}]\n{c['text'][:800]}"
        for c in chunks
    )
    body = f"Context:\n{ctx}\n\nQuestion: {question}"
    if fix_hint:
        body += (
            f"\n\n[REVISION REQUIRED] Previous citation issues: {fix_hint}\n"
            "Rewrite, citing only pages shown in the context above."
        )
    return body


# ─────────────────────────────── backoff ──────────────────────────────────────

def _backoff_ms(
    attempt: int, cfg: PipelineConfig, retry_after_s: float,
) -> float:
    """
    Return sleep in ms for retry number `attempt` (1-indexed).
    smart      — exponential + jitter, respects Retry-After header
    aggressive — 0 ms (immediate; designed to show spiral degradation)
    none       — 0 ms (caller should raise before retrying)
    """
    if cfg.retry_policy in ("aggressive", "none"):
        return 0.0
    base = (
        retry_after_s * 1_000 if retry_after_s > 0
        else cfg.backoff_base_ms * (2 ** (attempt - 1))
    )
    jitter = base * random.uniform(-cfg.backoff_jitter_frac, cfg.backoff_jitter_frac)
    return min(base + jitter, cfg.backoff_max_ms)


# ──────────────────────── LLM non-streaming call ──────────────────────────────

async def _call_llm(
    client:     httpx.AsyncClient,
    cfg:        PipelineConfig,
    rl:         RLState,
    messages:   list[dict],
    max_tokens: int = 256,
    *,
    t0: float,                  # pipeline start time for patience tracking
) -> tuple[str, int, int]:
    """
    Non-streaming LLM call.  Returns (content, tokens_in, tokens_out).

    On first 429 from primary → switches to fallback immediately (free retry).
    Subsequent 429s apply the configured backoff policy.
    """
    model         = cfg.primary_model
    used_fallback = False
    attempt       = 0

    while attempt <= cfg.max_retries:
        rem_ms = cfg.patience_ms - (time.perf_counter() - t0) * 1_000
        if rem_ms < 400:
            raise TimeoutError("patience exhausted before LLM call")

        # sleep between retries (not before first attempt, not on fallback switch)
        if attempt > 0:
            wait = _backoff_ms(attempt, cfg, rl.retry_after_s)
            if wait > rem_ms - 400:
                raise TimeoutError("retry sleep would exceed patience")
            if wait > 0:
                await asyncio.sleep(wait / 1_000)
                rl.total_retry_wait_ms += wait
                rl.retry_after_s = 0.0

        try:
            resp = await client.post(
                cfg.endpoint,
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens, "stream": False},
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                timeout=min((rem_ms - 200) / 1_000, 10.0),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            attempt += 1; rl.total_retries += 1
            if attempt > cfg.max_retries or cfg.retry_policy == "none":
                raise TimeoutError(str(exc)) from exc
            continue

        rl.update(resp.headers)

        if resp.status_code == 429:
            try:
                err_msg = resp.json().get("error", {}).get("message", "")
            except Exception:
                err_msg = ""
            if "per day" in err_msg.lower() or "daily" in err_msg.lower():
                rl.daily_exhausted = True
                raise DailyQuotaError("Daily quota exhausted")

            rl.retry_after_s = float(resp.headers.get("retry-after", 1))

            # first 429 on primary → free switch to fallback, no sleep
            if model == cfg.primary_model and not used_fallback:
                used_fallback = True
                model = cfg.fallback_model
                continue

            if cfg.retry_policy == "none":
                raise RateLimitError(f"429 retry-after={rl.retry_after_s}s")

            attempt += 1; rl.total_retries += 1
            continue

        if resp.status_code != 200:
            raise PipelineAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        data    = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        usage   = data.get("usage", {})
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    raise RateLimitError("max retries exceeded (_call_llm)")


# ──────────────────────── LLM streaming call ──────────────────────────────────

@dataclass
class _StreamResult:
    content:    str
    tokens_in:  int
    tokens_out: int
    ttft_ms:    float    # request-sent → first delta.content token (server TTFT)
    total_ms:   float    # total duration of the streaming call
    fallback:   bool
    model_used: str


async def _stream_llm(
    client:     httpx.AsyncClient,
    cfg:        PipelineConfig,
    rl:         RLState,
    messages:   list[dict],
    max_tokens: int = 512,
    *,
    t0: float,
) -> _StreamResult:
    """
    Streaming LLM call with accurate TTFT measurement.

    openai/gpt-oss-20b emits reasoning in delta.reasoning before delta.content.
    TTFT and content collection both ignore delta.reasoning — TTFT is measured
    from request-sent to the first non-empty delta.content token.
    """
    model         = cfg.primary_model
    used_fallback = False
    attempt       = 0

    while attempt <= cfg.max_retries:
        rem_ms = cfg.patience_ms - (time.perf_counter() - t0) * 1_000
        if rem_ms < _BUDGET_DRAFT:
            raise TimeoutError("patience too low for streaming call")

        if attempt > 0:
            wait = _backoff_ms(attempt, cfg, rl.retry_after_s)
            if wait > rem_ms - _BUDGET_DRAFT:
                raise TimeoutError("retry sleep would exceed patience")
            if wait > 0:
                await asyncio.sleep(wait / 1_000)
                rl.total_retry_wait_ms += wait
                rl.retry_after_s = 0.0

        t_send         = time.perf_counter()
        content_parts: list[str] = []
        t_first:       float | None = None
        tokens_in = tokens_out = 0

        try:
            timeout = httpx.Timeout(
                connect=5.0,
                read=min((rem_ms - 200) / 1_000, 30.0),
                write=5.0, pool=5.0,
            )
            async with client.stream(
                "POST", cfg.endpoint,
                json={
                    "model":          model,
                    "messages":       messages,
                    "max_tokens":     max_tokens,
                    "stream":         True,
                    "stream_options": {"include_usage": True},
                },
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                timeout=timeout,
            ) as resp:
                rl.update(resp.headers)

                if resp.status_code == 429:
                    body_bytes = await resp.aread()
                    try:
                        err_msg = json.loads(body_bytes).get("error", {}).get("message", "")
                    except Exception:
                        err_msg = ""
                    if "per day" in err_msg.lower() or "daily" in err_msg.lower():
                        rl.daily_exhausted = True
                        raise DailyQuotaError("Daily quota exhausted")

                    rl.retry_after_s = float(resp.headers.get("retry-after", 1))

                    if model == cfg.primary_model and not used_fallback:
                        used_fallback = True
                        model = cfg.fallback_model
                        continue

                    if cfg.retry_policy == "none":
                        raise RateLimitError(f"429 on stream retry-after={rl.retry_after_s}s")

                    attempt += 1; rl.total_retries += 1
                    continue

                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    raise PipelineAPIError(
                        f"HTTP {resp.status_code}: {body_bytes.decode()[:200]}"
                    )

                async for raw in resp.aiter_lines():
                    raw = raw.strip()
                    if not raw.startswith("data:"):
                        continue
                    data_str = raw[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        # Only delta.content counts — skip delta.reasoning
                        # (gpt-oss-20b prefixes a reasoning chain before content)
                        text = delta.get("content") or ""
                        if text:
                            if t_first is None:
                                t_first = time.perf_counter()
                            content_parts.append(text)

                    usage = chunk.get("usage")
                    if usage:
                        tokens_in  = usage.get("prompt_tokens", 0)
                        tokens_out = usage.get("completion_tokens", 0)

        except (httpx.TimeoutException, httpx.ReadTimeout) as exc:
            attempt += 1; rl.total_retries += 1
            if attempt > cfg.max_retries or cfg.retry_policy == "none":
                raise TimeoutError(f"stream timeout: {exc}") from exc
            continue

        t_end = time.perf_counter()
        return _StreamResult(
            content    = "".join(content_parts).strip(),
            tokens_in  = tokens_in,
            tokens_out = tokens_out,
            ttft_ms    = ((t_first - t_send) * 1_000) if t_first else 0.0,
            total_ms   = (t_end - t_send) * 1_000,
            fallback   = used_fallback,
            model_used = model,
        )

    raise RateLimitError("max retries exceeded (_stream_llm)")


# ──────────────────────── citation verifier ───────────────────────────────────

def _verify_citations(
    answer: str, chunks: list[dict],
) -> tuple[bool, str]:
    """
    Programmatic citation check.  Returns (ok, issue_description).

    Checks:
      1. At least one [paper_id, p.N] citation present (unless CANNOT_ANSWER)
      2. Every cited (paper_id, page) exists in the retrieved chunks
    """
    if answer.strip() == "CANNOT_ANSWER":
        return True, ""

    citations = CITATION_RE.findall(answer)
    if not citations:
        return False, "no [paper_id, p.N] citations found"

    retrieved = {(c["paper_id"], c["page"]) for c in chunks}
    bad = [
        f"[{pid}, p.{pg}]"
        for pid, pg in citations
        if (pid, int(pg)) not in retrieved
    ]
    if bad:
        return False, f"citations not grounded in retrieved pages: {bad}"

    return True, ""


# ───────────────────────── result helpers ─────────────────────────────────────

def _base_result(
    rid: str, scenario: str, source: str,
    setting: str, start_iso: str,
) -> dict:
    return {
        # results.csv columns
        "request_id":        rid,
        "scenario":          scenario,
        "source":            source,
        "start_time":        start_iso,
        "first_token_ms":    None,
        "total_ms":          None,
        "status":            None,
        "retries":           0,
        "tokens_in":         0,
        "tokens_out":        0,
        "retrieval_setting": setting,
        "answer_grade":      "",
        # timing breakdown
        "queue_wait_ms": 0.0,
        "retry_wait_ms": 0.0,
        "pre_ttft_ms":   None,
        "post_ttft_ms":  None,
        # extra context
        "steps":          [],
        "load_shed":      False,
        "fallback_used":  False,
        "refusal_reason": None,
        "answer":         None,     # not in CSV, for inspection only
    }


def _finalize(
    result: dict,
    steps: list[str],
    t0: float,
    rl: RLState,
    *,
    status: str,
    refusal: str = "",
    answer: str = "",
) -> dict:
    result["total_ms"]      = (time.perf_counter() - t0) * 1_000
    result["status"]        = status
    result["steps"]         = steps
    result["retries"]       = rl.total_retries
    result["retry_wait_ms"] = rl.total_retry_wait_ms
    if answer:
        result["answer"]    = answer
    if refusal:
        result["refusal_reason"] = refusal
    # fall back to total_ms so the CSV never has a null in first_token_ms
    if result["first_token_ms"] is None:
        result["first_token_ms"] = result["total_ms"]
    return result


# ─────────────────────────── main pipeline ────────────────────────────────────

async def run_question(
    question:   str,
    cfg:        PipelineConfig,
    bm25:       BM25Index,
    client:     httpx.AsyncClient,
    *,
    rl:         RLState | None = None,
    request_id: str | None     = None,
    scenario:   str            = "default",
) -> dict:
    """
    Run one question through the 4-step pipeline.

    Returns a dict with all CSV_COLUMNS fields plus a timing breakdown
    (queue_wait_ms, retry_wait_ms, pre_ttft_ms, post_ttft_ms) and step list.

    Pass a shared RLState across consecutive calls to accumulate quota info.
    """
    t0        = time.perf_counter()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    rid       = request_id or f"req-{uuid.uuid4().hex[:8]}"
    rl        = rl or RLState()
    rl.reset_per_request()

    result = _base_result(rid, scenario, cfg.source,
                          cfg.retrieval_setting, start_iso)
    steps: list[str] = []

    def _rem() -> float:          # patience remaining in ms
        return cfg.patience_ms - (time.perf_counter() - t0) * 1_000

    # ── 0. pre-flight ─────────────────────────────────────────────────────────
    if rl.daily_exhausted:
        return _finalize(result, steps, t0, rl,
                         status="daily_exhausted",
                         refusal="Daily quota exhausted. Please retry tomorrow.")

    load_shed = rl.should_shed_load(cfg.shed_threshold)
    result["load_shed"] = load_shed

    # queue_wait_ms: time from call start to first LLM request being sent.
    # Usually ~0 ms; becomes non-zero if a semaphore or queue is added upstream.
    t_first_call = time.perf_counter()
    result["queue_wait_ms"] = (t_first_call - t0) * 1_000

    # ── 1. router (skipped when load-shedding to conserve tokens) ─────────────
    in_scope = True
    if not load_shed:
        if _rem() < _BUDGET_ROUTER:
            return _finalize(result, steps, t0, rl, status="timeout")

        steps.append("router")
        try:
            scope_str, t_in, t_out = await _call_llm(
                client, cfg, rl,
                messages=[
                    {"role": "system", "content": _ROUTER_SYS},
                    {"role": "user",   "content": f"Question: {question}"},
                ],
                max_tokens=_ROUTER_MAX_TOKENS,
                t0=t0,
            )
            result["tokens_in"]  += t_in
            result["tokens_out"] += t_out
            in_scope = "IN_SCOPE" in scope_str.upper()

        except TimeoutError:
            return _finalize(result, steps, t0, rl, status="timeout")
        except DailyQuotaError as e:
            return _finalize(result, steps, t0, rl,
                             status="daily_exhausted", refusal=str(e))
        except (RateLimitError, PipelineAPIError) as e:
            return _finalize(result, steps, t0, rl,
                             status="error", refusal=str(e))

    if not in_scope:
        return _finalize(
            result, steps, t0, rl,
            status="refused",
            refusal=(
                "Question is out of scope for this knowledge base "
                "(not about Transformer, Moshi, or H-Net / Dynamic Chunking)."
            ),
        )

    # ── 2. BM25 retrieval ─────────────────────────────────────────────────────
    # Synchronous but fast (< 50 ms even for 10 k chunks).
    steps.append("retrieve")
    chunks = bm25.retrieve(question, cfg.retrieval_setting, shed=load_shed)

    # ── 3. streaming drafter ──────────────────────────────────────────────────
    if _rem() < _BUDGET_DRAFT:
        return _finalize(result, steps, t0, rl, status="timeout")

    steps.append("draft")
    t_draft_sent = time.perf_counter()

    try:
        stream = await _stream_llm(
            client, cfg, rl,
            messages=[
                {"role": "system", "content": _DRAFTER_SYS},
                {"role": "user",   "content": _drafter_user(question, chunks)},
            ],
            max_tokens=_DRAFT_MAX_TOKENS,
            t0=t0,
        )
    except TimeoutError:
        return _finalize(result, steps, t0, rl, status="timeout")
    except DailyQuotaError as e:
        return _finalize(result, steps, t0, rl,
                         status="daily_exhausted", refusal=str(e))
    except (RateLimitError, PipelineAPIError) as e:
        return _finalize(result, steps, t0, rl,
                         status="error", refusal=str(e))

    result["tokens_in"]    += stream.tokens_in
    result["tokens_out"]   += stream.tokens_out
    result["fallback_used"] = stream.fallback

    # first_token_ms: pipeline start → first visible content token (user TTFT)
    result["first_token_ms"] = (
        (t_draft_sent - t0) * 1_000   # pipeline work before draft
        + stream.ttft_ms               # server-side TTFT
    )
    # pre_ttft_ms:  server TTFT of the drafter request
    # post_ttft_ms: drafter streaming time after first token
    result["pre_ttft_ms"]  = stream.ttft_ms
    result["post_ttft_ms"] = max(0.0, stream.total_ms - stream.ttft_ms)

    answer = stream.content

    if answer.strip() == "CANNOT_ANSWER":
        return _finalize(
            result, steps, t0, rl,
            status="refused",
            answer=answer,
            refusal="Retrieved context does not contain sufficient information.",
        )

    # ── 4. citation verifier + optional fix loop ──────────────────────────────
    # Skipped entirely when load-shedding or when patience is below threshold.
    if not load_shed and _rem() > _BUDGET_VERIFY:
        steps.append("verify")
        ok, issue = _verify_citations(answer, chunks)

        if not ok:
            if _rem() > _BUDGET_FIX:
                # one loop-back with a fix hint (non-streaming to be faster)
                steps.append("fix")
                try:
                    fixed, t_in, t_out = await _call_llm(
                        client, cfg, rl,
                        messages=[
                            {"role": "system", "content": _DRAFTER_SYS},
                            {"role": "user",
                             "content": _drafter_user(
                                 question, chunks, fix_hint=issue)},
                        ],
                        max_tokens=_FIX_MAX_TOKENS,
                        t0=t0,
                    )
                    result["tokens_in"]  += t_in
                    result["tokens_out"] += t_out

                    ok2, issue2 = _verify_citations(fixed, chunks)
                    if ok2:
                        answer = fixed
                    else:
                        # second failure → fail safe (don't loop again)
                        return _finalize(
                            result, steps, t0, rl,
                            status="refused",
                            answer=answer,
                            refusal=f"Citations could not be verified: {issue2}",
                        )

                except TimeoutError:
                    return _finalize(result, steps, t0, rl,
                                     status="timeout", answer=answer)
                except DailyQuotaError as e:
                    return _finalize(result, steps, t0, rl,
                                     status="daily_exhausted",
                                     answer=answer, refusal=str(e))
                except (RateLimitError, PipelineAPIError) as e:
                    # fix call failed → still return the unverified draft as refused
                    return _finalize(
                        result, steps, t0, rl,
                        status="refused",
                        answer=answer,
                        refusal=f"Fix call failed ({e}); original issue: {issue}",
                    )
            else:
                # not enough time for the fix loop
                return _finalize(
                    result, steps, t0, rl,
                    status="refused",
                    answer=answer,
                    refusal=f"Citations not verified (patience exhausted): {issue}",
                )

    return _finalize(result, steps, t0, rl, status="ok", answer=answer)


# ─────────────────────────────── CSV writer ───────────────────────────────────

def write_csv_row(result: dict, path: str = "results.csv") -> None:
    p = Path(path)
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {**result}
        if isinstance(row.get("steps"), list):
            row["steps"] = " → ".join(row["steps"]) if row["steps"] else ""
        # round floats for readability
        for k in ("first_token_ms", "total_ms", "queue_wait_ms",
                  "retry_wait_ms", "pre_ttft_ms", "post_ttft_ms"):
            if isinstance(row.get(k), float):
                row[k] = round(row[k], 1)
        writer.writerow(row)


# ─────────────────────────────── CLI printer ──────────────────────────────────

def _print_result(r: dict) -> None:
    status = r.get("status", "?")
    icon = {
        "ok":              "✓",
        "refused":         "⊘",
        "timeout":         "⏱",
        "error":           "✗",
        "daily_exhausted": "⛔",
    }.get(status, "?")
    print(f"\n  {icon} [{status.upper()}]  {r['request_id']}")
    print(f"    steps        : {' → '.join(r.get('steps') or [])}")
    print(f"    first_tok_ms : {r.get('first_token_ms') or 0:.0f}")
    print(f"    total_ms     : {r.get('total_ms') or 0:.0f}")
    print(f"    tokens in/out: {r.get('tokens_in', 0)} / {r.get('tokens_out', 0)}")
    print(f"    retries      : {r.get('retries', 0)}")
    print(f"    load_shed    : {r.get('load_shed', False)}")
    print(f"    fallback     : {r.get('fallback_used', False)}")
    if r.get("refusal_reason"):
        print(f"    refusal      : {r['refusal_reason']}")
    if r.get("answer"):
        snippet = str(r["answer"])[:200].replace("\n", " ")
        print(f"    answer       : {snippet}…" if len(str(r["answer"])) > 200
              else f"    answer       : {r['answer']}")
    p = (
        f"    timing (ms)  : "
        f"queue={r.get('queue_wait_ms') or 0:.0f}  "
        f"retry={r.get('retry_wait_ms') or 0:.0f}  "
        f"pre_ttft={r.get('pre_ttft_ms') or 0:.0f}  "
        f"post_ttft={r.get('post_ttft_ms') or 0:.0f}"
    )
    print(p)


# ─────────────────────────────── CLI ──────────────────────────────────────────

async def _run_all(
    questions: list[dict],
    cfg:       PipelineConfig,
    bm25:      BM25Index,
    scenario:  str,
) -> None:
    rl = RLState()   # one shared state so quota info accumulates
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(questions, 1):
            qtext = q["question"]
            cat   = q.get("category", "?")
            qid   = q.get("id", f"q{i:02d}")
            print(f"\n[{i:02d}/{len(questions)}] {qid} ({cat})")
            print(f"  Q: {qtext[:100]}{'…' if len(qtext) > 100 else ''}")

            result = await run_question(
                qtext, cfg, bm25, client,
                rl=rl,
                request_id=f"{qid}-{uuid.uuid4().hex[:6]}",
                scenario=scenario,
            )
            write_csv_row(result, cfg.results_csv)
            _print_result(result)

    print(f"\n[relay] done — results appended to {cfg.results_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="The Relay — 4-step RAG pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", "-q", metavar="TEXT",
                       help="Run a single question")
    group.add_argument("--all", "-a", action="store_true",
                       help="Run all questions from --questions-file")

    parser.add_argument(
        "--source", choices=["groq", "mock"],
        help="'mock' points base_url at http://localhost:8001/openai/v1",
    )
    parser.add_argument(
        "--retrieval", default="med_k5_norerank",
        choices=["small_k3_norerank", "med_k5_norerank", "med_k8_rerank3"],
    )
    parser.add_argument(
        "--retry", default="smart",
        choices=["smart", "aggressive", "none"],
        dest="retry_policy",
    )
    parser.add_argument("--patience", type=float, default=8_000.0,
                        metavar="MS", help="Per-request timeout ms (default 8000)")
    parser.add_argument("--scenario", default="default",
                        help="Label written to the scenario column in results.csv")
    parser.add_argument("--corpus",   default="corpus.json")
    parser.add_argument("--csv",      default="results.csv", dest="results_csv")
    parser.add_argument("--questions-file", default="questions.json")

    args = parser.parse_args()

    cfg = PipelineConfig(
        retrieval_setting=args.retrieval,
        retry_policy=args.retry_policy,
        patience_ms=args.patience,
        corpus_path=args.corpus,
        results_csv=args.results_csv,
    )
    if args.source == "mock":
        cfg.base_url = "http://localhost:8001/openai/v1"
    elif args.source == "groq":
        cfg.base_url = "https://api.groq.com/openai/v1"

    if cfg.source == "groq" and not cfg.api_key:
        print("ERROR: GROQ_API_KEY is not set.")
        return

    bm25 = BM25Index(cfg.corpus_path)

    print(
        f"\n[relay] source={cfg.source}  model={cfg.primary_model}  "
        f"retrieval={cfg.retrieval_setting}  retry={cfg.retry_policy}  "
        f"patience={cfg.patience_ms:.0f} ms"
    )

    if args.question:
        async def _single() -> None:
            rl = RLState()
            async with httpx.AsyncClient() as client:
                result = await run_question(
                    args.question, cfg, bm25, client,
                    rl=rl, scenario=args.scenario,
                )
            write_csv_row(result, cfg.results_csv)
            _print_result(result)

        asyncio.run(_single())

    else:
        qfile = Path(args.questions_file)
        if not qfile.exists():
            print(f"ERROR: {qfile} not found")
            return
        questions = json.loads(qfile.read_text(encoding="utf-8"))
        asyncio.run(_run_all(questions, cfg, bm25, scenario=args.scenario))


if __name__ == "__main__":
    main()