"""
1_probe_and_ingest.py
─────────────────────
Phase 1 of The Relay pipeline:

  1. INGEST  — downloads 3 arXiv PDFs, extracts page-numbered text,
               writes corpus.json  [{paper_id, page, text}, ...]

  2. PROBE   — sends 12 test requests to Groq (3 scenarios) with streaming,
               prints x-ratelimit-* / retry-after headers, TTFT, total
               latency, token counts, and any errors — raw numbers you can
               use to calibrate the mock server later.

"""

import asyncio
import json
import os
import time
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
 
load_dotenv()

# ── configuration ────────────────────────────────────────────────────────────

PAPERS = [
    "1706.03762v7",   # Attention Is All You Need
    "2410.00037v1",   # Moshi: speech-text foundation model
    "2507.07955v2",   # H-Net: Dynamic Chunking
]

ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}"

PDF_DIR    = Path("pdfs")
CORPUS_OUT = Path("corpus.json")

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "qwen/qwen3.8-27b"   # fast, free-tier friendly; change as needed
MAX_TOKENS = 150                       # keep responses short to stay within limits

# 12 probe prompts — varied length to get a spread of token counts
PROBE_PROMPTS = [
    # spaced (A)
    "In one sentence, what is the core architectural innovation of the Transformer?",
    "What is multi-head attention and why is it useful?",
    "Explain positional encoding in the Transformer in two sentences.",
    "What are the two sub-layers in every Transformer encoder block?",
    # sequential (B)
    "What BLEU score did the Transformer base model achieve on WMT 2014 English-to-German?",
    "What optimizer did Vaswani et al. use to train the Transformer, and what was special about their learning rate schedule?",
    "What is full-duplex speech dialogue and why is it challenging for AI systems?",
    "Why is tokenization considered a barrier to truly end-to-end sequence modeling?",
    # burst (C)
    "Name one advantage of self-attention over recurrent layers for long sequences.",
    "What does 'Inner Monologue' mean in the context of a speech-language model?",
    "What is dynamic chunking and how does it differ from byte-pair encoding?",
    "How does a hierarchical sequence model improve data efficiency over flat baselines?",
]

# ── PDF download ──────────────────────────────────────────────────────────────

async def download_pdf(client: httpx.AsyncClient, paper_id: str) -> Path:
    dest = PDF_DIR / f"{paper_id}.pdf"
    if dest.exists():
        print(f"  [cache]    {paper_id}.pdf  ({dest.stat().st_size:,} bytes)")
        return dest

    url = ARXIV_PDF_URL.format(paper_id=paper_id)
    print(f"  [download] {url}")
    r = await client.get(url, follow_redirects=True, timeout=90.0)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  [saved]    {dest}  ({len(r.content):,} bytes)")
    return dest


# ── text extraction ───────────────────────────────────────────────────────────

def extract_pages(paper_id: str, pdf_path: Path) -> list[dict]:
    """Return [{paper_id, page, text}, ...] for every non-empty page."""
    doc   = fitz.open(str(pdf_path))
    pages = []
    for n, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"paper_id": paper_id, "page": n, "text": text})
    doc.close()
    print(f"  [extract]  {paper_id}: {len(pages)} pages with text")
    return pages


# ── Groq streaming probe ──────────────────────────────────────────────────────

async def probe_one(
    client:     httpx.AsyncClient,
    api_key:    str,
    prompt:     str,
    request_id: int,
    scenario:   str,
) -> dict:
    """
    Send one streaming chat request to Groq and collect:
      - HTTP status code
      - all x-ratelimit-* and retry-after response headers
      - TTFT  (time from request start → first non-empty content delta)
      - total latency  (request start → stream closed)
      - prompt / completion token counts  (from last stream chunk usage)
      - any error message
    """
    result: dict = {
        "request_id":        request_id,
        "scenario":          scenario,
        "prompt_snippet":    prompt[:72] + ("…" if len(prompt) > 72 else ""),
        "status":            None,
        "error":             None,
        "ttft_ms":           None,
        "total_ms":          None,
        "tokens_in":         None,
        "tokens_out":        None,
        "ratelimit_headers": {},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens":     MAX_TOKENS,
        "stream":         True,
        "stream_options": {"include_usage": True},   # usage in final SSE chunk
    }

    t0            = time.perf_counter()
    t_first_token = None

    try:
        async with client.stream(
            "POST", GROQ_URL,
            headers=headers, json=payload, timeout=30.0,
        ) as resp:
            # ── capture rate-limit headers immediately ──
            result["status"] = resp.status_code
            result["ratelimit_headers"] = {
                k: v for k, v in resp.headers.items()
                if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
            }

            if resp.status_code != 200:
                body = await resp.aread()
                result["error"]    = body.decode(errors="replace")
                result["total_ms"] = (time.perf_counter() - t0) * 1000
                return result

            # ── stream SSE lines ──
            async for raw_line in resp.aiter_lines():
                raw_line = raw_line.strip()
                if not raw_line.startswith("data:"):
                    continue
                payload_str = raw_line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                # first content delta → TTFT
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content") and t_first_token is None:
                        t_first_token          = time.perf_counter()
                        result["ttft_ms"]      = (t_first_token - t0) * 1000

                # usage arrives in the last chunk (stream_options.include_usage)
                usage = chunk.get("usage")
                if usage:
                    result["tokens_in"]  = usage.get("prompt_tokens")
                    result["tokens_out"] = usage.get("completion_tokens")

        result["total_ms"] = (time.perf_counter() - t0) * 1000

    except httpx.HTTPStatusError as exc:
        result["status"]    = exc.response.status_code
        result["error"]     = str(exc)
        result["total_ms"]  = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        result["status"]    = "exception"
        result["error"]     = f"{type(exc).__name__}: {exc}"
        result["total_ms"]  = (time.perf_counter() - t0) * 1000

    return result


# ── pretty-print one probe result ────────────────────────────────────────────

def print_result(r: dict) -> None:
    sep = "  "
    print(f"\n{sep}#{r['request_id']:02d} [{r['scenario']}]")
    print(f"{sep}  prompt  : {r['prompt_snippet']}")
    print(f"{sep}  status  : {r['status']}")
    if r["error"]:
        snippet = r["error"][:300].replace("\n", " ")
        print(f"{sep}  error   : {snippet}")
    if r["ttft_ms"] is not None:
        print(f"{sep}  TTFT    : {r['ttft_ms']:.1f} ms")
    if r["total_ms"] is not None:
        print(f"{sep}  total   : {r['total_ms']:.1f} ms")
    if r["tokens_in"] is not None:
        print(f"{sep}  tokens  : in={r['tokens_in']}  out={r['tokens_out']}")
    if r["ratelimit_headers"]:
        print(f"{sep}  headers :")
        for k, v in sorted(r["ratelimit_headers"].items()):
            print(f"{sep}    {k}: {v}")


# ── run all probe scenarios ───────────────────────────────────────────────────

async def run_probes(api_key: str) -> None:
    print("\n" + "=" * 60)
    print("GROQ API PROBE")
    print("=" * 60)
    print(f"model      : {GROQ_MODEL}")
    print(f"max_tokens : {MAX_TOKENS}")

    all_results: list[dict] = []
    req_id = 0

    async with httpx.AsyncClient() as client:

        # ── Scenario A: spaced (1 s gap) ────────────────────────────────────
        print("\n--- Scenario A: spaced  (1 s between requests) ---")
        for i in range(4):
            req_id += 1
            r = await probe_one(client, api_key, PROBE_PROMPTS[i], req_id, "spaced")
            print_result(r)
            all_results.append(r)
            if i < 3:
                await asyncio.sleep(1.0)

        # ── Scenario B: sequential (back-to-back, no sleep) ──────────────────
        print("\n--- Scenario B: sequential  (no sleep) ---")
        for i in range(4, 8):
            req_id += 1
            r = await probe_one(client, api_key, PROBE_PROMPTS[i], req_id, "sequential")
            print_result(r)
            all_results.append(r)

        # ── Scenario C: burst (4 concurrent tasks) ───────────────────────────
        print("\n--- Scenario C: burst  (4 concurrent) ---")
        tasks = []
        for i in range(8, 12):
            req_id += 1
            tasks.append(
                probe_one(client, api_key, PROBE_PROMPTS[i], req_id, "burst")
            )
        burst_results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in burst_results:
            if isinstance(item, Exception):
                print(f"\n  burst task raised: {item}")
            else:
                print_result(item)
                all_results.append(item)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ok  = [r for r in all_results if r["status"] == 200]
    err = [r for r in all_results if r["status"] != 200]
    print(f"  total / ok / error : {len(all_results)} / {len(ok)} / {len(err)}")

    if ok:
        ttfts  = [r["ttft_ms"]  for r in ok if r["ttft_ms"]  is not None]
        totals = [r["total_ms"] for r in ok if r["total_ms"] is not None]
        if ttfts:
            print(
                f"  TTFT   min/avg/max : "
                f"{min(ttfts):.0f} / {sum(ttfts)/len(ttfts):.0f} / {max(ttfts):.0f} ms"
            )
        if totals:
            print(
                f"  total  min/avg/max : "
                f"{min(totals):.0f} / {sum(totals)/len(totals):.0f} / {max(totals):.0f} ms"
            )

    # aggregate all unique rate-limit header values across the run
    seen_rl: dict[str, set] = {}
    for r in all_results:
        for k, v in r.get("ratelimit_headers", {}).items():
            seen_rl.setdefault(k, set()).add(v)

    if seen_rl:
        print("\n  Rate-limit header values seen across all requests:")
        for k in sorted(seen_rl):
            print(f"    {k}: {', '.join(sorted(seen_rl[k]))}")

    if err:
        print("\n  Failed requests:")
        for r in err:
            print(f"    #{r['request_id']:02d} [{r['scenario']}]  status={r['status']}  {(r['error'] or '')[:120]}")


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set GROQ_API_KEY in your environment before running.")
        return

    # ────────────────────────── INGEST ──────────────────────────────────────
    print("=" * 60)
    print("PDF INGEST")
    print("=" * 60)

    PDF_DIR.mkdir(exist_ok=True)
    corpus: list[dict] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; relay-ingest/1.0)"},
    ) as client:
        for paper_id in PAPERS:
            print(f"\nPaper: {paper_id}")
            pdf_path = await download_pdf(client, paper_id)
            corpus.extend(extract_pages(paper_id, pdf_path))

    CORPUS_OUT.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total_chars = sum(len(p["text"]) for p in corpus)
    print(
        f"\n[corpus] {len(corpus)} pages | {total_chars:,} chars → {CORPUS_OUT}"
    )

    # ────────────────────────── PROBE ───────────────────────────────────────
    await run_probes(api_key)


if __name__ == "__main__":
    asyncio.run(main())