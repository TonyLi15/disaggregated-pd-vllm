#!/usr/bin/env python3
# Simple TTFT & throughput benchmark against the proxy.

import asyncio, aiohttp, time, json, statistics as st, os, argparse, random, string

def make_prompt(n_tokens:int)->str:
    words = ["".join(random.choices(string.ascii_lowercase, k=5)) for _ in range(n_tokens)]
    return " ".join(words)

async def ttft_one(session, url, headers, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    async with session.post(url, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        ttft = None
        async for raw in resp.content:
            if not raw:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
                break
    return ttft

async def throughput_one(session, url, headers, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(url, headers=headers, json=payload) as resp:
        text = await resp.text()
        t1 = time.perf_counter()
        if resp.status != 200:
            return None, None, resp.status, text
    try:
        data = json.loads(text)
        comp = data.get("usage", {}).get("completion_tokens", None)
    except Exception:
        comp = None
    dur = t1 - t0
    return comp, dur, 200, None

async def run(args):
    base = f"http://{args.host}:{args.port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','sk-noop')}",
    }
    prompts = [make_prompt(args.prompt_tokens) for _ in range(args.requests)]
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # TTFT
        sem = asyncio.Semaphore(args.concurrency)
        ttfts = []
        async def ttft_task(p):
            async with sem:
                try:
                    v = await ttft_one(session, base, headers, args.model, p, args.max_tokens)
                    if v is not None: ttfts.append(v)
                except Exception as e:
                    print(f"[TTFT] error: {e}")

        await asyncio.gather(*[ttft_task(p) for p in prompts])
        if ttfts:
            print(f"\n== TTFT (stream=true) N={len(ttfts)} ==")
            q95 = st.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else max(ttfts)
            print(f"  p50={st.median(ttfts):.3f}s  p95={q95:.3f}s  min={min(ttfts):.3f}s  max={max(ttfts):.3f}s")
        else:
            print("\n== TTFT: none ==")

        # Throughput
        sem2 = asyncio.Semaphore(args.concurrency)
        toks, durs = [], []
        errors = 0
        async def thr_task(p):
            nonlocal errors
            async with sem2:
                try:
                    comp, dur, code, err = await throughput_one(session, base, headers, args.model, p, args.max_tokens)
                    if code == 200 and comp is not None:
                        toks.append(comp); durs.append(dur)
                    else:
                        errors += 1
                        if err: print(f"[THR] HTTP {code} body={err[:300]}")
                except Exception as e:
                    errors += 1
                    print(f"[THR] error: {e}")

        t0 = time.perf_counter()
        await asyncio.gather(*[thr_task(p) for p in prompts])
        wall = time.perf_counter() - t0

        if toks:
            tps_each = [t/d for t,d in zip(toks,durs) if d>0]
            q95 = st.quantiles(tps_each, n=20)[18] if len(tps_each) >= 20 else max(tps_each)
            print(f"\n== Throughput (stream=false) N={len(toks)}, errors={errors} ==")
            print(f"  per-request tokens/sec: p50={st.median(tps_each):.1f}  p95={q95:.1f}  mean={st.mean(tps_each):.1f}")
            print(f"  total generated tokens = {sum(toks)}")
            print(f"  wall time (whole run)   = {wall:.2f}s")
            print(f"  aggregate throughput    = {sum(toks)/wall:.1f} tokens/sec")
        else:
            print("\n== Throughput: none ==")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("SRV_IP","127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PROXY_HTTP_PORT","10001")))
    ap.add_argument("--model", default=os.getenv("MODEL","Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--http-timeout", type=float, default=600)
    args = ap.parse_args()
    asyncio.run(run(args))