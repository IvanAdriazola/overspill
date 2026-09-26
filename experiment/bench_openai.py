"""Decode tok/s via streaming OpenAI chat completion (run from Windows).
usage: python bench.py <base_url> <model> [runs]"""
import json, os, sys, time, urllib.request

PROMPT = ("Explain, in about 250 words, how a Mixture-of-Experts transformer layer routes "
          "tokens to experts, and why that makes offloading experts to CPU RAM practical.")
base, model = sys.argv[1], sys.argv[2]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
# Agent prompts are large (tools + transcript + file contents), so prompt
# processing speed matters as much as decode: _long_prompt feeds ~24k chars of
# real code (roughly 6-8k tokens) and a short answer, so ttft ~ prefill time.
LONG = extra.pop("_long_prompt", False)
if LONG:
    from pathlib import Path
    # Any ~24k chars of Python source works; the published numbers used a private Python module of that
    # size (6,446 prompt tokens). By default this uses a FreeToken source file of similar length.
    src = os.environ.get("LONG_PROMPT_FILE") or str(Path(__file__).resolve().parents[1] / "python/freetoken/engine/engine.py")
    code = Path(src).read_text(encoding="utf-8")[:24000]
    PROMPT = "Summarize what this Python module does in 3 short bullet points.\n\n" + code
# _prompt overrides the prompt (e.g. a coding question); _full_text prints the whole answer.
PROMPT = extra.pop("_prompt", PROMPT)
FULL = extra.pop("_full_text", False)

def one():
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 120 if LONG else 300, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; chunks = 0; usage = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for c in j.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    now = time.time(); first = first or now; last = now; chunks += 1; text.append(piece)
    n = (usage or {}).get("completion_tokens", chunks)
    prompt_tokens = (usage or {}).get("prompt_tokens")
    dec = (n - 1) / (last - first) if last and last > first else 0
    ttft = round(first - t0, 2) if first else None
    print(json.dumps({"mode": "long_prompt" if LONG else "short", "prompt_tokens": prompt_tokens,
                      "completion_tokens": n, "chunks": chunks, "ttft_s": ttft,
                      "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None,
                      "total_s": round(time.time() - t0, 2), "decode_tok_s": round(dec, 2)}))
    print("TEXT:", "".join(text) if FULL else "".join(text)[:200].replace("\n", " "))

for i in range(runs):
    one()
