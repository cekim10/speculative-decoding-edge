# verify_service.py
import time
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

# vLLM OpenAI server that hosts the TARGET (verifier) model (your 1.3B)
OPENAI_BASE = "http://127.0.0.1:8000/v1"
TARGET_MODEL = "facebook/opt-1.3b"

tok = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=True)

print("[VERIFY] model (via vLLM):", TARGET_MODEL)
print("[VERIFY] bos_token_id:", tok.bos_token_id, "eos_token_id:", tok.eos_token_id)


class VerifyReq(BaseModel):
    seq_id: str
    prompt_ids: List[int]
    draft_token_ids: List[int]
    want_bonus: bool = True


class VerifyResp(BaseModel):
    seq_id: str
    accepted_len: int
    bonus_token_id: Optional[int] = None
    recovered_token_id: Optional[int] = None   # <-- NEW
    rtt_ms: float
    verify_ms: float


app = FastAPI()


def vllm_next_token(prefix_text: str) -> Tuple[int, str]:
    """
    Ask the vLLM OpenAI server for ONE greedy next token on prefix_text.
    Returns (token_id, token_str).
    """
    payload = {
        "model": TARGET_MODEL,
        "prompt": prefix_text,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "logprobs": 5,
        "echo": False,
    }
    headers = {"Content-Type": "application/json"}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{OPENAI_BASE}/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    choice = data["choices"][0]
    lp = choice.get("logprobs", {}) or {}
    toks = lp.get("tokens") or []
    if toks:
        tstr = toks[0]
        tid = tok.convert_tokens_to_ids(tstr)
        return int(tid), tstr

    # fallback if server didn't return tokens (rare)
    out_text = choice.get("text", "")
    ids = tok(out_text, add_special_tokens=False).input_ids
    if ids:
        tid = int(ids[0])
        tstr = tok.convert_ids_to_tokens([tid])[0]
        return tid, tstr

    eos = int(tok.eos_token_id) if tok.eos_token_id is not None else -1
    tstr = tok.convert_ids_to_tokens([eos])[0] if eos >= 0 else ""
    return eos, tstr


def append_token_str(prefix_text: str, token_str: str) -> str:
    """Append a single tokenizer token string faithfully to text."""
    return prefix_text + tok.convert_tokens_to_string([token_str])


def greedy_verify_vllm(
    prompt_ids: List[int], draft_token_ids: List[int]
) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Return (accepted_len, bonus_token_id, recovered_token_id).
    recovered_token_id is the verifier's greedy token at the first mismatch.
    """
    prefix_text = tok.decode(prompt_ids, skip_special_tokens=False)
    print("[VERIFY] tail_text:", repr(prefix_text[-80:]))

    # visibility: greedy next at current prefix
    expect_id, expect_str = vllm_next_token(prefix_text)
    print(f"[VERIFY] first_greedy_next_id={expect_id} token_str={repr(expect_str)}")

    accepted_len = 0
    recovered_token_id: Optional[int] = None

    for draft_tid in draft_token_ids:
        expect_id, expect_str = vllm_next_token(prefix_text)
        if expect_id == draft_tid:
            accepted_len += 1
            prefix_text = append_token_str(prefix_text, expect_str)
        else:
            recovered_token_id = expect_id  # <-- key: provide recovered
            break

    bonus = None
    if accepted_len == len(draft_token_ids):
        bonus_id, bonus_str = vllm_next_token(prefix_text)
        bonus = bonus_id

    return accepted_len, bonus, recovered_token_id


@app.post("/verify", response_model=VerifyResp)
def verify(req: VerifyReq):
    print(
        f"[VERIFY] /verify seq={req.seq_id} "
        f"prompt_ids_len={len(req.prompt_ids)} "
        f"draft_len={len(req.draft_token_ids)} "
        f"draft_first={req.draft_token_ids[0] if req.draft_token_ids else None}"
    )
    print("[VERIFY] prompt_ids[:10]:", req.prompt_ids[:10])

    t0 = time.time()
    ts = time.time()
    acc_len, bonus, recovered = greedy_verify_vllm(req.prompt_ids, req.draft_token_ids)
    t1 = time.time()

    print(f"[VERIFY] accepted_len={acc_len} bonus={bonus} recovered={recovered}")
    return VerifyResp(
        seq_id=req.seq_id,
        accepted_len=acc_len,
        bonus_token_id=bonus,
        recovered_token_id=recovered,
        rtt_ms=(t1 - t0) * 1000.0,
        verify_ms=(t1 - ts) * 1000.0,
    )

