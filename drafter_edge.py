# # drafter_edge.py
# import math
# import os
# import queue
# import threading
# import uuid
# from typing import List, Tuple, Dict
#
# import httpx, ujson as json
# from transformers import AutoTokenizer
#
# # Drafter points to its own vLLM OpenAI server (your 350M)
# OPENAI_BASE = "http://127.0.0.1:8001/v1"
# # Verifier microservice (this file: verify_service.py)
# VERIFY_URL = "http://127.0.0.1:7001"
#
# MODEL_NAME = "facebook/opt-350m"
# TOKENIZER = MODEL_NAME
#
# # vary per process to emulate heterogeneity
# TEMPERATURE = float(os.getenv("TEMP", "0.7"))
# TOP_P       = float(os.getenv("TOPP", "0.9"))
# TOP_K       = int(os.getenv("TOPK", "0"))   # keep 0 unless you intend to clip
#
# alpha_pas = 0.0
# TOP_K_PAS = 4
#
# tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
#
# print("[EDGE] model:", MODEL_NAME)
# print("[EDGE] bos_token_id:", tok.bos_token_id, "eos_token_id:", tok.eos_token_id)
#
#
# def bucketize_prefix(ids: List[int]) -> Tuple[int, int]:
#     a = ids[-2] if len(ids) >= 2 else -1
#     b = ids[-1] if len(ids) >= 1 else -1
#     return a, b
#
#
# PAS: Dict[Tuple[int, int], Dict[int, Tuple[int, int]]] = {}
#
#
# def pas_get_topk(bucket: Tuple[int, int], k=TOP_K_PAS):
#     tab = PAS.get(bucket, {})
#     return sorted(tab.items(), key=lambda x: x[1][0], reverse=True)[:k]
#
#
# def pas_update(bucket: Tuple[int, int], token_id: int, accepted: bool):
#     tab = PAS.setdefault(bucket, {})
#     acc_q8, w = tab.get(token_id, (128, 1))  # ~0.5 initial
#     p = acc_q8 / 255.0
#     target = 1.0 if accepted else 0.0
#     p_new = 0.85 * p + 0.15 * target
#     acc_q8_new = max(5, min(250, int(round(p_new * 255))))
#     w = min(w + 1, 1000)
#     tab[token_id] = (acc_q8_new, w)
#     if len(tab) > TOP_K_PAS:
#         keep = sorted(tab.items(), key=lambda x: x[1][0], reverse=True)[:TOP_K_PAS]
#         PAS[bucket] = dict(keep)
#
#
# _seen_debug = False
#
#
# def openai_sample_one_from_text(prefix_text: str, logit_bias: Dict[int, float]) -> tuple[int, str]:
#     """
#     Ask vLLM (OpenAI completions) for ONE token continuation of prefix_text.
#     Returns (token_id, token_str).
#     """
#     global _seen_debug
#     headers = {"Content-Type": "application/json"}
#     payload = {
#         "model": MODEL_NAME,
#         "prompt": prefix_text,
#         "max_tokens": 1,
#         "temperature": TEMPERATURE,
#         "top_p": TOP_P,
#         "top_k": TOP_K,
#         "logprobs": 1,
#         "echo": False,
#         "logit_bias": {str(k): v for k, v in logit_bias.items()},
#     }
#     with httpx.Client(timeout=30.0) as client:
#         r = client.post(f"{OPENAI_BASE}/completions", headers=headers, data=json.dumps(payload))
#         r.raise_for_status()
#         data = r.json()
#
#     if not _seen_debug:
#         print("[EDGE] /completions raw keys:", list(data.keys()))
#         try:
#             ch = data["choices"][0]
#             print("[EDGE] choices[0].keys:", list(ch.keys()))
#             lp = ch.get("logprobs", {})
#             if isinstance(lp, dict):
#                 print("[EDGE] logprobs keys:", list(lp.keys()))
#                 print("[EDGE] sample logprobs snippet:",
#                       {k: lp.get(k, None) for k in ["tokens", "top_logprobs", "text_offset"]})
#         finally:
#             pass
#
#     choice = data["choices"][0]
#     lp = choice.get("logprobs", {})
#     tokens = lp.get("tokens") or []
#     if tokens:
#         token_str = tokens[0]
#         token_id = tok.convert_tokens_to_ids(token_str)
#         return int(token_id), token_str
#
#     out_text = choice.get("text", "")
#     ids = tok(out_text, add_special_tokens=False).input_ids
#     if ids:
#         tid = int(ids[0])
#         tstr = tok.convert_ids_to_tokens([tid])[0]
#         return tid, tstr
#     return int(tok.eos_token_id), tok.convert_ids_to_tokens([tok.eos_token_id])[0]
#
#
# def drafter_step(prefix_text: str, B: int, prefix_ids_for_pas: List[int]) -> Tuple[List[int], List[str]]:
#     out_ids: List[int] = []
#     out_toks: List[str] = []
#     print("[EDGE] draft_against_tail:", repr(prefix_text[-80:]))
#
#     for i in range(B):
#         bucket = bucketize_prefix(prefix_ids_for_pas + out_ids)
#         print(f"bucket: {bucket}")
#         bias: Dict[int, float] = {}
#         for tok_id, (acc_q8, _) in pas_get_topk(bucket):
#             p_hat = max(0.05, acc_q8 / 255.0)
#             bias[tok_id] = alpha_pas * math.log(p_hat)
#
#         tid, tstr = openai_sample_one_from_text(
#             prefix_text + tok.convert_tokens_to_string(out_toks), bias
#         )
#         if i == 0:
#             print(f"[EDGE] first_drafted_id={tid} token_str={repr(tstr)}")
#         out_ids.append(tid)
#         out_toks.append(tstr)
#     return out_ids, out_toks
#
#
# class Controller:
#     def __init__(self, B=16, Bmin=4, Bmax=64, r_hi=0.8, r_lo=0.4, alpha=0.2):
#         self.B = B
#         self.Bmin = Bmin
#         self.Bmax = Bmax
#         self.r = 0.6
#         self.alpha = alpha
#         self.r_hi = r_hi
#         self.r_lo = r_lo
#         self.hi = 0
#         self.lo = 0
#
#     def update(self, batch_r: float):
#         self.r = (1 - self.alpha) * self.r + self.alpha * batch_r
#         if batch_r < 0.1:
#             self.B = max(self.B // 2, self.Bmin)
#             self.hi = self.lo = 0
#             return
#         if self.r >= self.r_hi:
#             self.hi += 1
#             self.lo = 0
#             if self.hi >= 2:
#                 self.B = min(self.B + 8, self.Bmax)
#         elif self.r <= self.r_lo:
#             self.lo += 1
#             self.hi = 0
#             if self.lo >= 2:
#                 self.B = max(self.B // 2, self.Bmin)
#         else:
#             self.hi = self.lo = 0
#
#
# def run_edge(name: str, prompt: str, max_tokens=128):
#     seq_id = f"{name}-{uuid.uuid4().hex[:6]}"
#     print(f"seq_id={seq_id}")
#
#     # Canonical source of truth = text
#     prefix_text = prompt
#     prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
#
#     print(f"[EDGE] prompt_text({len(prompt)} chars)='{prompt[:80]}...'")
#     print("[EDGE] initial prefix_ids[:10]:", prefix_ids[:10])
#
#     ctrl = Controller(B=16, Bmin=4, Bmax=64)
#     inflight: queue.Queue = queue.Queue()
#     generated = 0
#
#     def offload(draft_ids: List[int], prompt_text_for_verify: str):
#         prompt_ids = tok(prompt_text_for_verify, add_special_tokens=False).input_ids
#         print("[EDGE] offload_tail:", repr(tok.decode(prompt_ids, skip_special_tokens=False)[-80:]))
#         payload = {
#             "seq_id": seq_id,
#             "prompt_ids": prompt_ids,
#             "draft_token_ids": draft_ids,
#             "want_bonus": True,
#         }
#         with httpx.Client(timeout=60.0) as client:
#             r = client.post(f"{VERIFY_URL}/verify", json=payload)
#             r.raise_for_status()
#             resp = r.json()
#         inflight.put((
#             resp["accepted_len"],
#             len(draft_ids),
#             resp.get("bonus_token_id"),
#             resp.get("recovered_token_id"),  # <-- NEW
#         ))
#
#     while generated < max_tokens:
#         B = ctrl.B
#         draft_ids, draft_token_strs = drafter_step(prefix_text, B, prefix_ids)
#
#         threading.Thread(target=offload, args=(draft_ids, prefix_text), daemon=True).start()
#
#         acc_len, dlen, bonus, recovered = inflight.get()
#         batch_r = acc_len / max(dlen, 1)
#         ctrl.update(batch_r)
#
#         # 1) Commit accepted token strings -> text
#         if acc_len > 0:
#             accepted_text = tok.convert_tokens_to_string(draft_token_strs[:acc_len])
#             prefix_text += accepted_text
#             generated += acc_len
#             prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
#
#         # 2) Commit recovered token on first mismatch (critical to avoid stalling)
#         if recovered is not None and (acc_len < dlen):
#             rec_tok = tok.convert_ids_to_tokens([recovered])[0]
#             rec_text = tok.convert_tokens_to_string([rec_tok])
#             prefix_text += rec_text
#             generated += 1
#             prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
#
#         # 3) Commit bonus if all accepted
#         if bonus is not None:
#             bonus_tok = tok.convert_ids_to_tokens([bonus])[0]
#             bonus_text = tok.convert_tokens_to_string([bonus_tok])
#             prefix_text += bonus_text
#             generated += 1
#             prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
#
#         # PAS update
#         bucket = bucketize_prefix(prefix_ids[:-acc_len]) if acc_len > 0 else bucketize_prefix(prefix_ids)
#         for i in range(min(acc_len, dlen)):
#             pas_update(bucket, draft_ids[i], True)
#         if acc_len < dlen and dlen > 0:
#             pas_update(bucket, draft_ids[acc_len], False)
#
#         print(f"[{name}] r={ctrl.r:.2f}  B={ctrl.B:02d}  batch_r={batch_r:.2f}  gen={generated}")
#
#     print(f"[{name}] done. tail: {prefix_text[-200:]}")
#
#
# if __name__ == "__main__":
#     run_edge("edge-A", "Explain gradient descent in simple terms. ", max_tokens=96)
#

# drafter_edge.py
import json as pyjson
import math
import os
import queue
import threading
import uuid
from typing import Dict, List, Tuple

import httpx
import redis
import ujson as json
from transformers import AutoTokenizer

# -----------------------
# Config
# -----------------------
# vLLM OpenAI server used by the *drafter* (your smaller model)
OPENAI_BASE = os.getenv("OPENAI_BASE", "http://127.0.0.1:8001/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/opt-350m")
TOKENIZER = MODEL_NAME

# verify_service FastAPI endpoint
VERIFY_URL = os.getenv("VERIFY_URL", "http://127.0.0.1:7001")

# Drafting sampling knobs (per edge, set via env)
TEMPERATURE = float(os.getenv("TEMP", "0.7"))
TOP_P       = float(os.getenv("TOPP", "0.9"))
TOP_K       = int(os.getenv("TOPK", "40"))

# Shared PAS knobs (via Redis)
EDGE_ID     = os.getenv("EDGE_ID", "edge-unknown")
REDIS_HOST  = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB    = int(os.getenv("REDIS_DB", "0"))

# PAS influence; start small, increase once acceptance is healthy
ALPHA_PAS   = float(os.getenv("ALPHA_PAS", "0.2"))  # strength of logit_bias
TOP_K_PAS   = int(os.getenv("TOP_K_PAS", "4"))      # how many tokens per bucket

# Soft cap per bucket to prevent unbounded growth (keeps storage tidy)
BUCKET_MAX_ITEMS = int(os.getenv("BUCKET_MAX_ITEMS", "64"))

# -----------------------
# Init
# -----------------------
tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
print("[EDGE] model:", MODEL_NAME)
print("[EDGE] bos_token_id:", tok.bos_token_id, "eos_token_id:", tok.eos_token_id)
print(f"[EDGE] OPENAI_BASE={OPENAI_BASE} VERIFY_URL={VERIFY_URL} EDGE_ID={EDGE_ID}")

R = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# -----------------------
# Helpers
# -----------------------
def bucketize_prefix(ids: List[int]) -> Tuple[int, int]:
    a = ids[-2] if len(ids) >= 2 else -1
    b = ids[-1] if len(ids) >= 1 else -1
    return a, b

def _key(a: int, b: int) -> str:
    return f"pas:{a}:{b}"          # Redis ZSET: token_id -> score

def _meta(a: int, b: int) -> str:
    return f"pasmeta:{a}:{b}"      # Redis HASH: token_id -> {"acc":X,"rej":Y,"last_edge":...}

def _score(acc: int, rej: int) -> float:
    total = max(1, acc + rej)
    return (acc / total) * total   # acceptance rate * support

def _acc_q8(acc: int, rej: int) -> int:
    total = max(1, acc + rej)
    p = acc / total
    return max(5, min(250, int(round(p * 255))))

def pas_get_topk(bucket: Tuple[int, int], k: int = TOP_K_PAS):
    a, b = bucket
    # Highest scores first
    items = R.zrevrange(_key(a, b), 0, k - 1, withscores=True)
    out = []
    from_edges = set()
    for tid_str, _sc in items:
        meta_raw = R.hget(_meta(a, b), tid_str)
        acc_q8 = 128
        if meta_raw:
            m = pyjson.loads(meta_raw)
            acc_q8 = _acc_q8(m.get("acc", 0), m.get("rej", 0))
            if "last_edge" in m:
                from_edges.add(m["last_edge"])
        out.append((int(tid_str), (acc_q8, int(round(_sc)))))
    if out:
        print(f"[EDGE] PAS_TOPK bucket={bucket} from_edges={list(from_edges)}")
    return out

def pas_update(bucket: Tuple[int, int], token_id: int, accepted: bool):
    a, b = bucket
    tid = str(token_id)

    pipe = R.pipeline()
    meta_key = _meta(a, b)
    score_key = _key(a, b)

    # read & update meta
    meta_raw = R.hget(meta_key, tid)
    if meta_raw:
        m = pyjson.loads(meta_raw)
    else:
        m = {"acc": 0, "rej": 0}
    if accepted:
        m["acc"] = m.get("acc", 0) + 1
    else:
        m["rej"] = m.get("rej", 0) + 1
    m["last_edge"] = EDGE_ID

    # new score
    sc = _score(m["acc"], m["rej"])

    # write back
    pipe.hset(meta_key, tid, pyjson.dumps(m))
    pipe.zadd(score_key, {tid: sc})
    # keep top-K_PAS window + small buffer
    if BUCKET_MAX_ITEMS > 0:
        # Remove lowest-ranked items beyond cap
        extra = BUCKET_MAX_ITEMS - 1
        pipe.zremrangebyrank(score_key, 0, -(extra + 1))
    pipe.execute()

# -----------------------
# OpenAI draft (vLLM)
# -----------------------
_seen_debug = False  # one-time shape print

def openai_sample_one_from_text(prefix_text: str, logit_bias: Dict[int, float]) -> Tuple[int, str]:
    """
    Ask vLLM (OpenAI completions) for ONE token continuation of prefix_text.
    Returns (token_id, token_str). Maps 'logprobs.tokens[0]' -> id via tokenizer.
    """
    global _seen_debug
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prefix_text,
        "max_tokens": 1,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,  # can be 0 for "no top-k"
        "logprobs": 1,
        "echo": False,
        "logit_bias": {str(k): v for k, v in logit_bias.items()},
        "repetition_penalty": 1.0,
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{OPENAI_BASE}/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    if not _seen_debug:
        print("[EDGE] /completions raw keys:", list(data.keys()))
        try:
            ch = data["choices"][0]
            print("[EDGE] choices[0].keys:", list(ch.keys()))
            lp = ch.get("logprobs", {})
            if isinstance(lp, dict):
                print("[EDGE] logprobs keys:", list(lp.keys()))
                print("[EDGE] sample logprobs snippet:",
                      {k: lp.get(k, None) for k in ["tokens", "top_logprobs", "text_offset"]})
        except Exception as e:
            print("[EDGE] debug print error:", e)
        _seen_debug = True

    choice = data["choices"][0]
    lp = choice.get("logprobs", {}) or {}
    tokens = lp.get("tokens") or []
    if tokens:
        token_str = tokens[0]  # e.g., 'Ċ'
        token_id = tok.convert_tokens_to_ids(token_str)
        return int(token_id), token_str

    # Fallback (rare): use text; map to first token string/id ourselves
    out_text = choice.get("text", "")
    ids = tok(out_text, add_special_tokens=False).input_ids
    if ids:
        tid = int(ids[0])
        tstr = tok.convert_ids_to_tokens([tid])[0]
        return tid, tstr
    return int(tok.eos_token_id), tok.convert_ids_to_tokens([tok.eos_token_id])[0]

# -----------------------
# Drafting loop
# -----------------------
def drafter_step(prefix_text: str, B: int, prefix_ids_for_pas: List[int]) -> Tuple[List[int], List[str]]:
    """
    Draft B tokens based on prefix_text.
    Returns (draft_token_ids, draft_token_strs).
    """
    out_ids: List[int] = []
    out_toks: List[str] = []

    print("[EDGE] draft_against_tail:", repr(prefix_text[-80:]))

    for i in range(B):
        bucket = bucketize_prefix(prefix_ids_for_pas + out_ids)
        # Gather PAS top-k and turn into logit_bias
        bias: Dict[int, float] = {}
        for tok_id, (acc_q8, _) in pas_get_topk(bucket):
            p_hat = max(0.05, acc_q8 / 255.0)
            bias[tok_id] = ALPHA_PAS * math.log(p_hat)

        # Predict next token from text (faithful spacing/newlines)
        tid, tstr = openai_sample_one_from_text(prefix_text + tok.convert_tokens_to_string(out_toks), bias)
        if i == 0:
            print(f"[EDGE] first_drafted_id={tid} token_str={repr(tstr)}")
        out_ids.append(tid)
        out_toks.append(tstr)
    return out_ids, out_toks

class Controller:
    def __init__(self, B=16, Bmin=4, Bmax=64, r_hi=0.8, r_lo=0.4, alpha=0.2):
        self.B = B
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.r = 0.6
        self.alpha = alpha
        self.r_hi = r_hi
        self.r_lo = r_lo
        self.hi = 0
        self.lo = 0

    def update(self, batch_r: float):
        self.r = (1 - self.alpha) * self.r + self.alpha * batch_r
        if batch_r < 0.1:
            self.B = max(self.B // 2, self.Bmin)
            self.hi = self.lo = 0
            return
        if self.r >= self.r_hi:
            self.hi += 1
            self.lo = 0
            if self.hi >= 2:
                self.B = min(self.B + 8, self.Bmax)
        elif self.r <= self.r_lo:
            self.lo += 1
            self.hi = 0
            if self.lo >= 2:
                self.B = max(self.B // 2, self.Bmin)
        else:
            self.hi = self.lo = 0

# -----------------------
# End-to-end run
# -----------------------
def run_edge(name: str, prompt: str, max_tokens=128):
    seq_id = f"{EDGE_ID}-{uuid.uuid4().hex[:6]}"
    print(f"seq_id={seq_id}")

    # Canonical source of truth = text
    prefix_text = prompt
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids

    print(f"[EDGE] prompt_text({len(prompt)} chars)='{prompt[:80]}...'")
    print("[EDGE] initial prefix_ids[:10]:", prefix_ids[:10])

    ctrl = Controller(B=16, Bmin=4, Bmax=64)
    inflight: queue.Queue = queue.Queue()
    generated = 0

    def offload(draft_ids: List[int], prompt_text_for_verify: str):
        # Build prompt_ids from the same text we used for drafting
        prompt_ids = tok(prompt_text_for_verify, add_special_tokens=False).input_ids
        print("[EDGE] offload_tail:", repr(tok.decode(prompt_ids, skip_special_tokens=False)[-80:]))

        payload = {
            "seq_id": seq_id,
            "prompt_ids": prompt_ids,
            "draft_token_ids": draft_ids,
            "want_bonus": True,
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{VERIFY_URL}/verify", json=payload)
            r.raise_for_status()
            resp = r.json()
        inflight.put((resp["accepted_len"], len(draft_ids), resp.get("bonus_token_id")))

    while generated < max_tokens:
        B = ctrl.B
        # Draft using TEXT (keeps spacing/newlines faithful)
        draft_ids, draft_token_strs = drafter_step(prefix_text, B, prefix_ids)

        # Offload verification using the exact text we drafted against
        threading.Thread(target=offload, args=(draft_ids, prefix_text), daemon=True).start()

        # Wait for verifier
        acc_len, dlen, bonus = inflight.get()
        batch_r = acc_len / max(dlen, 1)
        ctrl.update(batch_r)

        # Commit accepted tokens to TEXT (convert token strings -> text)
        if acc_len > 0:
            accepted_text = tok.convert_tokens_to_string(draft_token_strs[:acc_len])
            prefix_text += accepted_text
            generated += acc_len
            # Recompute ids from text for PAS buckets
            prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids

        # Commit bonus (if any) to TEXT
        if bonus is not None:
            bonus_tok = tok.convert_ids_to_tokens([bonus])[0]
            bonus_text = tok.convert_tokens_to_string([bonus_tok])
            prefix_text += bonus_text
            generated += 1
            prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids

        # PAS update: reward accepted, penalize first rejected
        # Use the *bucket before the drafted token* (same as we did when sampling)
        base_bucket = bucketize_prefix(prefix_ids[:-acc_len]) if acc_len > 0 else bucketize_prefix(prefix_ids)
        for i in range(min(acc_len, dlen)):
            pas_update(base_bucket, draft_ids[i], True)
        if acc_len < dlen and dlen > 0:
            pas_update(base_bucket, draft_ids[acc_len], False)

        print(f"[{name}] r={ctrl.r:.2f}  B={ctrl.B:02d}  batch_r={batch_r:.2f}  gen={generated}")

    print(f"[{name}] done. tail: {prefix_text[-200:]}")


if __name__ == "__main__":
    # Example single run; launch multiple edges with different env to see PAS exchange
    # e.g.:
    #   EDGE_ID=edge-A TEMP=0.3 TOPP=0.9 TOPK=40 python drafter_edge.py
    #   EDGE_ID=edge-B TEMP=0.7 TOPP=0.9 TOPK=40 python drafter_edge.py
    #   EDGE_ID=edge-C TEMP=0.9 TOPP=0.95 TOPK=50 python drafter_edge.py
    edge_id = os.getenv("EDGE_ID", "edge-A")
    run_edge("edge-A", "Explain gradient descent in simple terms.", max_tokens=96)
