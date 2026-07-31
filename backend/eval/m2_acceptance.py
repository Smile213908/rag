# -*- coding: utf-8 -*-
"""M2 验收(PRD v1.1 口径,2026-07-31):流式实测 TTFT/端到端/路由/缓存/多轮/拒答回归/看板。

判定(对齐 PRD v1.1 §2.2/§5 + M2 交付物):
- TTFT(首 delta,排除缓存命中与拒答)P95 ≤ 3s;
- 简单问题(meta.model=快模型)端到端 P95 ≤ 30s;
- 答案缓存:重复问题(归一化同键)cache_hit=true、latency≈0、答案与首问一致;
- 多轮:追问 standalone_question 与原文不同(指代消解生效);
- 库外拒答回归(抽样 A01/A03/A07):refused=true(目标 100%);
- 看板:GET /admin/stats/overview 返回核心指标键齐全。

用法(需先起 Chroma:docker compose up -d):
  python eval/m2_acceptance.py --serve --max 6   # 脚本内起 uvicorn 跑一批(断点续跑)
  python eval/m2_acceptance.py --combine          # 汇总报告
结果追加写 eval/results/m2_accept.jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALSET = os.path.join(ROOT, "eval", "evalset_v1.jsonl")
OUT = os.path.join(ROOT, "eval", "results", "m2_accept.jsonl")
BASE = "http://localhost:8080/api/v1"
FAST_MODEL = os.environ.get("FAST_LLM_MODEL", "kimi-for-coding")

# 验收用例(事实/综合/对抗题取自 evalset_v1,answer_point 一并带出)
CASES = [
    {"id": "F15", "type": "fact_simple"},
    {"id": "F16", "type": "fact_simple"},
    {"id": "F17", "type": "fact_simple"},
    {"id": "F19", "type": "fact_simple"},
    {"id": "F21", "type": "fact_simple"},
    {"id": "F15-cache", "type": "cache_repeat",
     "question": "公司工作日的上下班时间是几点", "ref": "F15"},  # 去问号,归一化同键
    {"id": "MT1", "type": "multiturn_first", "ref": "F07"},       # 新会话首问
    {"id": "MT2", "type": "multiturn_followup",
     "question": "那交通费呢?", "session_of": "MT1"},             # 同会话追问
    {"id": "A01", "type": "outlib_refuse"},
    {"id": "A03", "type": "outlib_refuse"},
    {"id": "A07", "type": "outlib_refuse"},
    {"id": "S01", "type": "synthesis"},
    {"id": "M03", "type": "complex_nolimit"},                     # 35字,超路由长度→默认模型
    {"id": "BOARD", "type": "admin_stats"},
]


def load_evalset() -> dict:
    items = [json.loads(l) for l in open(EVALSET, encoding="utf-8") if l.strip()]
    return {i["id"]: i for i in items}


def sse_ask(question: str, session_id: str | None = None,
            timeout: int = 300) -> dict:
    """流式提问,记录 TTFT(首 delta 到达时间)。"""
    body = json.dumps({"question": question, "session_id": session_id},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/ask", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    meta, done, ttft_ms = None, None, None
    answer = ""
    event = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    payload = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if event == "meta":
                    meta = payload
                elif event == "delta":
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - t0) * 1000)
                    answer += payload.get("text", "")
                elif event == "done":
                    done = payload
    wall_ms = int((time.time() - t0) * 1000)
    if meta is None or done is None:
        raise RuntimeError("SSE 帧不完整")
    return {"meta": meta, "done": done, "answer": answer,
            "ttft_ms": ttft_ms, "wall_ms": wall_ms}


def get_json(path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(proc: subprocess.Popen, timeout: int = 240) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn 提前退出 code={proc.returncode}")
        try:
            r = get_json("/health", timeout=5)
            if r.get("code") == 0:
                print(f"服务就绪({int(time.time() - t0)}s)")
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("等待 /health 超时")


def run_case(case: dict, es: dict, sessions: dict) -> dict:
    t0 = time.time()
    ctype = case["type"]
    if ctype == "admin_stats":
        r = get_json("/admin/stats/overview")
        data = r.get("data") or {}
        keys = ["total_queries", "hit_rate", "refuse_rate",
                "thumbs_up_rate", "p95_latency_ms"]
        return {"id": case["id"], "type": ctype,
                "keys_present": all(k in data for k in keys),
                "data": data, "error": None,
                "wall_ms": int((time.time() - t0) * 1000)}

    q = case.get("question")
    ref = case.get("ref")
    if q is None:
        q = es[ref or case["id"]]["question"]
    item = es.get(ref) if ref else es.get(case["id"])
    sid = sessions.get(case.get("session_of")) if case.get("session_of") else None
    r = sse_ask(q, session_id=sid)
    meta, done = r["meta"], r["done"]
    if case["id"] == "MT1":
        sessions["MT1"] = meta.get("session_id")
    rec = {
        "id": case["id"], "type": ctype, "question": q,
        "answer_point": (item or {}).get("answer_point", ""),
        "refused": meta.get("refused"),
        "cache_hit": bool(meta.get("cache_hit")),
        "model": meta.get("model"),
        "standalone_question": meta.get("standalone_question"),
        "answer": done.get("answer", ""),
        "latency_ms": done.get("latency_ms"),
        "tokens": done.get("tokens"),
        "ttft_ms": r["ttft_ms"],
        "wall_ms": r["wall_ms"],
        "n_sources": len(meta.get("sources") or []),
        "error": None,
    }
    if ctype == "cache_repeat":
        rec["answer_match_ref"] = None  # combine 时比对
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="脚本内起 uvicorn(跑完即停)")
    ap.add_argument("--max", type=int, default=None, help="本批最多跑 N 条")
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    es = load_evalset()
    done_ids = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                done_ids.add(json.loads(l)["id"])
    todo = [c for c in CASES if c["id"] not in done_ids]
    if args.max:
        todo = todo[: args.max]
    if not todo:
        print("全部用例已完成,可直接 --combine")
        return

    proc = None
    if args.serve:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.server:app", "--port", "8080"],
            cwd=ROOT)
    try:
        if proc:
            print("等待服务加载模型…")
            wait_health(proc)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        sessions: dict = {}
        with open(OUT, "a", encoding="utf-8") as f:
            for n, case in enumerate(todo, 1):
                try:
                    rec = run_case(case, es, sessions)
                except Exception as exc:  # 单条失败不中断批次
                    rec = {"id": case["id"], "type": case["type"],
                           "error": str(exc)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                flag = ("ERR " if rec.get("error") else
                        "缓存" if rec.get("cache_hit") else
                        "拒答" if rec.get("refused") else "作答")
                print(f"[{n}/{len(todo)}] {case['id']} {flag} "
                      f"model={rec.get('model')} "
                      f"ttft={rec.get('ttft_ms')}ms "
                      f"e2e={(rec.get('latency_ms') or 0) / 1000:.1f}s "
                      f"{(rec.get('error') or '')[:80]}")
        print(f"已追加 {len(todo)} 条 → {OUT}")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("uvicorn 已停")


def _p(vals: list, q: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def combine() -> None:
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    ok = [r for r in recs if not r.get("error")]

    def pct(part, whole):
        return f"{part}/{whole} = {part / whole:.1%}" if whole else "—"

    # TTFT 样本:非缓存、非拒答、有 delta 的问答
    ttft = [r["ttft_ms"] for r in ok
            if r.get("ttft_ms") is not None and not r.get("cache_hit")
            and not r.get("refused")]
    # 简单问题端到端:路由到快模型的样本
    fast = [r for r in ok if r.get("model") == FAST_MODEL
            and not r.get("cache_hit") and not r.get("refused")]
    fast_e2e = [r["latency_ms"] for r in fast]
    # 缓存复核
    cache = next((r for r in ok if r["id"] == "F15-cache"), None)
    ref = next((r for r in ok if r["id"] == "F15"), None)
    cache_ok = bool(cache and ref and cache.get("cache_hit")
                    and (cache.get("latency_ms") or 1e9) < 500
                    and cache.get("answer") == ref.get("answer"))
    # 多轮复核
    mt2 = next((r for r in ok if r["id"] == "MT2"), None)
    mt_ok = bool(mt2 and mt2.get("standalone_question")
                 and mt2["standalone_question"] != mt2["question"])
    # 拒答回归
    outlib = [r for r in ok if r["type"] == "outlib_refuse"]
    refused_n = sum(1 for r in outlib if r.get("refused"))
    # 看板
    board = next((r for r in ok if r["type"] == "admin_stats"), None)
    board_ok = bool(board and board.get("keys_present"))

    print("\n========== M2 验收汇总(PRD v1.1 口径) ==========")
    print(f"TTFT P95        {round(_p(ttft, 0.95))}ms (样本{len(ttft)}, 目标 ≤3000ms)"
          f"  P50={round(_p(ttft, 0.5))}ms")
    print(f"简单问题端到端   P95={round(_p(fast_e2e, 0.95))}ms"
          f" (快模型样本{len(fast)}, 目标 ≤30000ms) P50={round(_p(fast_e2e, 0.5))}ms")
    print(f"答案缓存        {'✅' if cache_ok else '❌'}"
          f" (cache_hit={cache and cache.get('cache_hit')},"
          f" latency={cache and cache.get('latency_ms')}ms,"
          f" 答案一致={bool(cache and ref and cache.get('answer') == ref.get('answer'))})")
    print(f"多轮指代消解     {'✅' if mt_ok else '❌'}"
          f" (standalone={mt2 and mt2.get('standalone_question')!r})")
    print(f"库外拒答回归     {pct(refused_n, len(outlib))} (目标 100%)")
    print(f"看板接口        {'✅' if board_ok else '❌'}")
    routed = {m: sum(1 for r in ok if r.get("model") == m)
              for m in {r.get("model") for r in ok if r.get("model")}}
    print(f"路由分布        {routed}")
    print(f"错误: {[r['id'] for r in recs if r.get('error')] or '无'}")
    out = os.path.join(os.path.dirname(OUT),
                       f"m2_accept_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ttft_ms": ttft, "fast_e2e_ms": fast_e2e,
                   "cache_ok": cache_ok, "mt_ok": mt_ok,
                   "refuse": f"{refused_n}/{len(outlib)}",
                   "board_ok": board_ok, "records": recs},
                  f, ensure_ascii=False, indent=2)
    print(f"报告已存:{out}")


if __name__ == "__main__":
    main()
