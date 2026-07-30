# -*- coding: utf-8 -*-
"""M1 验收取样(docs/04 验收标准):对评估集条目走 /chat/ask/sync 全链路。

判定规则(对齐 PRD §2.2,需先起服务 uvicorn api.server:app --port 8080):
- 库外(adversarial_refusal):refused=True 记通过(拒答率目标 ≥95%);
  未拒答即"误编",记录答案供人工复核;
- 库内事实型(fact):答案含 answer_point(去空白归一化后子串)记正确
  (Top-1 正确率目标 ≥90%,PRD 要求人工抽检——未命中的条目人工复核);
- 来源覆盖率:所有非拒答响应 sources 非空(目标 100%);
- 延迟:取 done 帧 latency_ms,分组算 P50/P95(目标 P95 ≤ 8s)。

用法:
  python eval/acceptance_run.py --type fact --start 0 --end 5   # 分批跑
  python eval/acceptance_run.py --combine                         # 汇总报告
结果追加写 eval/results/accept.jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALSET = os.path.join(ROOT, "eval", "evalset_v1.jsonl")
OUT = os.path.join(ROOT, "eval", "results", "accept.jsonl")
BASE = "http://localhost:8080/api/v1"


def norm(s: str) -> str:
    """去空白 + 全角转半角,用于 answer_point 子串匹配。"""
    s = re.sub(r"\s+", "", s)
    return s.translate(str.maketrans(
        "０１２３４５６７８９（），。：", "0123456789(),.:"))


def ask(question: str, timeout: int = 180) -> dict:
    body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/ask/sync", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("code") != 0:
        raise RuntimeError(f"ask 失败: {payload}")
    return payload["data"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default=None, help="只跑该类型(fact/adversarial_refusal)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    items = [json.loads(l) for l in open(EVALSET, encoding="utf-8") if l.strip()]
    if args.type:
        items = [i for i in items if i["type"] == args.type]
    items = items[args.start:args.end]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "a", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            t0 = time.time()
            try:
                r = ask(item["question"])
                rec = {
                    "id": item["id"], "type": item["type"],
                    "in_library": item["in_library"],
                    "question": item["question"],
                    "answer_point": item.get("answer_point", ""),
                    "refused": r["refused"],
                    "answer": r["answer"],
                    "n_sources": len(r.get("sources") or []),
                    "latency_ms": r["latency_ms"],
                    "point_hit": (norm(item.get("answer_point", "")) in norm(r["answer"]))
                                 if item["in_library"] else None,
                    "error": None,
                }
            except Exception as exc:  # 单条失败不中断批次
                rec = {"id": item["id"], "type": item["type"],
                       "in_library": item["in_library"],
                       "question": item["question"], "error": str(exc)}
            rec["wall_ms"] = int((time.time() - t0) * 1000)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = ("拒答" if rec.get("refused") else
                    ("命中" if rec.get("point_hit") else
                     ("错答?" if rec.get("point_hit") is False else "ERR")))
            print(f"[{n}/{len(items)}] {item['id']} {flag} "
                  f"{rec.get('latency_ms', 0) / 1000:.1f}s {rec.get('error') or ''}")
    print(f"已追加 {len(items)} 条 → {OUT}")


def combine() -> None:
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    ok = [r for r in recs if not r.get("error")]
    outlib = [r for r in ok if not r["in_library"]]
    facts = [r for r in ok if r["in_library"] and r["type"] == "fact"]

    def pct(part, whole):
        return f"{part}/{whole} = {part / whole:.1%}" if whole else "—"

    def p(vals, q):
        vals = sorted(vals)
        if not vals:
            return 0
        k = (len(vals) - 1) * q
        lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    refuse_rate = sum(r["refused"] for r in outlib) / len(outlib) if outlib else 0
    correct = sum(1 for r in facts if r["point_hit"])
    src_ok = sum(1 for r in facts if not r["refused"] and r["n_sources"] > 0)
    src_total = sum(1 for r in facts if not r["refused"])
    lat_fact = [r["latency_ms"] for r in facts]
    report = {
        "库外拒答率": refuse_rate,
        "事实型Top1正确率": correct / len(facts) if facts else 0,
        "来源覆盖率": src_ok / src_total if src_total else 0,
        "延迟ms": {"fact_P50": round(p(lat_fact, 0.5)),
                   "fact_P95": round(p(lat_fact, 0.95)),
                   "fact_max": max(lat_fact) if lat_fact else 0},
        "误杀(库内被拒)": [r["id"] for r in facts if r["refused"]],
        "误编(库外未拒)": [r["id"] for r in outlib if not r["refused"]],
        "未命中answer_point": [r["id"] for r in facts
                               if not r["refused"] and not r["point_hit"]],
        "errors": [r["id"] for r in recs if r.get("error")],
    }
    print("\n========== M1 验收汇总 ==========")
    print(f"库外拒答率      {pct(sum(r['refused'] for r in outlib), len(outlib))}  (目标 ≥95%)")
    print(f"事实型Top1正确率 {pct(correct, len(facts))}  (目标 ≥90%)")
    print(f"来源覆盖率      {pct(src_ok, src_total)}  (目标 100%)")
    print(f"事实型延迟      P50={report['延迟ms']['fact_P50']}ms "
          f"P95={report['延迟ms']['fact_P95']}ms max={report['延迟ms']['fact_max']}ms (目标 P95≤8000)")
    print(f"误杀: {report['误杀(库内被拒)'] or '无'}  误编: {report['误编(库外未拒)'] or '无'}")
    print(f"未命中要点: {report['未命中answer_point'] or '无'}  错误: {report['errors'] or '无'}")
    out = os.path.join(os.path.dirname(OUT),
                       f"accept_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "records": recs}, f, ensure_ascii=False, indent=2)
    print(f"报告已存:{out}")


if __name__ == "__main__":
    main()
