"""离线评估 v1(PRD §6.2/§6.3):检索层指标 + 拒答判定正确率 + 拒答阈值标定。

评估集:eval/evalset_v1.jsonl,50 条标注问答(四类):
  fact(事实型)/synthesis(综合型)/multi_hop(多跳) — 库内,标注答案所在块;
  adversarial_refusal(对抗拒答) — 库外,应触发拒答。

指标(PRD §6.2):
  检索层  Hit Rate@5(精排 Top-5 是否命中标注块)、MRR(首个命中的倒数排名);
  拒答层  库外拒答率(应 ≥95%)、库内误杀率;
  阈值标定 扫描候选阈值,按「库内通过率 + 库外拒获率」平衡准确率择优,
            同分时宁严勿松取更高阈值(路线图 M1 风险条目)。

用法:
  python eval.py                    # 全量评估 + 阈值扫描(用 .env 的 REFUSE_THRESHOLD 做基准)
  python eval.py --threshold 0.6    # 指定基准阈值

生成层指标(RAGAS Faithfulness / Answer Relevancy)需 LLM 裁判,留待 v2 接入。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

EVALSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalset_v1.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RECALL_TOP_N = int(os.environ.get("RECALL_TOP_N", "50"))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "5"))
DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", "0.7"))


def load_evalset(path: str = EVALSET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(threshold: float, items: list[dict]) -> list[dict]:
    """对每条标注跑「检索→精排」(不调用 LLM),记录命中情况与最高分。"""
    from engine.reranker import BGEReranker
    from engine.retriever import BGERetriever

    retriever = BGERetriever()
    reranker = BGEReranker()
    assert retriever.load_existing(), "Chroma 无索引,先 python -m engine.pipeline 建索引"
    print(f"索引 {retriever.col.count()} 块,评估 {len(items)} 条(检索+精排,不调 LLM)...")

    records = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        recalled = retriever.search(q, top_k=RECALL_TOP_N,
                                    dense_weight=DENSE_WEIGHT)
        ranked = reranker.rerank(q, [c for c, _ in recalled],
                                 top_k=RERANK_TOP_K)
        hit_ids = [c.cid for c, _ in ranked]
        top_score = ranked[0][1] if ranked else 0.0
        rel = set(item["relevant"])
        first_hit_rank = next((r for r, cid in enumerate(hit_ids, 1)
                               if cid in rel), None)
        records.append({
            **item,
            "top_score": round(top_score, 4),
            "hit_ids": hit_ids,
            "hit_at_5": first_hit_rank is not None,
            "rr": round(1.0 / first_hit_rank, 4) if first_hit_rank else 0.0,
            "refused_at_threshold": top_score < threshold,
        })
        print(f"  [{i:2d}/{len(items)}] {item['id']} top={top_score:.3f} "
              f"{'命中' if first_hit_rank else ('拒答域' if not item['in_library'] else 'MISS')}")
    return records


def summarize(records: list[dict], threshold: float) -> dict:
    inlib = [r for r in records if r["in_library"]]
    outlib = [r for r in records if not r["in_library"]]
    by_type = {}
    for r in inlib:
        t = by_type.setdefault(r["type"], {"n": 0, "hit": 0, "rr": 0.0})
        t["n"] += 1
        t["hit"] += r["hit_at_5"]
        t["rr"] += r["rr"]
    summary = {
        "threshold": threshold,
        "in_library": {
            "n": len(inlib),
            "hit_rate@5": round(sum(r["hit_at_5"] for r in inlib) / len(inlib), 4),
            "mrr": round(sum(r["rr"] for r in inlib) / len(inlib), 4),
            "误杀数": sum(1 for r in inlib if r["refused_at_threshold"]),
        },
        "out_library": {
            "n": len(outlib),
            "拒答率": round(sum(r["refused_at_threshold"] for r in outlib)
                          / len(outlib), 4),
        },
        "by_type": {t: {"n": v["n"],
                        "hit_rate@5": round(v["hit"] / v["n"], 4),
                        "mrr": round(v["rr"] / v["n"], 4)}
                    for t, v in by_type.items()},
    }
    return summary


def scan_thresholds(records: list[dict]) -> list[dict]:
    """扫描候选阈值:库内通过率(不误杀)+ 库外拒获率,平衡准确率择优。"""
    inlib = [r["top_score"] for r in records if r["in_library"]]
    outlib = [r["top_score"] for r in records if not r["in_library"]]
    rows = []
    for t in [i / 100 for i in range(5, 96, 5)]:
        keep = sum(s >= t for s in inlib) / len(inlib)      # 库内通过率
        catch = sum(s < t for s in outlib) / len(outlib)    # 库外拒获率
        rows.append({
            "threshold": t,
            "库内通过率": round(keep, 4),
            "库外拒获率": round(catch, 4),
            "平衡准确率": round((keep + catch) / 2, 4),
        })
    return rows


def recommend(rows: list[dict]) -> float:
    """平衡准确率最高者;同分宁严勿松,取更高阈值。"""
    best = max(r["平衡准确率"] for r in rows)
    candidates = [r["threshold"] for r in rows if r["平衡准确率"] == best]
    return max(candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float,
                    default=float(os.environ.get("REFUSE_THRESHOLD", "0.5")))
    ap.add_argument("--start", type=int, default=0, help="评估集起始下标(分批跑)")
    ap.add_argument("--end", type=int, default=None, help="评估集结束下标(不含)")
    ap.add_argument("--combine", action="store_true",
                    help="合并 results/partial_*.json 出总报告(不再跑检索)")
    args = ap.parse_args()

    if args.combine:
        import glob
        records = {}
        for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "partial_*.json"))):
            for r in json.load(open(p, encoding="utf-8")):
                records[r["id"]] = r
        records = [records[k] for k in sorted(records)]
        assert len(records) == len(load_evalset()), \
            f"分批结果不完整:{len(records)}/{len(load_evalset())}"
    else:
        items = load_evalset()[args.start:args.end]
        records = run_eval(args.threshold, items)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        part = os.path.join(
            RESULTS_DIR, f"partial_{items[0]['id']}_{items[-1]['id']}.json")
        with open(part, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"本批明细已存:{part}(全部跑完后加 --combine 出总报告)")
        if len(items) < len(load_evalset()):
            return

    summary = summarize(records, args.threshold)
    rows = scan_thresholds(records)
    rec = recommend(rows)

    print("\n===== 检索层指标(库内) =====")
    print(f"Hit Rate@5 = {summary['in_library']['hit_rate@5']:.1%}  "
          f"MRR = {summary['in_library']['mrr']:.3f}  "
          f"(阈值 {args.threshold} 下误杀 {summary['in_library']['误杀数']} 条)")
    for t, v in summary["by_type"].items():
        print(f"  {t:10s} n={v['n']:2d}  hit@5={v['hit_rate@5']:.1%}  mrr={v['mrr']:.3f}")
    print(f"\n===== 拒答层(阈值 {args.threshold}) =====")
    print(f"库外拒答率 = {summary['out_library']['拒答率']:.1%}  "
          f"(PRD 目标 ≥ 95%)")
    print("\n===== 阈值扫描(库内通过率 / 库外拒获率 / 平衡准确率) =====")
    for r in rows:
        mark = " ◀ 当前" if r["threshold"] == args.threshold else ""
        mark += " ★ 推荐" if r["threshold"] == rec else ""
        print(f"  {r['threshold']:.2f}  {r['库内通过率']:.1%} / "
              f"{r['库外拒获率']:.1%} / {r['平衡准确率']:.1%}{mark}")
    print(f"\n推荐阈值:{rec}(当前 {args.threshold})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR,
                       f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "recommended_threshold": rec,
                   "threshold_scan": rows, "records": records},
                  f, ensure_ascii=False, indent=2)
    print(f"明细已存:{out}")


if __name__ == "__main__":
    main()
