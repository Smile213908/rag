// 运营看板(docs/03 §4):核心指标卡 + 高频问题 + bad case 闭环
import { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, Loader2, RefreshCw, ThumbsDown } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  fetchBadCases,
  fetchHotQuestions,
  fetchOverview,
  resolveBadCase,
} from '@/lib/api'
import type { BadCase, HotQuestion, Overview } from '@/types'
import { ISSUE_TYPES } from '@/types'

const issueLabel = (v: string | null) =>
  ISSUE_TYPES.find((t) => t.value === v)?.label ?? (v ? v : '未归因')

export default function Dashboard() {
  const [ov, setOv] = useState<Overview | null>(null)
  const [hot, setHot] = useState<HotQuestion[]>([])
  const [cases, setCases] = useState<BadCase[]>([])
  const [tab, setTab] = useState<'open' | 'resolved'>('open')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')

  const reload = useCallback(async () => {
    try {
      const [o, h, c] = await Promise.all([
        fetchOverview(),
        fetchHotQuestions(),
        fetchBadCases(tab),
      ])
      setOv(o)
      setHot(h)
      setCases(c)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [tab])

  useEffect(() => {
    setLoading(true)
    void reload()
  }, [reload])

  async function onResolve(c: BadCase) {
    const action = window.prompt(`处理「${c.comment ?? c.qa_id}」\n请输入处理动作(如:补充文档/调整阈值):`)
    if (!action) return
    setBusy(c.qa_id)
    try {
      await resolveBadCase(c.qa_id, action)
      await reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '操作失败')
    } finally {
      setBusy('')
    }
  }

  const cards = ov
    ? [
        { label: '总提问数', value: String(ov.total_queries), sub: `今日 ${ov.today_queries}` },
        { label: '命中率', value: pct(ov.hit_rate), sub: '作答占比(1-拒答率)' },
        { label: '拒答率', value: pct(ov.refuse_rate), sub: 'PRD 目标区间' },
        { label: '👍率', value: pct(ov.thumbs_up_rate), sub: 'M2 目标 ≥85%' },
        { label: 'P95 延迟', value: `${(ov.p95_latency_ms / 1000).toFixed(1)}s`, sub: '端到端' },
        { label: 'Token 消耗', value: fmtNum(ov.tokens_total), sub: '累计' },
        { label: '待处理 bad case', value: String(ov.open_bad_cases), sub: '👎 样本池' },
      ]
    : []

  return (
    <div className="mx-auto h-full max-w-5xl overflow-y-auto px-4 py-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">运营看板</h2>
          <p className="text-xs text-slate-400">指标来自 logs/*.jsonl 实时扫描</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void reload()} disabled={loading}>
          <RefreshCw className={`mr-1 h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          刷新
        </Button>
      </div>

      {error && (
        <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}(请确认后端已启动)
        </div>
      )}

      {/* 指标卡 */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
        {cards.map((c) => (
          <div key={c.label} className="rounded-xl border bg-white p-3">
            <div className="text-xs text-slate-400">{c.label}</div>
            <div className="mt-1 text-xl font-semibold text-slate-800">{c.value}</div>
            <div className="mt-0.5 text-[10px] text-slate-400">{c.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 高频问题 */}
        <div className="rounded-xl border bg-white p-4">
          <h3 className="mb-3 text-sm font-medium text-slate-700">高频问题 Top-10</h3>
          {hot.length === 0 && <p className="py-6 text-center text-xs text-slate-400">暂无提问数据</p>}
          <div className="space-y-2">
            {hot.map((h, i) => (
              <div key={i} className="flex items-center justify-between gap-2 text-sm">
                <span className="truncate text-slate-600">
                  <span className="mr-1.5 text-xs text-slate-300">{i + 1}.</span>
                  {h.question}
                </span>
                <Badge variant="secondary" className="shrink-0 text-xs">{h.count} 次</Badge>
              </div>
            ))}
          </div>
        </div>

        {/* bad case 池 */}
        <div className="rounded-xl border bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-medium text-slate-700">bad case 池</h3>
            <div className="flex gap-1 rounded-md bg-slate-100 p-0.5 text-xs">
              {(['open', 'resolved'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setTab(s)}
                  className={`rounded px-2 py-1 ${tab === s ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500'}`}
                >
                  {s === 'open' ? '待处理' : '已处理'}
                </button>
              ))}
            </div>
          </div>
          {cases.length === 0 && (
            <p className="py-6 text-center text-xs text-slate-400">
              {tab === 'open' ? '没有待处理的 bad case 🎉' : '暂无已处理记录'}
            </p>
          )}
          <div className="space-y-2.5">
            {cases.map((c) => (
              <div key={c.qa_id + c.ts} className="rounded-lg border border-slate-100 bg-slate-50 p-2.5">
                <div className="mb-1 flex items-center gap-1.5 text-xs">
                  <ThumbsDown className="h-3 w-3 text-red-400" />
                  <Badge variant="outline" className="text-[10px]">{issueLabel(c.issue_type)}</Badge>
                  <span className="text-slate-300">{c.ts}</span>
                </div>
                {c.comment && <p className="mb-1.5 text-xs text-slate-600">{c.comment}</p>}
                {c.status === 'resolved' && c.resolve_action && (
                  <p className="mb-1.5 text-xs text-emerald-600">处理:{c.resolve_action}</p>
                )}
                {c.status === 'open' && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    disabled={busy === c.qa_id}
                    onClick={() => void onResolve(c)}
                  >
                    {busy === c.qa_id ? (
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                    ) : (
                      <CheckCircle2 className="mr-1 h-3 w-3" />
                    )}
                    标记已处理
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function pct(v: number | null): string {
  return v === null ? '—' : `${(v * 100).toFixed(1)}%`
}

function fmtNum(n: number): string {
  return n >= 10000 ? `${(n / 10000).toFixed(1)}w` : String(n)
}
