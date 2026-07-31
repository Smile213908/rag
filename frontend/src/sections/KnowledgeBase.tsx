// 知识库管理页:文档列表 / 上传(全流程动态进度) / 重建 / 删除(docs/03 §3)
import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, Loader2, RefreshCw, RotateCw, Trash2, Upload, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  deleteDocument,
  getDocumentStatus,
  listDocuments,
  rebuildDocument,
  uploadDocument,
} from '@/lib/api'
import type { DocItem, DocStatus } from '@/types'

const STATUS: Record<string, { label: string; cls: string }> = {
  done: { label: '已完成', cls: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
  indexing: { label: '索引中', cls: 'bg-blue-50 text-blue-700 border-blue-200' },
  failed: { label: '失败', cls: 'bg-red-50 text-red-700 border-red-200' },
}

// RAG 处理流程分段(与后端 task.stage 对齐)
const STAGES = [
  { key: 'uploaded', label: '上传落盘' },
  { key: 'parsing', label: '解析文档' },
  { key: 'chunked', label: '结构分块' },
  { key: 'encoding', label: '向量编码入库' },
  { key: 'finalizing', label: 'BM25+缓存收尾' },
  { key: 'done', label: '完成' },
]

export default function KnowledgeBase() {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState('')
  // 当前正在跟踪的索引任务(上传/重建共用流程面板)
  const [task, setTask] = useState<{ docId: string; filename: string; st: DocStatus } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const reload = useCallback(async () => {
    try {
      const r = await listDocuments()
      setDocs(r.items)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  // 有索引中的文档时轮询列表
  useEffect(() => {
    if (!docs.some((d) => d.status === 'indexing')) return
    const t = setInterval(() => void reload(), 2500)
    return () => clearInterval(t)
  }, [docs, reload])

  // 任务级轮询:实时拉分段进度
  useEffect(() => {
    if (!task || task.st.status === 'done' || task.st.status === 'failed') return
    pollRef.current = setInterval(async () => {
      try {
        const st = await getDocumentStatus(task.docId)
        setTask((t) => (t ? { ...t, st } : t))
        if (st.status === 'done' || st.status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current)
          void reload()
        }
      } catch {
        /* 单次轮询失败忽略,下拍重试 */
      }
    }, 1200)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [task, reload])

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusyId(file.name)
    try {
      const r = await uploadDocument(file)
      setTask({
        docId: r.doc_id,
        filename: file.name,
        st: { doc_id: r.doc_id, status: 'indexing', progress: 0.1, stage: 'uploaded' },
      })
      await reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '上传失败')
    } finally {
      setBusyId('')
    }
  }

  async function onRebuild(d: DocItem) {
    setBusyId(d.doc_id)
    try {
      await rebuildDocument(d.doc_id)
      setTask({
        docId: d.doc_id,
        filename: d.filename,
        st: { doc_id: d.doc_id, status: 'indexing', progress: 0.05, stage: 'queued' },
      })
      await reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '重建失败')
    } finally {
      setBusyId('')
    }
  }

  async function onDelete(d: DocItem) {
    if (!window.confirm(`确认删除「${d.filename}」?\n将级联删除其 ${d.chunks} 个索引块与源文件,不可恢复。`))
      return
    setBusyId(d.doc_id)
    try {
      await deleteDocument(d.doc_id)
      await reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除失败')
    } finally {
      setBusyId('')
    }
  }

  return (
    <div className="mx-auto h-full max-w-4xl overflow-y-auto px-4 py-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">知识库管理</h2>
          <p className="text-xs text-slate-400">共 {docs.length} 篇文档 · 支持 .md / .txt / .pdf</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => void reload()} disabled={loading}>
            <RefreshCw className={`mr-1 h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            刷新
          </Button>
          <Button size="sm" onClick={() => fileRef.current?.click()} disabled={!!busyId}>
            {busyId && !docs.find((d) => d.doc_id === busyId) ? (
              <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Upload className="mr-1 h-3.5 w-3.5" />
            )}
            上传文档
          </Button>
          <input
            ref={fileRef}
            type="file"
            accept=".md,.txt,.pdf"
            className="hidden"
            onChange={(e) => void onUpload(e)}
          />
        </div>
      </div>

      {error && (
        <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}(请确认后端 FastAPI 已启动:uvicorn api.server:app --port 8080)
        </div>
      )}

      {/* RAG 处理流程动态进度 */}
      {task && <ProgressPanel task={task} onClose={() => setTask(null)} />}

      <div className="rounded-xl border bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>文档</TableHead>
              <TableHead className="w-20 text-right">块数</TableHead>
              <TableHead className="w-24">状态</TableHead>
              <TableHead className="w-40">上传时间</TableHead>
              <TableHead className="w-36 text-right">操作</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {docs.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={5} className="py-10 text-center text-sm text-slate-400">
                  知识库为空,点击右上角「上传文档」开始
                </TableCell>
              </TableRow>
            )}
            {docs.map((d) => {
              const st = STATUS[d.status] ?? STATUS.done
              const busy = busyId === d.doc_id
              return (
                <TableRow key={d.doc_id}>
                  <TableCell className="font-medium text-slate-700">{d.filename}</TableCell>
                  <TableCell className="text-right text-slate-500">{d.chunks}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className={st.cls}>
                      {d.status === 'indexing' && (
                        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      )}
                      {st.label}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-slate-400">
                    {d.uploaded_at ?? '—'}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      title="重建索引"
                      disabled={busy || d.status === 'indexing'}
                      onClick={() => void onRebuild(d)}
                    >
                      {busy ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RotateCw className="h-4 w-4 text-slate-500" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      title="删除"
                      disabled={busy || d.status === 'indexing'}
                      onClick={() => void onDelete(d)}
                    >
                      <Trash2 className="h-4 w-4 text-red-400" />
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

// ---------- RAG 处理流程面板:分段步骤 + 实时进度 ----------
function ProgressPanel({
  task,
  onClose,
}: {
  task: { docId: string; filename: string; st: DocStatus }
  onClose: () => void
}) {
  const { st } = task
  const curIdx = st.status === 'done'
    ? STAGES.length - 1
    : Math.max(0, STAGES.findIndex((s) => s.key === st.stage))
  const failed = st.status === 'failed'
  const pct = Math.round(st.progress * 100)

  return (
    <div className="mb-4 rounded-xl border bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-slate-800">{task.filename}</div>
          <div className="text-xs text-slate-400">
            {failed ? (
              <span className="text-red-500">处理失败:{st.error || '未知错误'}</span>
            ) : st.status === 'done' ? (
              <span className="text-emerald-600">索引完成,共 {st.chunks_total ?? '—'} 个块</span>
            ) : (
              <>
                {STAGES[curIdx]?.label ?? '排队中'}中
                {st.stage === 'encoding' && st.chunks_total
                  ? `(${st.chunks_done ?? 0}/${st.chunks_total} 块)`
                  : ''}
                · {pct}%
              </>
            )}
          </div>
        </div>
        {(st.status === 'done' || failed) && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            收起
          </Button>
        )}
      </div>

      {/* 分段步骤条 */}
      <div className="mb-3 flex items-center">
        {STAGES.map((s, i) => {
          const done = !failed && i < curIdx
          const current = !failed && st.status !== 'done' && i === curIdx
          const reached = done || (st.status === 'done') || current
          return (
            <div key={s.key} className="flex flex-1 items-center last:flex-none">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`flex h-6 w-6 items-center justify-center rounded-full border text-xs ${
                    failed && i === curIdx
                      ? 'border-red-300 bg-red-50 text-red-500'
                      : done || st.status === 'done'
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-600'
                        : current
                          ? 'border-blue-300 bg-blue-50 text-blue-600'
                          : 'border-slate-200 text-slate-300'
                  }`}
                >
                  {failed && i === curIdx ? (
                    <XCircle className="h-3.5 w-3.5" />
                  ) : done || st.status === 'done' ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : current ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    i + 1
                  )}
                </div>
                <span
                  className={`whitespace-nowrap text-[11px] ${
                    reached && !failed ? 'text-slate-600' : failed && i === curIdx ? 'text-red-500' : 'text-slate-300'
                  }`}
                >
                  {s.label}
                </span>
              </div>
              {i < STAGES.length - 1 && (
                <div
                  className={`mx-1 mb-4 h-px flex-1 ${
                    i < curIdx || st.status === 'done' ? 'bg-emerald-300' : 'bg-slate-200'
                  }`}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* 进度条 */}
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            failed ? 'bg-red-400' : st.status === 'done' ? 'bg-emerald-500' : 'bg-blue-500'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
