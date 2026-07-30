// 知识库管理页:文档列表 / 上传 / 重建 / 删除(docs/03 §3)
import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, RefreshCw, RotateCw, Trash2, Upload } from 'lucide-react'
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
  listDocuments,
  rebuildDocument,
  uploadDocument,
} from '@/lib/api'
import type { DocItem } from '@/types'

const STATUS: Record<string, { label: string; cls: string }> = {
  done: { label: '已完成', cls: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
  indexing: { label: '索引中', cls: 'bg-blue-50 text-blue-700 border-blue-200' },
  failed: { label: '失败', cls: 'bg-red-50 text-red-700 border-red-200' },
}

export default function KnowledgeBase() {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

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

  // 有索引中的文档时轮询
  useEffect(() => {
    if (!docs.some((d) => d.status === 'indexing')) return
    const t = setInterval(() => void reload(), 2500)
    return () => clearInterval(t)
  }, [docs, reload])

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusyId(file.name)
    try {
      await uploadDocument(file)
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
