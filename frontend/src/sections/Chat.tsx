// 问答主界面:会话管理侧栏(新增/删除/改标题/切换不失活)+ SSE 流式 + 引用溯源 + 👍👎 反馈(docs/03 §2)
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Check,
  FileText,
  Loader2,
  MessageSquarePlus,
  Pencil,
  Send,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  X,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import {
  askStream,
  clearSession,
  getSessionHistory,
  listSessions,
  renameSession,
  sendFeedback,
} from '@/lib/api'
import type { ChatMessage, SessionItem, SourceItem } from '@/types'
import { ISSUE_TYPES } from '@/types'

let seq = 0
const nid = () => `m${Date.now()}_${seq++}`
const NEW_KEY = 'new' // 未发送过消息的新会话在 store 里的键

export default function Chat() {
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  // 按会话分存消息:切换会话/页签都不丢状态;流式写入也按会话定向
  const [store, setStore] = useState<Record<string, ChatMessage[]>>({ [NEW_KEY]: [] })
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const activeKey = activeId ?? NEW_KEY
  const messages = store[activeKey] ?? []

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions())
    } catch {
      /* 后端未起时静默 */
    }
  }, [])

  useEffect(() => {
    void refreshSessions()
  }, [refreshSessions])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const patch = (key: string, id: string, p: Partial<ChatMessage>) =>
    setStore((s) => ({
      ...s,
      [key]: (s[key] ?? []).map((m) => (m.id === id ? { ...m, ...p } : m)),
    }))

  // ---------- 会话操作 ----------
  function selectSession(sid: string) {
    if (sid === activeId) return
    setActiveId(sid)
    if (!store[sid]) {
      // 本地没有缓存(如刚挂载/重启过)→ 从后端拉历史回填
      getSessionHistory(sid)
        .then((h) => {
          const msgs: ChatMessage[] = h.turns.flatMap((t, i) => [
            { id: `h${i}q`, role: 'user' as const, content: t.q },
            { id: `h${i}a`, role: 'assistant' as const, content: t.a },
          ])
          setStore((s) => ({ ...s, [sid]: msgs }))
        })
        .catch(() => setStore((s) => ({ ...s, [sid]: [] })))
    }
  }

  function newSession() {
    setActiveId(null)
    setStore((s) => ({ ...s, [NEW_KEY]: [] }))
  }

  async function removeSession(sid: string) {
    if (!window.confirm('确认删除该会话?其上下文将一并清空。')) return
    try {
      await clearSession(sid)
    } catch {
      /* 已不存在则忽略 */
    }
    setStore((s) => {
      const next = { ...s }
      delete next[sid]
      return next
    })
    setSessions((ss) => ss.filter((x) => x.session_id !== sid))
    if (sid === activeId) newSession()
  }

  async function rename(sid: string, title: string) {
    title = title.trim()
    if (!title) return
    try {
      await renameSession(sid, title)
      setSessions((ss) =>
        ss.map((x) => (x.session_id === sid ? { ...x, title } : x)),
      )
    } catch {
      /* 静默,列表下次刷新自愈 */
    }
  }

  // ---------- 提问 ----------
  async function ask(question: string) {
    const aid = nid()
    let key = activeKey
    setStore((s) => ({
      ...s,
      [key]: [
        ...(s[key] ?? []),
        { id: nid(), role: 'user', content: question },
        { id: aid, role: 'assistant', content: '', streaming: true },
      ],
    }))
    setSending(true)
    await askStream(
      question,
      {
        onMeta: (meta) => {
          if (meta.session_id && key === NEW_KEY) {
            // 新会话首发:迁移消息到真实 session_id 并激活
            key = meta.session_id
            setActiveId(meta.session_id)
            setStore((s) => {
              const migrated = s[NEW_KEY] ?? []
              return { ...s, [NEW_KEY]: [], [key]: migrated }
            })
          }
          patch(key, aid, {
            sources: meta.sources,
            refused: meta.refused,
            qaId: meta.qa_id,
            standalone: meta.standalone_question ?? undefined,
          })
        },
        onDelta: (d) => {
          const text = d.text ?? d.delta ?? d.content ?? ''
          if (text)
            setStore((s) => ({
              ...s,
              [key]: (s[key] ?? []).map((m) =>
                m.id === aid ? { ...m, content: m.content + text } : m,
              ),
            }))
        },
        onDone: (d) => {
          setStore((s) => ({
            ...s,
            [key]: (s[key] ?? []).map((m) =>
              m.id === aid
                ? {
                    ...m,
                    streaming: false,
                    content: m.content || d.answer,
                    latencyMs: d.latency_ms,
                    tokens: d.tokens,
                  }
                : m,
            ),
          }))
          void refreshSessions() // 首条问题已生成标题,刷新侧栏
        },
        onError: (msg) => patch(key, aid, { streaming: false, failed: true, content: msg }),
      },
      activeId,
    )
    setSending(false)
  }

  function submit() {
    const q = input.trim()
    if (!q || sending) return
    setInput('')
    void ask(q)
  }

  return (
    <div className="flex h-full">
      {/* 会话侧栏 */}
      <SessionSidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={selectSession}
        onNew={newSession}
        onDelete={(sid) => void removeSession(sid)}
        onRename={(sid, title) => void rename(sid, title)}
      />

      {/* 对话区 */}
      <div className="mx-auto flex h-full w-full max-w-3xl flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6">
          {messages.length === 0 && (
            <div className="mt-24 text-center text-slate-400">
              <FileText className="mx-auto mb-3 h-10 w-10" />
              <p className="text-sm">向企业知识库提问,答案附引用溯源</p>
              <p className="mt-1 text-xs">库外问题会被诚实拒答;欢迎对答案点 👍👎</p>
            </div>
          )}
          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 text-sm text-white">
                  {m.content}
                </div>
              </div>
            ) : (
              <AssistantBubble
                key={m.id}
                m={m}
                onPatch={(id, p) => patch(activeKey, id, p)}
              />
            ),
          )}
          <div ref={bottomRef} />
        </div>

        {/* 输入区 */}
        <div className="border-t bg-white px-4 py-3">
          <div className="flex items-end gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault()
                  submit()
                }
              }}
              placeholder="输入问题,Enter 发送,Shift+Enter 换行"
              className="min-h-[44px] max-h-32 resize-none"
              rows={1}
            />
            <Button onClick={submit} disabled={sending || !input.trim()} size="icon">
              {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------- 会话侧栏:新增 / 切换 / 改标题 / 删除 ----------
function SessionSidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onRename,
}: {
  sessions: SessionItem[]
  activeId: string | null
  onSelect: (sid: string) => void
  onNew: () => void
  onDelete: (sid: string) => void
  onRename: (sid: string, title: string) => void
}) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  function commitRename(sid: string) {
    onRename(sid, draft)
    setEditingId(null)
  }

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r bg-white">
      <div className="border-b p-3">
        <Button size="sm" className="w-full" onClick={onNew}>
          <MessageSquarePlus className="mr-1.5 h-4 w-4" /> 新会话
        </Button>
      </div>
      <div className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {activeId === null && (
          <div className="rounded-lg bg-blue-50 px-3 py-2 text-sm font-medium text-blue-700">
            新会话(未发送)
          </div>
        )}
        {sessions.map((s) => {
          const active = s.session_id === activeId
          const editing = editingId === s.session_id
          return (
            <div
              key={s.session_id}
              className={`group rounded-lg px-3 py-2 text-sm transition-colors ${
                active ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              {editing ? (
                <div className="flex items-center gap-1">
                  <input
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitRename(s.session_id)
                      if (e.key === 'Escape') setEditingId(null)
                    }}
                    className="h-7 w-full rounded border border-blue-300 px-1.5 text-xs outline-none"
                  />
                  <button onClick={() => commitRename(s.session_id)} title="确认">
                    <Check className="h-3.5 w-3.5 text-emerald-600" />
                  </button>
                  <button onClick={() => setEditingId(null)} title="取消">
                    <X className="h-3.5 w-3.5 text-slate-400" />
                  </button>
                </div>
              ) : (
                <div className="flex items-center justify-between gap-1">
                  <button
                    onClick={() => onSelect(s.session_id)}
                    className="min-w-0 flex-1 text-left"
                    title={s.last_question || s.title}
                  >
                    <div className={`truncate ${active ? 'font-medium' : ''}`}>
                      {s.title || '新会话'}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-400">
                      {s.turns} 轮 · {s.updated_at.slice(5, 16)}
                    </div>
                  </button>
                  <div
                    className={`flex shrink-0 items-center gap-0.5 ${
                      active ? '' : 'opacity-0 group-hover:opacity-100'
                    }`}
                  >
                    <button
                      onClick={() => {
                        setEditingId(s.session_id)
                        setDraft(s.title)
                      }}
                      title="修改标题"
                      className="rounded p-1 hover:bg-slate-200"
                    >
                      <Pencil className="h-3.5 w-3.5 text-slate-400" />
                    </button>
                    <button
                      onClick={() => onDelete(s.session_id)}
                      title="删除会话"
                      className="rounded p-1 hover:bg-red-100"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-red-400" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
        {sessions.length === 0 && activeId !== null && null}
        {sessions.length === 0 && (
          <p className="px-3 py-6 text-center text-xs text-slate-300">
            暂无历史会话,提问后自动创建
          </p>
        )}
      </div>
    </aside>
  )
}

// ---------- 答案气泡:来源 + 反馈 ----------
function AssistantBubble({
  m,
  onPatch,
}: {
  m: ChatMessage
  onPatch: (id: string, p: Partial<ChatMessage>) => void
}) {
  const [openSrc, setOpenSrc] = useState<number | null>(null)
  const [votingDown, setVotingDown] = useState(false)
  const [issue, setIssue] = useState<string>('other')
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState(false)

  async function vote(rating: 1 | -1, issueType?: string, cmt?: string) {
    if (!m.qaId || busy) return
    setBusy(true)
    try {
      await sendFeedback(m.qaId, rating, issueType, cmt)
      onPatch(m.id, { feedback: rating })
      setVotingDown(false)
    } catch {
      /* 反馈失败静默,不打断问答 */
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[85%] rounded-2xl rounded-bl-sm border px-4 py-3 text-sm ${
          m.failed
            ? 'border-red-200 bg-red-50 text-red-700'
            : m.refused
              ? 'border-amber-200 bg-amber-50'
              : 'border-slate-200 bg-white'
        }`}
      >
        {m.refused && (
          <div className="mb-1 flex items-center gap-1 text-xs text-amber-600">
            <AlertTriangle className="h-3.5 w-3.5" /> 已拒答(检索置信度不足)
          </div>
        )}
        {m.standalone && (
          <div className="mb-1 text-xs text-slate-400">理解为:{m.standalone}</div>
        )}
        <div className="whitespace-pre-wrap leading-relaxed text-slate-800">
          {m.content}
          {m.streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-blue-400" />}
        </div>

        {/* 引用溯源 */}
        {m.sources && m.sources.length > 0 && (
          <div className="mt-3 border-t border-slate-100 pt-2">
            <div className="flex flex-wrap gap-1.5">
              {m.sources.map((s) => (
                <button
                  key={s.n}
                  onClick={() => setOpenSrc(openSrc === s.n ? null : s.n)}
                  className={`rounded-full border px-2 py-0.5 text-xs transition-colors ${
                    openSrc === s.n
                      ? 'border-blue-400 bg-blue-50 text-blue-700'
                      : 'border-slate-200 text-slate-500 hover:border-blue-300'
                  }`}
                >
                  [{s.n}] {s.doc} · {s.score.toFixed(2)}
                </button>
              ))}
            </div>
            {openSrc !== null && <SourceDetail s={m.sources.find((x) => x.n === openSrc)!} />}
          </div>
        )}

        {/* 底部:统计 + 反馈 */}
        {!m.streaming && m.qaId && (
          <div className="mt-2.5 flex items-center justify-between gap-3">
            <span className="text-xs text-slate-400">
              {m.latencyMs !== undefined && `${(m.latencyMs / 1000).toFixed(1)}s`}
              {m.tokens ? ` · ${m.tokens} tokens` : ''}
            </span>
            <div className="flex items-center gap-1">
              {m.feedback ? (
                <Badge variant="secondary" className="text-xs">
                  {m.feedback === 1 ? '已赞,感谢反馈' : '已记录,进入 bad case 池'}
                </Badge>
              ) : (
                <>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => void vote(1)} disabled={busy}>
                    <ThumbsUp className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setVotingDown(!votingDown)}
                    disabled={busy}
                  >
                    <ThumbsDown className="h-3.5 w-3.5" />
                  </Button>
                </>
              )}
            </div>
          </div>
        )}

        {/* 👎 归因表单 */}
        {votingDown && !m.feedback && (
          <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5">
            <div className="mb-2 text-xs text-slate-500">哪里有问题?(可选)</div>
            <div className="mb-2 flex flex-wrap gap-1.5">
              {ISSUE_TYPES.map((t) => (
                <button
                  key={t.value}
                  onClick={() => setIssue(t.value)}
                  className={`rounded-full border px-2 py-0.5 text-xs ${
                    issue === t.value
                      ? 'border-red-400 bg-red-50 text-red-600'
                      : 'border-slate-200 text-slate-500'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="flex gap-1.5">
              <input
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="补充说明(可选)"
                className="h-8 flex-1 rounded-md border border-slate-200 px-2 text-xs outline-none focus:border-blue-400"
              />
              <Button size="sm" className="h-8" onClick={() => void vote(-1, issue, comment)} disabled={busy}>
                提交
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function SourceDetail({ s }: { s: SourceItem }) {
  return (
    <div className="mt-2 rounded-lg bg-slate-50 p-2.5 text-xs text-slate-600">
      <div className="mb-1 font-medium text-slate-700">
        [{s.n}] {s.doc} · 片段 {s.chunk_id} · 相关度 {s.score.toFixed(3)}
      </div>
      <div className="whitespace-pre-wrap leading-relaxed">{s.snippet}…</div>
    </div>
  )
}
