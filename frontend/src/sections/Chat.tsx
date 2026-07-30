// 问答主界面:SSE 流式 + 引用溯源 + 👍👎 反馈(docs/03 §2)
import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, FileText, Loader2, MessageSquarePlus, Send, ThumbsDown, ThumbsUp } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { askStream, clearSession, sendFeedback } from '@/lib/api'
import type { ChatMessage, SourceItem } from '@/types'
import { ISSUE_TYPES } from '@/types'

let seq = 0
const nid = () => `m${Date.now()}_${seq++}`

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const patch = (id: string, p: Partial<ChatMessage>) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...p } : m)))

  async function ask(question: string) {
    const aid = nid()
    setMessages((ms) => [
      ...ms,
      { id: nid(), role: 'user', content: question },
      { id: aid, role: 'assistant', content: '', streaming: true },
    ])
    setSending(true)
    await askStream(question, {
      onMeta: (meta) => {
        if (meta.session_id) setSessionId(meta.session_id)
        patch(aid, {
          sources: meta.sources,
          refused: meta.refused,
          qaId: meta.qa_id,
          standalone: meta.standalone_question ?? undefined,
        })
      },
      onDelta: (d) => {
        const text = d.text ?? d.delta ?? d.content ?? ''
        if (text)
          setMessages((ms) =>
            ms.map((m) => (m.id === aid ? { ...m, content: m.content + text } : m)),
          )
      },
      onDone: (d) =>
        setMessages((ms) =>
          ms.map((m) =>
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
        ),
      onError: (msg) => patch(aid, { streaming: false, failed: true, content: msg }),
    }, sessionId)
    setSending(false)
  }

  function newTopic() {
    if (sessionId) void clearSession(sessionId).catch(() => {})
    setSessionId(null)
    setMessages([])
  }

  function submit() {
    const q = input.trim()
    if (!q || sending) return
    setInput('')
    void ask(q)
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col">
      {/* 消息区 */}
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6">
        {messages.length > 0 && (
          <div className="flex justify-end">
            <button
              onClick={newTopic}
              className="flex items-center gap-1 rounded-full border border-slate-200 px-2.5 py-1 text-xs text-slate-400 hover:border-blue-300 hover:text-blue-600"
            >
              <MessageSquarePlus className="h-3.5 w-3.5" /> 新话题
            </button>
          </div>
        )}
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
            <AssistantBubble key={m.id} m={m} onPatch={patch} />
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
