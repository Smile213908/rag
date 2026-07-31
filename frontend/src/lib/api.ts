// FastAPI 后端封装(docs/03)。开发态经 vite proxy 走同源 /api。
import type { AskDone, AskMeta, BadCase, DocItem, DocStatus, Health, HotQuestion, Overview, SessionHistory, SessionItem } from '@/types'

const BASE = '/api/v1'

async function unwrap<T>(resp: Response): Promise<T> {
  const body = await resp.json()
  if (!resp.ok || body.code !== 0) {
    throw new Error(body.message || `HTTP ${resp.status}`)
  }
  return body.data as T
}

// ---------- 问答(SSE 流式) ----------
export interface AskHandlers {
  onMeta: (m: AskMeta) => void
  onDelta: (d: { text?: string; delta?: string; content?: string }) => void
  onDone: (d: AskDone) => void
  onError: (msg: string) => void
}

export async function askStream(
  question: string,
  h: AskHandlers,
  sessionId?: string | null,
): Promise<void> {
  let resp: Response
  try {
    resp = await fetch(`${BASE}/chat/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId ?? null }),
    })
  } catch {
    h.onError('无法连接后端服务,请确认 FastAPI 已启动(端口 8080)')
    return
  }
  if (!resp.ok || !resp.body) {
    h.onError(`请求失败:HTTP ${resp.status}`)
    return
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      let event = 'message'
      let data = ''
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (!data) continue
      try {
        const payload = JSON.parse(data)
        if (event === 'meta') h.onMeta(payload as AskMeta)
        else if (event === 'delta') h.onDelta(payload)
        else if (event === 'done') h.onDone(payload as AskDone)
        else if (event === 'error') h.onError(payload.message || '服务内部错误')
      } catch {
        /* 忽略半帧/坏帧 */
      }
    }
  }
}

// ---------- 反馈 ----------
export async function sendFeedback(
  qaId: string,
  rating: 1 | -1,
  issueType?: string,
  comment?: string,
): Promise<void> {
  const resp = await fetch(`${BASE}/chat/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      qa_id: qaId,
      rating,
      issue_type: issueType ?? null,
      comment: comment || null,
    }),
  })
  await unwrap(resp)
}

// ---------- 知识库 ----------
export async function listDocuments(): Promise<{ total: number; items: DocItem[] }> {
  return unwrap(await fetch(`${BASE}/documents?page=1&size=100`))
}

export async function uploadDocument(file: File): Promise<{ doc_id: string; task_id: string }> {
  const fd = new FormData()
  fd.append('file', file)
  return unwrap(await fetch(`${BASE}/documents`, { method: 'POST', body: fd }))
}

export async function deleteDocument(docId: string): Promise<{ deleted_chunks: number }> {
  return unwrap(
    await fetch(`${BASE}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' }),
  )
}

export async function rebuildDocument(docId: string): Promise<{ task_id: string }> {
  return unwrap(
    await fetch(`${BASE}/documents/${encodeURIComponent(docId)}/rebuild`, { method: 'POST' }),
  )
}

export async function getDocumentStatus(docId: string): Promise<DocStatus> {
  return unwrap(await fetch(`${BASE}/documents/${encodeURIComponent(docId)}/status`))
}

// ---------- 会话管理 ----------
export async function listSessions(): Promise<SessionItem[]> {
  const r = await unwrap<{ items: SessionItem[] }>(await fetch(`${BASE}/chat/sessions`))
  return r.items
}

export async function getSessionHistory(sessionId: string): Promise<SessionHistory> {
  return unwrap(await fetch(`${BASE}/chat/sessions/${encodeURIComponent(sessionId)}`))
}

export async function renameSession(sessionId: string, title: string): Promise<void> {
  await unwrap(
    await fetch(`${BASE}/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }),
  )
}

export async function clearSession(sessionId: string): Promise<void> {
  await unwrap(
    await fetch(`${BASE}/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
    }),
  )
}

// ---------- 运营看板 ----------
export async function fetchOverview(): Promise<Overview> {
  return unwrap(await fetch(`${BASE}/admin/stats/overview`))
}

export async function fetchHotQuestions(): Promise<HotQuestion[]> {
  const r = await unwrap<{ items: HotQuestion[] }>(
    await fetch(`${BASE}/admin/stats/hot-questions`),
  )
  return r.items
}

export async function fetchBadCases(status: string): Promise<BadCase[]> {
  const r = await unwrap<{ items: BadCase[] }>(
    await fetch(`${BASE}/admin/bad-cases?status=${status}`),
  )
  return r.items
}

export async function resolveBadCase(qaId: string, action: string): Promise<void> {
  await unwrap(
    await fetch(`${BASE}/admin/bad-cases/${encodeURIComponent(qaId)}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    }),
  )
}

// ---------- 健康 ----------
export async function fetchHealth(): Promise<Health> {
  return unwrap(await fetch(`${BASE}/health`))
}
