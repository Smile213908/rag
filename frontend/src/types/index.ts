// API 类型定义(对齐 docs/03 v1.3)

export interface SourceItem {
  n: number
  doc: string
  chunk_id: string
  score: number
  snippet: string
}

export interface AskMeta {
  sources: SourceItem[]
  refused: boolean
  refuse_reason?: string
  model?: string
  standalone_question?: string | null // 多轮改写后的独立问题(与原文不同时下发)
  qa_id: string
  session_id?: string | null
}

export interface AskDone {
  finish: boolean
  answer: string
  latency_ms: number
  tokens: number | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[]
  refused?: boolean
  qaId?: string
  standalone?: string // 多轮改写后的理解问题
  latencyMs?: number
  tokens?: number | null
  streaming?: boolean
  failed?: boolean
  feedback?: 1 | -1
}

export interface DocItem {
  doc_id: string
  filename: string
  chunks: number
  status: string // done / indexing / failed
  uploaded_at?: string | null
}

// 索引进度(§3.3,分段流程展示)
export interface DocStatus {
  doc_id: string
  status: string // indexing / done / failed
  progress: number
  stage: string // queued/uploaded/parsing/chunked/encoding/finalizing/done
  chunks_done?: number
  chunks_total?: number | null
  error?: string
}

// ---------- 会话管理(§2.4) ----------
export interface SessionItem {
  session_id: string
  title: string
  turns: number
  updated_at: string
  last_question: string
}

export interface SessionHistory {
  session_id: string
  title: string
  turns: { q: string; a: string }[]
}

export interface Health {
  chroma: string
  chroma_chunks: number
  embed_model: string
  rerank_model: string
  llm: string
  refuse_threshold: number
}

export const ISSUE_TYPES = [
  { value: 'not_found', label: '没查到' },
  { value: 'wrong_answer', label: '答错了' },
  { value: 'wrong_source', label: '引用错' },
  { value: 'bad_refuse', label: '拒答不当' },
  { value: 'other', label: '其他' },
] as const

// ---------- 运营看板 ----------
export interface Overview {
  total_queries: number
  today_queries: number
  hit_rate: number | null
  refuse_rate: number | null
  thumbs_up_rate: number | null
  p95_latency_ms: number
  tokens_total: number
  open_bad_cases: number
}

export interface HotQuestion {
  question: string
  count: number
}

export interface BadCase {
  ts: string
  qa_id: string
  rating: number
  issue_type: string | null
  comment: string | null
  status: string
  resolve_action?: string
}
