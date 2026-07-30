// 应用外壳:标题栏(健康状态)+ 问答 / 知识库 / 看板 三个页签
import { useEffect, useState } from 'react'
import { BarChart3, BookOpen, Database, MessageSquare } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import Chat from '@/sections/Chat'
import Dashboard from '@/sections/Dashboard'
import KnowledgeBase from '@/sections/KnowledgeBase'
import { fetchHealth } from '@/lib/api'
import type { Health } from '@/types'

type Tab = 'chat' | 'kb' | 'dash'

export default function Home() {
  const [tab, setTab] = useState<Tab>('chat')
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  return (
    <div className="flex h-screen flex-col bg-slate-50">
      <header className="flex items-center justify-between border-b bg-white px-5 py-3">
        <div className="flex items-center gap-2.5">
          <BookOpen className="h-5 w-5 text-blue-600" />
          <span className="font-semibold text-slate-800">企业知识库问答</span>
          {health ? (
            <Badge variant="secondary" className="ml-1 text-xs">
              {health.chroma_chunks} 块 · 拒答阈值 {health.refuse_threshold}
            </Badge>
          ) : (
            <Badge variant="outline" className="ml-1 border-red-200 text-xs text-red-500">
              后端未连接
            </Badge>
          )}
        </div>
        <nav className="flex gap-1 rounded-lg bg-slate-100 p-1">
          <NavBtn active={tab === 'chat'} onClick={() => setTab('chat')} icon={<MessageSquare className="h-4 w-4" />} label="问答" />
          <NavBtn active={tab === 'kb'} onClick={() => setTab('kb')} icon={<Database className="h-4 w-4" />} label="知识库" />
          <NavBtn active={tab === 'dash'} onClick={() => setTab('dash')} icon={<BarChart3 className="h-4 w-4" />} label="看板" />
        </nav>
      </header>
      <main className="min-h-0 flex-1">
        {tab === 'chat' ? <Chat /> : tab === 'kb' ? <KnowledgeBase /> : <Dashboard />}
      </main>
    </div>
  )
}

function NavBtn({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
        active ? 'bg-white font-medium text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
