import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listRuns, getDocumentStats } from '../lib/api'
import { formatDate } from '../lib/utils'
import LoadingSpinner from '../components/common/LoadingSpinner'
import type { TestRun } from '../types'

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
  completed: { label: '완료', cls: 'bg-emerald-100 text-emerald-700' },
  running:   { label: '실행 중', cls: 'bg-indigo-100 text-indigo-700' },
  pending:   { label: '대기', cls: 'bg-slate-100 text-slate-500' },
  failed:    { label: '실패', cls: 'bg-red-100 text-red-600' },
}

export default function DashboardPage() {
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['runs'],
    queryFn: listRuns,
    refetchInterval: 5000,
  })
  const { data: stats } = useQuery({ queryKey: ['docStats'], queryFn: getDocumentStats })

  const completed = runs.filter((r: TestRun) => r.status === 'completed')

  const statCards = [
    { label: '업로드 문서', value: stats?.total_documents ?? '—', note: '지식 베이스' },
    { label: '벡터 청크', value: stats ? Object.values(stats.embedding_chunk_counts as Record<string, number>).reduce((a, b) => a + b, 0) : '—', note: '임베딩 완료' },
    { label: '전체 실행', value: runs.length, note: `${completed.length}개 완료` },
    { label: '마지막 실행', value: runs[0] ? STATUS_MAP[runs[0].status]?.label : '없음', note: runs[0] ? formatDate(runs[0].created_at) : '' },
  ]

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold text-slate-900 mb-0.5">대시보드</h1>
      <p className="text-slate-400 text-sm mb-8">농산물 쇼핑몰 CS 챗봇 모델 성능 비교 시스템</p>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {statCards.map((c) => (
          <div key={c.label} className="bg-white border border-slate-200 rounded-2xl p-5">
            <div className="text-xs text-slate-400 mb-2">{c.label}</div>
            <div className="text-2xl font-bold text-slate-800 tabular-nums">{c.value}</div>
            <div className="text-xs text-slate-400 mt-1">{c.note}</div>
          </div>
        ))}
      </div>

      {/* Quick actions */}
      <div className="flex gap-3 mb-8 flex-wrap">
        <Link to="/run" className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors">
          새 테스트 실행
        </Link>
        <Link to="/documents" className="px-4 py-2 bg-white text-slate-600 border border-slate-200 text-sm rounded-xl hover:bg-slate-50 transition-colors">
          문서 업로드
        </Link>
        <Link to="/test-suite" className="px-4 py-2 bg-white text-slate-600 border border-slate-200 text-sm rounded-xl hover:bg-slate-50 transition-colors">
          질문 관리
        </Link>
      </div>

      {/* How it works */}
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 mb-8">
        <h2 className="font-semibold text-slate-700 text-sm mb-4">테스트 구조</h2>
        <div className="flex items-start gap-4 text-sm flex-wrap">
          {[
            { icon: '📄', title: '문서 업로드', desc: 'PDF/DOCX를 업로드하고 임베딩 모델로 벡터화합니다' },
            { icon: '🔍', title: '임베딩 모델', desc: '질문과 관련된 문서 청크를 검색합니다 (RAG)' },
            { icon: '🤖', title: 'LLM 모델', desc: '검색된 컨텍스트로 CS 답변을 생성합니다' },
            { icon: '⚖️', title: 'LLM Judge', desc: '관련성·정확성·도움성·한국어를 0-10점으로 평가합니다' },
          ].map((step) => (
            <div key={step.title} className="flex-1 min-w-40">
              <div className="text-xl mb-1">{step.icon}</div>
              <div className="font-medium text-slate-700">{step.title}</div>
              <div className="text-slate-400 text-xs mt-0.5">{step.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent runs */}
      <div className="bg-white border border-slate-200 rounded-2xl">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-semibold text-slate-800">최근 테스트</h2>
          <Link to="/results" className="text-indigo-600 text-xs hover:underline">전체 보기</Link>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12"><LoadingSpinner /></div>
        ) : runs.length === 0 ? (
          <div className="text-center text-slate-400 py-12 text-sm">
            아직 실행된 테스트가 없습니다.{' '}
            <Link to="/run" className="text-indigo-600 hover:underline">시작하기</Link>
          </div>
        ) : (
          <div className="divide-y divide-slate-50">
            {runs.slice(0, 6).map((run: TestRun) => {
              const st = STATUS_MAP[run.status]
              const ragOn = run.config.rag_enabled
              return (
                <div key={run.id} className="px-6 py-4 flex items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-slate-800 truncate">{run.name}</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 ${st?.cls}`}>{st?.label}</span>
                    </div>
                    <div className="text-xs text-slate-400 flex gap-3 flex-wrap">
                      {ragOn && <span>임베딩: {run.config.embedding_model_ids.length}개</span>}
                      <span>LLM: {run.config.llm_model_ids.length}개</span>
                      <span>진행: {run.completed_tests}/{run.total_tests}</span>
                      <span>{formatDate(run.created_at)}</span>
                    </div>
                  </div>
                  {run.status === 'completed' && (
                    <Link to={`/results/${run.id}`}
                      className="shrink-0 text-xs text-indigo-600 hover:underline font-medium">
                      결과 보기
                    </Link>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
