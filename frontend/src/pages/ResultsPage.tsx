import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listRuns, deleteRun } from '../lib/api'
import { formatDate } from '../lib/utils'
import LoadingSpinner from '../components/common/LoadingSpinner'
import type { TestRun } from '../types'

const STATUS: Record<string, { label: string; dot: string; text: string }> = {
  completed: { label: '완료',   dot: 'bg-emerald-400', text: 'text-emerald-600' },
  running:   { label: '실행 중', dot: 'bg-indigo-400 animate-pulse', text: 'text-indigo-600' },
  pending:   { label: '대기',   dot: 'bg-gray-300',   text: 'text-gray-500' },
  failed:    { label: '실패',   dot: 'bg-red-400',    text: 'text-red-500' },
}

export default function ResultsPage() {
  const qc = useQueryClient()
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['runs'],
    queryFn: listRuns,
    refetchInterval: (query) => {
      const data = query.state.data as TestRun[] | undefined
      return data?.some((r) => r.status === 'running' || r.status === 'pending') ? 3000 : false
    },
  })
  const deleteMut = useMutation({
    mutationFn: deleteRun,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runs'] }),
  })

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">결과</h1>
          <p className="text-sm text-gray-400 mt-0.5">{runs.length}개 실행</p>
        </div>
        <Link to="/run"
          className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors">
          새 테스트
        </Link>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-16"><LoadingSpinner /></div>
      ) : runs.length === 0 ? (
        <div className="bg-white border border-gray-200 rounded-xl text-center py-16 text-gray-400 text-sm">
          아직 결과가 없습니다.{' '}
          <Link to="/run" className="text-indigo-600 hover:underline">테스트 시작하기</Link>
        </div>
      ) : (
        <div className="space-y-2">
          {runs.map((run: TestRun) => {
            const st = STATUS[run.status]
            const ragOn = run.config.rag_enabled
            return (
              <div key={run.id}
                className="bg-white border border-gray-200 rounded-xl px-5 py-4 flex items-center gap-4 hover:border-gray-300 transition-colors">
                <div className={`w-2 h-2 rounded-full shrink-0 ${st.dot}`} />

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="font-medium text-gray-800 truncate text-sm">{run.name}</span>
                    <span className={`text-xs font-medium shrink-0 ${st.text}`}>{st.label}</span>
                  </div>
                  <div className="text-xs text-gray-400 flex gap-4 flex-wrap">
                    {ragOn
                      ? <span>{run.config.embedding_model_ids.length}임베딩 × {run.config.llm_model_ids.length}LLM</span>
                      : <span>{run.config.llm_model_ids.length}LLM (RAG 없음)</span>
                    }
                    <span>{run.completed_tests}/{run.total_tests} 완료</span>
                    <span>{formatDate(run.created_at)}</span>
                    {run.error && <span className="text-red-400 truncate">{run.error}</span>}
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {run.status === 'running' && <LoadingSpinner size="sm" />}
                  {run.status === 'completed' && (
                    <Link to={`/results/${run.id}`}
                      className="px-3 py-1.5 bg-indigo-50 text-indigo-700 text-xs font-medium rounded-lg hover:bg-indigo-100 transition-colors">
                      결과 보기
                    </Link>
                  )}
                  <button
                    onClick={() => { if (confirm('삭제할까요?')) deleteMut.mutate(run.id) }}
                    className="p-1.5 text-gray-300 hover:text-red-400 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                      <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                    </svg>
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
