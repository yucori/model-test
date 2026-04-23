import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import { getRunComparison, getRunResults } from '../lib/api'
import type { TestResult } from '../types'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ScoreBar from '../components/common/ScoreBar'

// ── Helpers ───────────────────────────────────────────────────────────────────

function scoreCell(score: number) {
  const bg =
    score >= 8 ? 'bg-emerald-100 text-emerald-700' :
    score >= 6 ? 'bg-amber-100 text-amber-700' :
    score >= 4 ? 'bg-orange-100 text-orange-700' :
                 'bg-red-100 text-red-600'
  return (
    <span className={`inline-block px-2 py-0.5 rounded-lg text-xs font-bold tabular-nums ${bg}`}>
      {score.toFixed(1)}
    </span>
  )
}

const PAIR_COLORS = ['#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

const METRIC_KEYS = [
  { key: 'avg_relevance', label: '관련성' },
  { key: 'avg_accuracy', label: '정확성' },
  { key: 'avg_helpfulness', label: '도움성' },
  { key: 'avg_korean_fluency', label: '한국어' },
  { key: 'avg_source_citation', label: '출처인용' },
  { key: 'avg_overall', label: '종합' },
  { key: 'avg_retrieval_score', label: '검색 유사도' },
]

// Metrics that correspond to EvaluationScores fields (excludes retrieval_score)
const SCORE_KEYS = [
  { key: 'relevance', label: '관련성' },
  { key: 'accuracy', label: '정확성' },
  { key: 'helpfulness', label: '도움성' },
  { key: 'korean_fluency', label: '한국어' },
  { key: 'source_citation', label: '출처인용' },
  { key: 'overall', label: '종합' },
]

function shortLabel(id: string | null) {
  if (!id) return 'RAG 없음'
  const llm = id
    .replace('-20251001', '')
    .replace('gemini-', 'G-')
  const noPrefix = llm.replace(/^(local|openai|ollama|hf):/, '')
  const name = noPrefix.includes('/') ? noPrefix.split('/').pop()! : noPrefix.split(':')[0]
  return name
    .replace('all-MiniLM-L6-v2', 'MiniLM')
    .replace('nomic-embed-text-v1.5', 'nomic-v1.5')
    .replace('ko-sroberta-multitask', 'ko-sroberta')
    .replace('jina-embeddings-v4', 'jina-v4')
    .replace('qwen3-embedding-4b', 'qwen3-4b')
    .replace('embeddinggemma-300m', 'gemma-300m')
    .replace('multilingual-e5-small', 'me5-small')
    .replace('text-embedding-3-small', 'emb-3-small')
    .replace('text-embedding-3-large', 'emb-3-large')
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ResultDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [expandedQ, setExpandedQ] = useState<string | null>(null)
  const [matrixMetric, setMatrixMetric] = useState<string>('avg_overall')

  const { data: comparison, isLoading: cLoading } = useQuery({
    queryKey: ['comparison', runId],
    queryFn: () => getRunComparison(runId!),
    enabled: !!runId,
  })
  const { data: results = [], isLoading: rLoading } = useQuery({
    queryKey: ['results', runId],
    queryFn: () => getRunResults(runId!),
    enabled: !!runId,
  })

  if (cLoading || rLoading) {
    return <div className="flex justify-center items-center h-64"><LoadingSpinner size="lg" /></div>
  }
  if (!comparison) {
    return (
      <div className="p-8 text-center text-slate-400">
        결과 없음.{' '}
        <Link to="/results" className="text-indigo-600 hover:underline">목록으로</Link>
      </div>
    )
  }

  const { pair_summaries, emb_avg, emb_detail, by_category } = comparison
  const best = pair_summaries[0]

  // Unique embedding and LLM IDs (type-safe)
  const allEmbIds = [...new Set(pair_summaries.map((p) => p.embedding_model_id))]
  const definedEmbIds = allEmbIds.filter((e): e is string => e !== null)
  const llmIds = [...new Set(pair_summaries.map((p) => p.llm_model_id))]
  const ragEnabled = allEmbIds.some((e) => e !== null)

  // Matrix lookup: "emb||llm" → PairSummary
  const pairMap = Object.fromEntries(
    pair_summaries.map((p) => [`${p.embedding_model_id ?? 'none'}||${p.llm_model_id}`, p])
  )

  // Bar chart data (LLM comparison)
  const llmBarData = METRIC_KEYS.map(({ key, label }) => {
    const entry: Record<string, string | number> = { metric: label }
    llmIds.forEach((lid) => {
      const ps = pair_summaries.filter((p) => p.llm_model_id === lid)
      const avg = ps.length ? ps.reduce((s, p) => s + (p[key as keyof typeof p] as number), 0) / ps.length : 0
      entry[lid] = Number(avg.toFixed(2))
    })
    return entry
  })

  // Bar chart data (Embedding comparison)
  const embBarData = ragEnabled ? METRIC_KEYS.map(({ key, label }) => {
    const entry: Record<string, string | number> = { metric: label }
    definedEmbIds.forEach((eid) => {
      const ps = pair_summaries.filter((p) => p.embedding_model_id === eid)
      const avg = ps.length ? ps.reduce((s, p) => s + (p[key as keyof typeof p] as number), 0) / ps.length : 0
      entry[eid] = Number(avg.toFixed(2))
    })
    return entry
  }) : []

  // Group results by question
  const byQuestion: Record<string, TestResult[]> = {}
  for (const r of results) {
    byQuestion[r.question_id] = byQuestion[r.question_id] ?? []
    byQuestion[r.question_id].push(r)
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-1">
        <Link to="/results" className="text-slate-400 hover:text-slate-600 text-sm">← 목록</Link>
        <h1 className="text-2xl font-bold text-slate-900">{comparison.run_name}</h1>
      </div>
      <p className="text-slate-500 text-sm mb-8">
        {pair_summaries.length}개 조합 · {results.length}개 응답
      </p>

      {/* Best result banner */}
      <div className="bg-gradient-to-r from-indigo-50 to-violet-50 border border-indigo-200 rounded-2xl p-5 mb-8 flex items-center gap-5">
        <div className="text-4xl">🏆</div>
        <div className="flex-1">
          <div className="text-xs font-semibold text-indigo-500 uppercase tracking-wide mb-1">최고 성능 조합</div>
          <div className="text-lg font-bold text-slate-900">{best.pair_label}</div>
          <div className="flex gap-4 mt-1 text-sm text-slate-600 flex-wrap">
            <span>종합 <b className="text-indigo-700">{best.avg_overall.toFixed(2)}</b>/10</span>
            <span>정확성 {best.avg_accuracy.toFixed(2)}</span>
            <span>관련성 {best.avg_relevance.toFixed(2)}</span>
            <span className="text-slate-400 text-xs">{Math.round(best.avg_latency_ms)}ms</span>
          </div>
        </div>
      </div>

      {/* ── Leaderboard table ── */}
      <div className="bg-white border border-slate-200 rounded-2xl mb-6 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100">
          <h2 className="font-semibold text-slate-800">전체 성능 비교 리더보드</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            모든 조합의 수치 지표 · 색상 = 열 내 상대 순위 (초록 우수 / 빨강 열위)
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100">
                <th className="px-4 py-3 text-left text-slate-400 font-medium w-8">#</th>
                <th className="px-4 py-3 text-left text-slate-500 font-medium min-w-48">조합</th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">
                  종합 <span className="text-slate-300">/10</span>
                </th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">관련성</th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">정확성</th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">도움성</th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">한국어</th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">출처인용</th>
                {ragEnabled && (
                  <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">검색유사도</th>
                )}
                <th className="px-3 py-3 text-center text-slate-300 font-medium whitespace-nowrap text-xs">
                  응답속도 ↓
                </th>
                <th className="px-3 py-3 text-center text-slate-300 font-medium whitespace-nowrap text-xs">
                  토큰 ↓
                </th>
                <th className="px-3 py-3 text-center text-slate-500 font-medium whitespace-nowrap">실패</th>
              </tr>
            </thead>
            <tbody>
              {pair_summaries.map((ps, rank) => {
                // Compute relative score for a value within its column's range
                const rel = (val: number, colKey: keyof typeof ps, higherIsBetter = true) => {
                  const vals = pair_summaries.map((p) => p[colKey] as number)
                  const min = Math.min(...vals); const max = Math.max(...vals)
                  if (max === min) return 0.5
                  return higherIsBetter ? (val - min) / (max - min) : (max - val) / (max - min)
                }
                const heatCell = (val: number, colKey: keyof typeof ps, fmt: string, higherIsBetter = true) => {
                  const r = rel(val, colKey, higherIsBetter)
                  const bg =
                    r >= 0.8 ? 'bg-emerald-100 text-emerald-800' :
                    r >= 0.5 ? 'bg-teal-50 text-teal-700' :
                    r >= 0.25 ? 'bg-amber-50 text-amber-700' :
                               'bg-red-50 text-red-600'
                  const isBest = r >= 0.99
                  return (
                    <td className="px-3 py-3 text-center">
                      <span className={`inline-block px-2 py-0.5 rounded font-bold tabular-nums ${bg} ${isBest ? 'ring-1 ring-offset-0 ring-emerald-400' : ''}`}>
                        {fmt}
                      </span>
                    </td>
                  )
                }

                const medals = ['🥇', '🥈', '🥉']
                return (
                  <tr key={ps.pair_label} className={`border-b border-slate-50 hover:bg-slate-50 transition-colors ${rank === 0 ? 'bg-indigo-50/30' : ''}`}>
                    <td className="px-4 py-3 text-center font-bold text-slate-400">
                      {rank < 3 ? medals[rank] : rank + 1}
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-semibold text-slate-800 leading-tight">{ps.pair_label}</div>
                      {ps.failed_tests > 0 && (
                        <div className="text-red-400 text-xs mt-0.5">{ps.failed_tests}건 실패</div>
                      )}
                    </td>
                    {heatCell(ps.avg_overall,        'avg_overall',        ps.avg_overall.toFixed(2))}
                    {heatCell(ps.avg_relevance,       'avg_relevance',      ps.avg_relevance.toFixed(2))}
                    {heatCell(ps.avg_accuracy,        'avg_accuracy',       ps.avg_accuracy.toFixed(2))}
                    {heatCell(ps.avg_helpfulness,     'avg_helpfulness',     ps.avg_helpfulness.toFixed(2))}
                    {heatCell(ps.avg_korean_fluency,  'avg_korean_fluency',  ps.avg_korean_fluency.toFixed(2))}
                    {heatCell(ps.avg_source_citation, 'avg_source_citation', ps.avg_source_citation.toFixed(2))}
                    {ragEnabled && heatCell(ps.avg_retrieval_score, 'avg_retrieval_score', ps.avg_retrieval_score.toFixed(3))}
                    <td className="px-3 py-3 text-center text-slate-400 tabular-nums text-xs">{Math.round(ps.avg_latency_ms)}ms</td>
                    <td className="px-3 py-3 text-center text-slate-400 tabular-nums text-xs">{Math.round(ps.avg_completion_tokens)}</td>
                    <td className="px-3 py-3 text-center">
                      <span className={ps.failed_tests > 0 ? 'text-red-500 font-bold' : 'text-slate-300'}>
                        {ps.failed_tests}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Matrix (embedding × LLM) — shown only when RAG is on */}
      {ragEnabled && definedEmbIds.length > 0 && llmIds.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl mb-6">
          <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
            <div>
              <h2 className="font-semibold text-slate-800">임베딩 × LLM 매트릭스</h2>
              <p className="text-xs text-slate-400 mt-0.5">행 = 임베딩 모델, 열 = 생성 모델</p>
            </div>
            <select
              value={matrixMetric}
              onChange={(e) => setMatrixMetric(e.target.value)}
              className="text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {METRIC_KEYS.map(({ key, label }) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          </div>
          <div className="p-6 overflow-x-auto">
            <table className="text-sm">
              <thead>
                <tr>
                  <th className="text-left pr-6 pb-3 text-slate-400 text-xs font-medium whitespace-nowrap">임베딩 ↓ / LLM →</th>
                  {llmIds.map((lid) => (
                    <th key={lid} className="px-4 pb-3 text-center text-xs font-medium text-slate-600 whitespace-nowrap">
                      {shortLabel(lid)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {definedEmbIds.map((eid) => (
                  <tr key={eid}>
                    <td className="pr-6 py-2 text-xs font-medium text-slate-600 whitespace-nowrap">{shortLabel(eid)}</td>
                    {llmIds.map((lid) => {
                      const ps = pairMap[`${eid ?? 'none'}||${lid}`]
                      const score = ps ? (ps[matrixMetric as keyof typeof ps] as number) : 0
                      const maxInCol = Math.max(...definedEmbIds.map((e) => {
                        const p = pairMap[`${e ?? 'none'}||${lid}`]
                        return p ? (p[matrixMetric as keyof typeof p] as number) : 0
                      }))
                      const isMax = score === maxInCol && score > 0
                      return (
                        <td key={lid} className="px-4 py-2 text-center">
                          <span className={`inline-block px-3 py-1 rounded-lg text-sm font-bold tabular-nums
                            ${!ps ? 'text-slate-300' :
                              score >= 8 ? 'bg-emerald-100 text-emerald-700' :
                              score >= 6 ? 'bg-amber-100 text-amber-700' :
                              score >= 4 ? 'bg-orange-100 text-orange-700' :
                                           'bg-red-100 text-red-600'}
                            ${isMax ? 'ring-2 ring-indigo-400' : ''}`}>
                            {ps ? score.toFixed(1) : '—'}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-slate-400 mt-3">테두리: 해당 열에서 최고 점수</p>
          </div>
        </div>
      )}

      {/* ── 임베딩 모델별 LLM 답변 품질 (방법 1 — 교차 분석) ── */}
      {ragEnabled && emb_detail && Object.keys(emb_detail).length > 1 && (
        <div className="bg-white border border-slate-200 rounded-2xl mb-6">
          <div className="px-6 py-4 border-b border-slate-100">
            <h2 className="font-semibold text-slate-800">임베딩 모델별 LLM 답변 품질</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              각 임베딩 모델 사용 시 LLM 답변 품질 평균 ({definedEmbIds.length > 0 ? `${Object.values(emb_detail)[0]?.llm_count ?? '?'}개 LLM 평균` : ''}) ·
              검색 컨텍스트 품질 → LLM 답변 품질의 인과 관계를 확인합니다
            </p>
          </div>
          <div className="p-5">
            <div className="bg-indigo-50 border border-indigo-100 rounded-xl px-4 py-2.5 text-xs text-indigo-700 mb-4">
              <span className="font-semibold">교차 분석 해석:</span>
              {' '}코사인 유사도(검색 유사도)가 높아도 LLM 답변 품질이 낮다면 false positive(무관한 청크가 상위 반환)를 의심하세요.
              반대로 유사도는 낮지만 답변 품질이 높다면 실제로 관련 있는 청크를 찾고 있다는 의미입니다.
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-100 text-slate-400">
                    <th className="text-left px-4 py-2.5 font-medium">임베딩 모델</th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="LLM 답변의 질문 관련성 평균 (0–10). 임베딩이 관련 청크를 잘 찾을수록 높아짐." className="cursor-help border-b border-dashed border-slate-300">관련성</span>
                    </th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="LLM 답변 정확도 평균 (0–10). 사실 오류 없이 정확한 정보를 제공하는 비율." className="cursor-help border-b border-dashed border-slate-300">정확성</span>
                    </th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="LLM 답변 도움성 평균 (0–10). 사용자 문제 해결에 실제로 도움이 되는 정도." className="cursor-help border-b border-dashed border-slate-300">도움성</span>
                    </th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="한국어 유창성 평균 (0–10). 자연스러운 한국어 표현 및 문법." className="cursor-help border-b border-dashed border-slate-300">한국어</span>
                    </th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="출처 인용 정확도 평균 (0–10). 문서·조·항을 올바르게 특정했는지." className="cursor-help border-b border-dashed border-slate-300">출처인용</span>
                    </th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="전체 품질 종합 점수 평균 (0–10)." className="cursor-help border-b border-dashed border-slate-300">종합</span>
                    </th>
                    <th className="text-center px-3 py-2.5 font-medium">
                      <span title="검색 청크의 평균 코사인 유사도 (0–1). 높다고 반드시 좋은 청크가 아님 (false positive 주의)." className="cursor-help border-b border-dashed border-slate-300">검색 유사도</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    const entries = Object.entries(emb_detail).sort(([, a], [, b]) => b.avg_overall - a.avg_overall)
                    const overallVals = entries.map(([, d]) => d.avg_overall)
                    const maxOverall = Math.max(...overallVals)
                    const minOverall = Math.min(...overallVals)

                    const embDetailMetrics: Array<{ key: keyof typeof entries[0][1]; scale: number }> = [
                      { key: 'avg_relevance',      scale: 10 },
                      { key: 'avg_accuracy',       scale: 10 },
                      { key: 'avg_helpfulness',    scale: 10 },
                      { key: 'avg_korean_fluency', scale: 10 },
                      { key: 'avg_source_citation',scale: 10 },
                      { key: 'avg_overall',        scale: 10 },
                      { key: 'avg_retrieval_score',scale: 1  },
                    ]

                    return entries.map(([embId, detail], idx) => {
                      const isTop = detail.avg_overall === maxOverall && detail.avg_overall > 0
                      const isBot = detail.avg_overall === minOverall && entries.length > 1

                      const cellColor = (val: number, scale: number) => {
                        const norm = val / scale
                        if (norm >= 0.8) return 'bg-emerald-100 text-emerald-700'
                        if (norm >= 0.65) return 'bg-teal-50 text-teal-700'
                        if (norm >= 0.5) return 'bg-amber-50 text-amber-700'
                        return 'bg-red-50 text-red-600'
                      }

                      return (
                        <tr key={embId} className={`border-b border-slate-50 hover:bg-slate-50 ${idx === 0 ? 'bg-indigo-50/20' : ''}`}>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              {isTop && <span className="text-sm">🏆</span>}
                              {isBot && !isTop && <span className="text-sm opacity-50">▼</span>}
                              <span className="font-medium text-slate-700">{shortLabel(embId)}</span>
                            </div>
                            <div className="text-[10px] text-slate-400 mt-0.5">{detail.llm_count}개 LLM 평균</div>
                          </td>
                          {embDetailMetrics.map(({ key, scale }) => {
                            const val = detail[key] as number
                            return (
                              <td key={key} className="px-3 py-3 text-center">
                                <span className={`inline-block px-2 py-0.5 rounded font-bold tabular-nums text-xs ${cellColor(val, scale)}`}>
                                  {scale === 1 ? val.toFixed(3) : val.toFixed(2)}
                                </span>
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })
                  })()}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-400 mt-2">
              ※ 행 순서: 종합 점수 기준 내림차순 · 검색 유사도가 높아도 종합 점수가 낮다면 false positive를 의심하세요.
            </p>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className={`grid gap-6 mb-6 ${ragEnabled && Object.keys(emb_avg).length > 1 ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
        {/* LLM comparison */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6">
          <h3 className="font-semibold text-slate-800 mb-1">LLM 지표별 비교</h3>
          <p className="text-xs text-slate-400 mb-4">임베딩 모델 평균값</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={llmBarData} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 10]} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {llmIds.map((lid, i) => (
                <Bar key={lid} dataKey={lid} name={shortLabel(lid)}
                  fill={PAIR_COLORS[i % PAIR_COLORS.length]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Embedding comparison */}
        {ragEnabled && Object.keys(emb_avg).length > 1 && (
          <div className="bg-white border border-slate-200 rounded-2xl p-6">
            <h3 className="font-semibold text-slate-800 mb-1">임베딩 지표별 비교</h3>
            <p className="text-xs text-slate-400 mb-4">LLM 평균값</p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={embBarData} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 10]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {definedEmbIds.map((eid, i) => (
                  <Bar key={eid!} dataKey={eid!} name={shortLabel(eid)}
                    fill={PAIR_COLORS[(i + 3) % PAIR_COLORS.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Category breakdown */}
      {Object.keys(by_category).length > 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl mb-6">
          <div className="px-6 py-4 border-b border-slate-100">
            <h3 className="font-semibold text-slate-800">카테고리별 종합 점수</h3>
          </div>
          <div className="p-6 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-400 border-b border-slate-100">
                  <th className="text-left py-2 pr-6">카테고리</th>
                  {pair_summaries.map((ps) => (
                    <th key={ps.pair_label} className="text-center px-3 py-2 whitespace-nowrap">
                      {ps.pair_label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(by_category).map(([cat, scores]) => {
                  const max = Math.max(...Object.values(scores))
                  return (
                    <tr key={cat} className="border-b border-slate-50">
                      <td className="py-2 pr-6 font-medium text-slate-700">{cat}</td>
                      {pair_summaries.map((ps) => {
                        const s = scores[ps.pair_label] ?? 0
                        return (
                          <td key={ps.pair_label} className="px-3 py-2 text-center">
                            {scoreCell(s)}
                            {s === max && s > 0 && <span className="ml-1 text-xs">★</span>}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per-question drill-down */}
      <div className="bg-white border border-slate-200 rounded-2xl">
        <div className="px-6 py-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800">질문별 상세</h3>
          <p className="text-xs text-slate-400 mt-0.5">행 클릭 시 전체 응답을 확인합니다</p>
        </div>

        {Object.entries(byQuestion).map(([qid, qResults]) => {
          const isOpen = expandedQ === qid
          const question = qResults[0]?.question ?? qid

          return (
            <div key={qid} className="border-b border-slate-50 last:border-0">
              <div
                className="px-6 py-4 cursor-pointer hover:bg-slate-50 transition-colors flex items-start gap-3"
                onClick={() => setExpandedQ(isOpen ? null : qid)}
              >
                <span className="text-slate-300 mt-0.5 text-sm">{isOpen ? '▾' : '▸'}</span>
                <div className="flex-1">
                  <p className="text-sm font-medium text-slate-800 mb-2">{question}</p>
                  <div className="flex gap-3 flex-wrap">
                    {qResults.map((r) => (
                      <div key={`${r.embedding_model_id}${r.llm_model_id}`} className="flex items-center gap-1 text-xs">
                        <span className="text-slate-400">
                          {shortLabel(r.embedding_model_id)}/{shortLabel(r.llm_model_id)}:
                        </span>
                        {r.scores ? scoreCell(r.scores.overall) : <span className="text-red-400 text-xs">오류</span>}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {isOpen && (
                <div className="px-6 pb-6">
                  <div
                    className="grid gap-4"
                    style={{ gridTemplateColumns: `repeat(${Math.min(qResults.length, 2)}, 1fr)` }}
                  >
                    {qResults.map((r) => (
                      <div key={`${r.embedding_model_id}${r.llm_model_id}`}
                        className="border border-slate-100 rounded-xl p-4">
                        {/* Model info */}
                        <div className="flex flex-wrap gap-2 mb-3">
                          <div className="text-xs bg-teal-50 text-teal-700 rounded px-2 py-0.5">
                            <b>임베딩:</b> {shortLabel(r.embedding_model_id)}
                          </div>
                          <div className="text-xs bg-indigo-50 text-indigo-700 rounded px-2 py-0.5">
                            <b>LLM:</b> {shortLabel(r.llm_model_id)}
                          </div>
                          <div className="text-xs text-slate-400">
                            {r.latency_ms.toFixed(0)}ms · {r.completion_tokens}tok
                          </div>
                        </div>

                        {/* Response */}
                        {r.error ? (
                          <div className="text-red-500 text-sm bg-red-50 rounded-lg p-3">{r.error}</div>
                        ) : (
                          <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed mb-4">
                            {r.response}
                          </p>
                        )}

                        {/* Scores */}
                        {r.scores && (
                          <div className="border-t border-slate-100 pt-3 space-y-1.5">
                            {SCORE_KEYS.map(({ key, label }) => (
                              <ScoreBar
                                key={key}
                                score={r.scores![key as keyof typeof r.scores] as number}
                                label={label}
                              />
                            ))}
                            {r.scores.reasoning && (
                              <details className="mt-2 border-t border-slate-100 pt-2">
                                <summary className="text-xs font-semibold text-amber-600 cursor-pointer hover:text-amber-700">
                                  ⚖️ Judge 평가 이유
                                </summary>
                                <p className="text-xs text-amber-900 mt-1.5 leading-relaxed bg-amber-50 rounded-lg p-2">
                                  {r.scores.reasoning}
                                </p>
                              </details>
                            )}
                          </div>
                        )}

                        {/* Retrieved chunks with scores */}
                        {r.retrieved_chunks && r.retrieved_chunks.length > 0 && (
                          <details className="mt-3">
                            <summary className="text-xs text-slate-400 cursor-pointer hover:text-slate-600 select-none">
                              🔍 검색 청크 ({r.retrieved_chunks.length}개)
                              <span className="ml-1.5 text-teal-500">
                                · 평균 {(r.retrieved_chunks.reduce((a, c) => a + c.final_score, 0) / r.retrieved_chunks.length).toFixed(3)}
                              </span>
                            </summary>
                            <div className="mt-2 space-y-1.5">
                              {r.retrieved_chunks.map((chunk, i) => (
                                <div key={i} className="bg-slate-50 rounded-lg p-2.5 border border-slate-100">
                                  <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                                    <span className="text-xs font-bold text-slate-400">#{i + 1}</span>
                                    {chunk.semantic_score != null && (
                                      <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                                        chunk.semantic_score >= 0.7 ? 'bg-emerald-100 text-emerald-700' :
                                        chunk.semantic_score >= 0.4 ? 'bg-amber-100 text-amber-700' :
                                        'bg-red-50 text-red-600'}`}>
                                        sem {chunk.semantic_score.toFixed(3)}
                                      </span>
                                    )}
                                    {chunk.bm25_score != null && (
                                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-blue-50 text-blue-700">
                                        bm25 {chunk.bm25_score.toFixed(2)}
                                      </span>
                                    )}
                                    {chunk.semantic_score != null && chunk.bm25_score != null && (
                                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-purple-50 text-purple-700">
                                        rrf {chunk.final_score.toFixed(4)}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-xs text-slate-600 line-clamp-4 leading-relaxed">{chunk.content}</p>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
