import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  listEmbeddingModels, listLLMModels, listQuestions,
  createRun, listDocuments, debugRetrieve, debugBatchRetrieve,
} from '../lib/api'
import type { EmbedAggregate } from '../lib/api'
import type { DocumentInfo, EmbeddingModelInfo, LLMModelInfo, TestQuestion, TestResult } from '../types'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { LLMBadge, EmbeddingBadge } from '../components/common/ModelBadge'
import ScoreBar from '../components/common/ScoreBar'

// ── Helpers ───────────────────────────────────────────────────────────────────


function shortId(id: string | null) {
  if (!id) return 'no-rag'
  // LLM 단축
  const llm = id
    .replace('-20251001', '')
    .replace('gemini-', 'G-')
  // provider 프리픽스 제거 (local:, openai:, ollama:, hf:)
  const noPrefix = llm.replace(/^(local|openai|ollama|hf):/, '')
  // HuggingFace org/model 형식이면 모델명만 추출
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

function scoreColor(score: number) {
  if (score >= 0.7) return 'bg-emerald-100 text-emerald-700 border-emerald-200'
  if (score >= 0.45) return 'bg-amber-100 text-amber-700 border-amber-200'
  return 'bg-red-100 text-red-600 border-red-200'
}

// ── Tab type ──────────────────────────────────────────────────────────────────

type Tab = 'embedding' | 'llm'

const TABS: { id: Tab; label: string; desc: string }[] = [
  { id: 'embedding', label: '① 검색 테스트',  desc: 'LLM 없이 임베딩 검색 품질만 비교' },
  { id: 'llm',       label: '② LLM 평가',     desc: '전체 파이프라인 자동 평가' },
]

type EmbMode = 'manual' | 'auto'

// ── EmbedMetricsPanel ─────────────────────────────────────────────────────────
// Rich multi-metric display for embedding model comparison results.

function RankSparkline({ rankScores }: { rankScores: number[] }) {
  if (!rankScores.length) return null
  const max = Math.max(...rankScores, 0.01)
  return (
    <div className="flex items-end gap-0.5 h-8">
      {rankScores.map((s, i) => (
        <div key={i} className="flex flex-col items-center gap-0.5 flex-1">
          <div
            className={`w-full rounded-sm transition-all ${
              s >= 0.7 ? 'bg-emerald-400' : s >= 0.5 ? 'bg-amber-400' : 'bg-red-300'
            }`}
            style={{ height: `${Math.max(4, (s / max) * 28)}px` }}
            title={`Rank ${i + 1}: ${s.toFixed(3)}`}
          />
          <span className="text-[9px] text-gray-400 leading-none">#{i + 1}</span>
        </div>
      ))}
    </div>
  )
}

function DistBar({ high, mid, low }: { high: number; mid: number; low: number }) {
  const total = high + mid + low || 1
  const hp = (high / total) * 100
  const mp = (mid  / total) * 100
  const lp = (low  / total) * 100
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex h-2 rounded-full overflow-hidden flex-1 min-w-[60px]">
        {hp > 0 && <div className="bg-emerald-400" style={{ width: `${hp}%` }} title={`고품질 ${high}개`} />}
        {mp > 0 && <div className="bg-amber-400"  style={{ width: `${mp}%` }} title={`보통 ${mid}개`} />}
        {lp > 0 && <div className="bg-red-300"    style={{ width: `${lp}%` }} title={`저품질 ${low}개`} />}
      </div>
      <span className="text-[10px] text-gray-400 whitespace-nowrap tabular-nums shrink-0">
        {high}/{mid}/{low}
      </span>
    </div>
  )
}

// Metric header with tooltip explanation
function MetricTh({ label, hint }: { label: string; hint: string }) {
  return (
    <th className="text-center px-2 py-2.5 font-medium whitespace-nowrap">
      <span title={hint} className="cursor-help border-b border-dashed border-gray-300">
        {label}
      </span>
    </th>
  )
}

// ── Analysis helpers ──────────────────────────────────────────────────────────

type Insight = { level: 'good' | 'warn' | 'bad'; text: string }

function buildInsights(agg: EmbedAggregate): Insight[] {
  const ins: Insight[] = []
  const thr = agg.relevance_threshold

  // MRR
  if (agg.mrr >= 0.8)
    ins.push({ level: 'good', text: `MRR ${agg.mrr.toFixed(3)} — 대부분의 쿼리에서 관련 청크가 1~2위 안에 위치합니다. 검색 순위 품질이 우수합니다.` })
  else if (agg.mrr >= 0.5)
    ins.push({ level: 'warn', text: `MRR ${agg.mrr.toFixed(3)} — 관련 청크를 찾지만 순위가 종종 2위 이하로 밀립니다. top-k를 높이거나 청크 크기를 줄여보세요.` })
  else
    ins.push({ level: 'bad',  text: `MRR ${agg.mrr.toFixed(3)} — 관련 청크 순위가 전반적으로 낮습니다. 임베딩 모델 교체 또는 청킹 전략 변경을 고려하세요.` })

  // Hit@1 vs Hit@3 gap
  const gap = agg.hit_at_3 - agg.hit_at_1
  if (gap >= 0.25)
    ins.push({ level: 'warn', text: `Hit@3(${(agg.hit_at_3 * 100).toFixed(0)}%) − Hit@1(${(agg.hit_at_1 * 100).toFixed(0)}%) = ${(gap * 100).toFixed(0)}%p 차이. 관련 청크는 벡터 DB에 있지만 1위로 올라오지 못합니다. 청크를 더 잘게 쪼개거나 중복 내용을 줄여보세요.` })
  else if (gap < 0.05 && agg.hit_at_1 >= 0.7)
    ins.push({ level: 'good', text: `Hit@1과 Hit@3 차이가 ${(gap * 100).toFixed(0)}%p로 매우 작습니다. 관련 청크가 거의 항상 1위에 위치합니다.` })

  // Consistency (std)
  if (agg.score_std > 0.18)
    ins.push({ level: 'bad',  text: `점수 편차 ±${agg.score_std.toFixed(3)} — 질문 유형에 따라 검색 품질 편차가 매우 큽니다. 특정 카테고리만 잘 찾고 나머지는 실패할 가능성이 높습니다. 카테고리별 분석을 확인하세요.` })
  else if (agg.score_std > 0.10)
    ins.push({ level: 'warn', text: `점수 편차 ±${agg.score_std.toFixed(3)} — 일부 질문 유형에서 검색 품질이 불안정합니다.` })
  else
    ins.push({ level: 'good', text: `점수 편차 ±${agg.score_std.toFixed(3)} — 모든 질문에 걸쳐 일관된 검색 성능을 보입니다.` })

  // Rank drop-off
  if (agg.rank_scores.length >= 3) {
    const drop = agg.rank_scores[0] - agg.rank_scores[agg.rank_scores.length - 1]
    if (drop > 0.35)
      ins.push({ level: 'warn', text: `순위별 점수 하락폭 ${drop.toFixed(3)} — 1위 청크는 좋지만 이후 청크 품질이 급격히 낮아집니다. RAG에서 1개 청크만 유용하고 나머지는 노이즈가 될 수 있습니다.` })
    else if (drop < 0.15)
      ins.push({ level: 'good', text: `순위별 점수 하락폭 ${drop.toFixed(3)} — 상위 ${agg.rank_scores.length}개 청크 모두 고른 품질을 유지합니다.` })
  }

  // Miss rate
  const missCount = agg.total_queries - agg.hit_queries
  if (missCount > 0)
    ins.push({ level: missCount > agg.total_queries * 0.3 ? 'bad' : 'warn',
      text: `${missCount}개 질문(${((missCount / agg.total_queries) * 100).toFixed(0)}%)에서 결과가 0개였습니다. 해당 주제 문서가 없거나 청킹 과정에서 관련 내용이 분리됐을 수 있습니다.` })

  // Hit@1 absolute level
  if (agg.hit_at_1 < thr)
    ins.push({ level: 'bad',  text: `Hit@1 ${(agg.hit_at_1 * 100).toFixed(0)}% — 절반 이상의 질문에서 바로 답변 가능한 청크를 1위로 찾지 못합니다. LLM에 불충분한 컨텍스트가 전달될 위험이 있습니다.` })

  return ins
}

type CategoryStat = { count: number; top1Sum: number; hitAt1: number; hitAt3: number; mrr: number }

function computeCategoryStats(
  questions: TestQuestion[],
  results: Record<string, Record<string, { text: string; score: number }[]>>,
  embId: string,
  threshold: number,
): Record<string, CategoryStat> {
  const stats: Record<string, CategoryStat> = {}
  for (const q of questions) {
    const cat = q.category || '기타'
    if (!stats[cat]) stats[cat] = { count: 0, top1Sum: 0, hitAt1: 0, hitAt3: 0, mrr: 0 }
    const chunks = results[q.question]?.[embId] ?? []
    const top1 = chunks[0]?.score ?? 0
    stats[cat].count++
    stats[cat].top1Sum += top1
    if (top1 >= threshold) stats[cat].hitAt1++
    if (chunks.slice(0, 3).some((c) => c.score >= threshold)) stats[cat].hitAt3++
    for (let r = 0; r < chunks.length; r++) {
      if (chunks[r].score >= threshold) { stats[cat].mrr += 1 / (r + 1); break }
    }
  }
  return stats
}

// ── Model analysis drawer ─────────────────────────────────────────────────────

function ModelAnalysisDrawer({
  embId, agg, questions, results,
}: {
  embId: string
  agg: EmbedAggregate
  questions: TestQuestion[]
  results: Record<string, Record<string, { text: string; score: number }[]>>
}) {
  const threshold = agg.relevance_threshold
  const insights = buildInsights(agg)
  const catStats = computeCategoryStats(questions, results, embId, threshold)

  // ── 허브벡터 감지 ──────────────────────────────────────────────────────────
  // 같은 청크가 여러 쿼리의 top-1로 나타나는지 확인
  const top1ChunkFreq = new Map<string, { count: number; queries: string[] }>()
  for (const q of questions) {
    const top1 = results[q.question]?.[embId]?.[0]
    if (top1) {
      const key = top1.text.slice(0, 120)
      const entry = top1ChunkFreq.get(key) ?? { count: 0, queries: [] }
      entry.count++
      entry.queries.push(q.question)
      top1ChunkFreq.set(key, entry)
    }
  }
  const hubEntries = [...top1ChunkFreq.entries()]
    .filter(([, v]) => v.count >= Math.max(2, Math.ceil(questions.length * 0.2)))
    .sort((a, b) => b[1].count - a[1].count)
  if (hubEntries.length > 0) {
    const [hubChunk, hubInfo] = hubEntries[0]
    insights.unshift({
      level: 'bad',
      text: `반복 오검색 감지 — "${hubChunk.slice(0, 60)}…" 청크가 ${hubInfo.count}개 쿼리(전체의 ${((hubInfo.count / questions.length) * 100).toFixed(0)}%)의 Top-1으로 반환됩니다. 이 모델이 한국어 텍스트를 제대로 구분하지 못해, 서로 다른 쿼리가 같은 청크에 수렴하고 있습니다. bge-m3 또는 qwen3-embedding으로 교체를 강력 권장합니다.`,
    })
  }

  // Sort categories by avg top1 asc (weakest first)
  const catEntries = Object.entries(catStats).sort(
    ([, a], [, b]) => (a.top1Sum / a.count) - (b.top1Sum / b.count)
  )

  // Failure cases: top-1 < threshold, sorted by score asc
  const failCases = questions
    .map((q) => ({ q, chunks: results[q.question]?.[embId] ?? [] }))
    .filter(({ chunks }) => (chunks[0]?.score ?? 0) < threshold)
    .sort((a, b) => (a.chunks[0]?.score ?? 0) - (b.chunks[0]?.score ?? 0))
    .slice(0, 5)

  // Best cases: top-1 >= 0.7, sorted by score desc
  const bestCases = questions
    .map((q) => ({ q, chunks: results[q.question]?.[embId] ?? [] }))
    .filter(({ chunks }) => (chunks[0]?.score ?? 0) >= 0.7)
    .sort((a, b) => (b.chunks[0]?.score ?? 0) - (a.chunks[0]?.score ?? 0))
    .slice(0, 3)

  const levelIcon = { good: '✅', warn: '⚠️', bad: '❌' }
  const levelColor = {
    good: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    warn: 'text-amber-700 bg-amber-50 border-amber-200',
    bad:  'text-red-700 bg-red-50 border-red-200',
  }

  return (
    <div className="border-t border-gray-100 bg-slate-50 px-5 py-5 space-y-5">

      {/* ── 자동 진단 ──────────────────────────────────────────────────── */}
      <div>
        <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">자동 진단</h5>
        <div className="space-y-1.5">
          {insights.map((ins, i) => (
            <div key={i} className={`flex gap-2 items-start text-xs px-3 py-2 rounded-lg border ${levelColor[ins.level]}`}>
              <span className="shrink-0">{levelIcon[ins.level]}</span>
              <span className="leading-relaxed">{ins.text}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── 카테고리별 성능 ────────────────────────────────────────────── */}
      {catEntries.length > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            카테고리별 성능 <span className="normal-case text-gray-400 font-normal">(MRR 오름차순 — 위로 갈수록 취약)</span>
          </h5>
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-100 text-gray-400">
                  <th className="text-left py-2 px-4 font-medium">카테고리</th>
                  <th className="text-center px-3 py-2 font-medium">질문 수</th>
                  <th className="text-center px-3 py-2 font-medium">평균 Top-1</th>
                  <th className="text-center px-3 py-2 font-medium">MRR</th>
                  <th className="text-center px-3 py-2 font-medium">Hit@1</th>
                  <th className="text-center px-3 py-2 font-medium">Hit@3</th>
                  <th className="px-4 py-2 font-medium">분포</th>
                </tr>
              </thead>
              <tbody>
                {catEntries.map(([cat, st]) => {
                  const avgTop1 = st.top1Sum / st.count
                  const mrr    = st.mrr    / st.count
                  const h1     = st.hitAt1 / st.count
                  const h3     = st.hitAt3 / st.count
                  // per-category distribution
                  const catQs = questions.filter((q) => (q.category || '기타') === cat)
                  const high = catQs.filter((q) => (results[q.question]?.[embId]?.[0]?.score ?? 0) >= 0.7).length
                  const mid  = catQs.filter((q) => { const s = results[q.question]?.[embId]?.[0]?.score ?? 0; return s >= 0.5 && s < 0.7 }).length
                  const low  = catQs.length - high - mid
                  return (
                    <tr key={cat} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-2.5 px-4 font-medium text-gray-700">{cat}</td>
                      <td className="px-3 py-2.5 text-center text-gray-500">{st.count}</td>
                      <td className="px-3 py-2.5 text-center">
                        <span className={`font-bold px-1.5 py-0.5 rounded border tabular-nums ${scoreColor(avgTop1)}`}>
                          {avgTop1.toFixed(3)}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        <span className={`font-bold px-1.5 py-0.5 rounded border tabular-nums ${scoreColor(mrr)}`}>
                          {mrr.toFixed(3)}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        <span className={`font-bold tabular-nums ${scoreColor(h1)}`}>
                          {(h1 * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        <span className={`font-bold tabular-nums ${scoreColor(h3)}`}>
                          {(h3 * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-4 py-2.5 min-w-[80px]">
                        <DistBar high={high} mid={mid} low={low} />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-5">
        {/* ── 저유사도 케이스 ──────────────────────────────────────────── */}
        {failCases.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              저유사도 케이스 <span className="normal-case text-gray-400 font-normal">— Top-1 &lt; {threshold.toFixed(1)} · {failCases.length}건</span>
            </h5>
            <div className="space-y-2">
              {failCases.map(({ q, chunks }) => (
                <div key={q.id} className="bg-white rounded-xl border border-red-100 p-3">
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-[10px] bg-red-100 text-red-600 rounded px-1.5 py-0.5 shrink-0 font-medium">{q.category}</span>
                    <p className="text-xs font-medium text-gray-800 leading-snug">{q.question}</p>
                  </div>
                  {chunks.length > 0 ? (
                    <div className="space-y-1.5">
                      {chunks.map((chunk, ri) => (
                        <div key={ri} className="bg-red-50 rounded-lg p-2.5">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[10px] text-red-500 font-semibold shrink-0">{ri + 1}위 · {chunk.score.toFixed(3)}</span>
                            <div className="flex-1 h-1 bg-red-100 rounded-full overflow-hidden">
                              <div className="h-full bg-red-400 rounded-full" style={{ width: `${chunk.score * 100}%` }} />
                            </div>
                          </div>
                          <p className="text-[11px] text-gray-600 leading-relaxed italic whitespace-pre-wrap">"{chunk.text}"</p>
                          {ri === 0 && (
                            <p className="text-[10px] text-red-400 mt-1">
                              {chunk.score < 0.2
                                ? '주제가 전혀 다른 청크 반환 — 문서에 관련 내용이 없거나 청킹에서 누락된 가능성'
                                : chunk.score < 0.35
                                ? '낮은 유사도 — 관련 내용이 있지만 표현 방식이 달라 매칭 실패'
                                : '임계값 미달 — 관련 청크이지만 충분히 높은 유사도를 확보하지 못함'}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="bg-gray-50 rounded-lg p-2 text-xs text-gray-400">결과 없음 — 벡터 DB가 비어있거나 유사도 임계값 초과</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── 고유사도 케이스 ──────────────────────────────────────────── */}
        {bestCases.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              고유사도 케이스 <span className="normal-case text-gray-400 font-normal">— Top-1 ≥ 0.7 · 상위 {bestCases.length}건</span>
            </h5>
            <p className="text-[10px] text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5 mb-2 leading-relaxed">
              ⚠️ 유사도 점수가 높아도 실제 관련 내용이 아닐 수 있습니다 (false positive). 청크 내용을 직접 확인하세요.
            </p>
            <div className="space-y-2">
              {bestCases.map(({ q, chunks }) => (
                <div key={q.id} className="bg-white rounded-xl border border-amber-100 p-3">
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-[10px] bg-amber-100 text-amber-700 rounded px-1.5 py-0.5 shrink-0 font-medium">{q.category}</span>
                    <p className="text-xs font-medium text-gray-800 leading-snug">{q.question}</p>
                  </div>
                  <div className="space-y-1.5">
                    {chunks.map((chunk, ri) => (
                      <div key={ri} className="bg-amber-50 rounded-lg p-2.5">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[10px] text-amber-600 font-semibold shrink-0">{ri + 1}위 · {chunk.score.toFixed(3)}</span>
                          <div className="flex-1 h-1 bg-amber-100 rounded-full overflow-hidden">
                            <div className="h-full bg-amber-400 rounded-full" style={{ width: `${chunk.score * 100}%` }} />
                          </div>
                        </div>
                        <p className="text-[11px] text-gray-600 leading-relaxed whitespace-pre-wrap">"{chunk.text}"</p>
                        {ri === 0 && (
                          <p className="text-[10px] text-amber-500 mt-1">질문과 실제로 관련있는지 내용 직접 확인 필요 (high score ≠ relevant)</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Hub-vector stats ──────────────────────────────────────────────────────────

function computeHubStats(
  questions: TestQuestion[],
  results: Record<string, Record<string, { text: string; score: number }[]>>,
  embId: string,
  threshold: number,
): { isHub: boolean; hubRatio: number; adjustedHit1: number } {
  const freq = new Map<string, number>()
  for (const q of questions) {
    const top1 = results[q.question]?.[embId]?.[0]
    if (top1) {
      const key = top1.text.slice(0, 120)
      freq.set(key, (freq.get(key) ?? 0) + 1)
    }
  }
  const hubMin = Math.max(2, Math.ceil(questions.length * 0.2))
  const hubChunks = new Set(
    [...freq.entries()].filter(([, c]) => c >= hubMin).map(([k]) => k)
  )
  if (hubChunks.size === 0) return { isHub: false, hubRatio: 0, adjustedHit1: 0 }

  let hubHitCount = 0
  let adjustedHit = 0
  for (const q of questions) {
    const top1 = results[q.question]?.[embId]?.[0]
    if (!top1) continue
    const key = top1.text.slice(0, 120)
    if (hubChunks.has(key)) {
      hubHitCount++
    } else if (top1.score >= threshold) {
      adjustedHit++
    }
  }
  return {
    isHub: true,
    hubRatio: hubHitCount / questions.length,
    adjustedHit1: adjustedHit / questions.length,
  }
}

// ── EmbedMetricsPanel ─────────────────────────────────────────────────────────

function EmbedMetricsPanel({
  modelIds, aggregates, results, questions, topK,
  expandedQuery, onExpandQuery,
}: {
  modelIds: string[]
  aggregates: Record<string, EmbedAggregate>
  results: Record<string, Record<string, { text: string; score: number }[]>>
  questions: TestQuestion[]
  topK: number
  expandedQuery: string | null
  onExpandQuery: (q: string | null) => void
}) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null)

  // threshold는 정렬 전에 필요하므로 임시로 첫 번째 모델 기준으로 추출
  const threshold = aggregates[modelIds[0]]?.relevance_threshold ?? 0.5

  // 허브 통계를 먼저 일괄 계산 (정렬 + 렌더에서 재사용)
  const hubStatsMap = Object.fromEntries(
    modelIds.map((id) => [id, computeHubStats(questions, results, id, threshold)])
  )

  // 반복오검색 모델은 하위로 밀어냄.
  // 정상 모델: dynamic_mrr 기준 (고정 임계값 편향 제거)
  // 반복오검색 모델끼리: adjustedHit1 기준 (반복 청크 제외한 실제 성능)
  const sorted = [...modelIds].sort((a, b) => {
    const ha = hubStatsMap[a]
    const hb = hubStatsMap[b]
    if (ha.isHub && !hb.isHub) return 1
    if (!ha.isHub && hb.isHub) return -1
    if (ha.isHub && hb.isHub) return hb.adjustedHit1 - ha.adjustedHit1
    return (aggregates[b]?.dynamic_mrr ?? 0) - (aggregates[a]?.dynamic_mrr ?? 0)
  })

  return (
    <div className="space-y-5">
      {/* ── 지표 안내 ─────────────────────────────────────────────────── */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-xl px-4 py-3 text-xs text-indigo-700 space-y-1">
        <p>
          <span className="font-semibold">관련성 임계값 {threshold.toFixed(1)} 기준 —</span>
          {' '}Hit@1·Hit@3·MRR은 유사도 ≥ {threshold.toFixed(1)} 인 청크를 "관련 있음"으로 판단합니다.
          Score 분포 막대는 <span className="text-emerald-600 font-medium">■ ≥0.7</span> /{' '}
          <span className="text-amber-500 font-medium">■ 0.5–0.7</span> /{' '}
          <span className="text-red-400 font-medium">■ &lt;0.5</span>.
        </p>
        <p className="text-indigo-600">
          <span className="font-semibold text-red-600">반복 오검색 감지 시 Hit@1(조정)</span>이 표시됩니다.
          동일 청크가 여러 쿼리의 Top-1으로 반복 등장하면 threshold를 통과해 Hit@1이 부풀려집니다.
          조정값은 해당 반복 청크 매칭을 miss로 처리한 실제 성능 추정치입니다.
        </p>
        <p className="text-indigo-600">
          <span className="font-semibold">동적 임계값(D)</span> 열은 각 모델의 Top-1 점수 중앙값을 기준으로 재산정한 Hit@1·MRR입니다.
          모델마다 점수 분포 범위가 달라 고정 임계값 {threshold.toFixed(1)}로는 공정 비교가 어렵습니다.
          동적 기준은 "이 모델 내에서 상위 50%에 드는가"를 측정해 모델 간 형평성을 보정합니다.
        </p>
      </div>

      {/* ── 종합 지표 테이블 ───────────────────────────────────────────── */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
          <h4 className="text-sm font-semibold text-gray-700">임베딩 모델 성능 비교</h4>
          <span className="text-xs text-gray-400">{questions.length}개 질문 기준 · top-{topK}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100 text-gray-400 bg-gray-50">
                <th className="text-left py-2.5 px-4 font-medium w-6">순위</th>
                <th className="text-left py-2.5 px-3 font-medium">모델</th>
                <MetricTh label="Top-1 유사도" hint="각 질문의 1위 청크 평균 코사인 유사도. 높을수록 가장 관련성 높은 청크를 잘 찾는다." />
                <MetricTh label="MRR" hint="Mean Reciprocal Rank — 첫 번째 관련 청크(≥고정임계값)의 순위 역수 평균. 1.0이면 항상 1위에 있음." />
                <MetricTh label="MRR(D)" hint="동적 임계값(모델별 Top-1 중앙값) 기준 MRR. 모델 간 점수 분포 차이를 보정해 공정 비교에 사용합니다." />
                <MetricTh label="Hit@1" hint="1위 청크가 관련 있음(≥고정임계값)인 질문 비율. 즉시 답변 가능한 질문 비율." />
                <MetricTh label="Hit@1(D)" hint="동적 임계값(모델별 Top-1 중앙값) 기준 Hit@1. 이 모델 내 상위 50% 점수를 기준으로 하므로 모델 간 비교가 공정합니다." />
                <MetricTh label="Hit@3" hint="상위 3개 중 관련 청크가 1개 이상인 질문 비율. 재현율 측면." />
                <MetricTh label="일관성(Std)" hint="Top-1 유사도의 표준편차. 낮을수록 질문에 관계없이 안정적으로 검색함." />
                <MetricTh label="순위별 하락" hint="1위→K위 순서로 평균 유사도 변화. 빠르게 떨어지면 1위 외에는 낮은 품질." />
                <MetricTh label="분포 (↑/≈/↓)" hint="Top-1 점수 분포: 고품질(≥0.7) / 보통(0.5–0.7) / 저품질(<0.5) 질문 수." />
                <th className="py-2.5 px-3 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((eid, rank) => {
                const agg = aggregates[eid]
                if (!agg) return null
                const isOpen = expandedModel === eid
                const hub = hubStatsMap[eid]
                return (
                  <>
                    <tr
                      key={eid}
                      className={`border-b border-gray-50 hover:bg-gray-50 cursor-pointer ${isOpen ? 'bg-indigo-50/40' : ''}`}
                      onClick={() => setExpandedModel(isOpen ? null : eid)}
                    >
                      <td className="py-3 px-4">
                        <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold
                          ${rank === 0 ? 'bg-amber-100 text-amber-700' : 'bg-gray-100 text-gray-500'}`}>
                          {rank + 1}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <EmbeddingBadge provider={eid.split(':')[0] as 'local' | 'openai' | 'ollama'} />
                          <span className="font-medium text-gray-700">{shortId(eid)}</span>
                          {hub.isHub && (
                            <span className="text-[9px] font-bold bg-red-100 text-red-600 border border-red-200 rounded px-1 py-0.5 leading-none">
                              반복오검색 {(hub.hubRatio * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      </td>
                      {/* Top-1 유사도 */}
                      <td className="px-2 py-3 text-center">
                        <div className="flex items-center gap-1.5 justify-center">
                          <div className="w-14 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full ${agg.avg_top1_score >= 0.7 ? 'bg-emerald-400' : agg.avg_top1_score >= 0.5 ? 'bg-amber-400' : 'bg-red-400'}`}
                              style={{ width: `${agg.avg_top1_score * 100}%` }} />
                          </div>
                          <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${hub.isHub ? 'text-red-500 bg-red-50 border-red-200' : scoreColor(agg.avg_top1_score)}`}>
                            {agg.avg_top1_score.toFixed(3)}
                            {hub.isHub && <span className="ml-0.5 text-[9px] text-red-400">↑과장</span>}
                          </span>
                        </div>
                      </td>
                      <td className="px-2 py-3 text-center">
                        <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${hub.isHub ? 'text-red-500 bg-red-50 border-red-200' : scoreColor(agg.mrr)}`}>
                          {agg.mrr.toFixed(3)}
                          {hub.isHub && <span className="ml-0.5 text-[9px] text-red-400">↑과장</span>}
                        </span>
                      </td>
                      <td className="px-2 py-3 text-center">
                        <div className="flex flex-col items-center gap-0.5">
                          <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${scoreColor(agg.dynamic_mrr)}`}>
                            {agg.dynamic_mrr.toFixed(3)}
                          </span>
                          <span className="text-[9px] text-indigo-400">thr={agg.dynamic_threshold.toFixed(3)}</span>
                        </div>
                      </td>
                      <td className="px-2 py-3 text-center">
                        {hub.isHub ? (
                          <div className="flex flex-col items-center gap-0.5">
                            <span className="font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] text-red-500 bg-red-50 border-red-200 line-through opacity-60">
                              {(agg.hit_at_1 * 100).toFixed(0)}%
                            </span>
                            <span className="font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] bg-orange-100 text-orange-700 border-orange-200">
                              → {(hub.adjustedHit1 * 100).toFixed(0)}%
                            </span>
                          </div>
                        ) : (
                          <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${scoreColor(agg.hit_at_1)}`}>
                            {(agg.hit_at_1 * 100).toFixed(0)}%
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-3 text-center">
                        <div className="flex flex-col items-center gap-0.5">
                          <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${scoreColor(agg.dynamic_hit_at_1)}`}>
                            {(agg.dynamic_hit_at_1 * 100).toFixed(0)}%
                          </span>
                          <span className="text-[9px] text-indigo-400">D</span>
                        </div>
                      </td>
                      <td className="px-2 py-3 text-center">
                        <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${scoreColor(agg.hit_at_3)}`}>
                          {(agg.hit_at_3 * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-2 py-3 text-center">
                        <span className={`font-bold tabular-nums px-1.5 py-0.5 rounded border text-[11px] ${
                          agg.score_std <= 0.08 ? 'bg-emerald-100 text-emerald-700 border-emerald-200' :
                          agg.score_std <= 0.15 ? 'bg-amber-100 text-amber-700 border-amber-200' :
                          'bg-red-100 text-red-600 border-red-200'
                        }`}>
                          ±{agg.score_std.toFixed(3)}
                        </span>
                      </td>
                      <td className="px-3 py-2 w-32">
                        <RankSparkline rankScores={agg.rank_scores} />
                      </td>
                      <td className="px-3 py-2 min-w-[100px]">
                        <DistBar high={agg.score_distribution.high} mid={agg.score_distribution.mid} low={agg.score_distribution.low} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`text-[11px] font-medium px-2 py-1 rounded-lg border transition-colors
                          ${isOpen ? 'bg-indigo-100 text-indigo-700 border-indigo-300' : 'bg-gray-50 text-gray-500 border-gray-200 hover:bg-indigo-50 hover:text-indigo-600'}`}>
                          {isOpen ? '닫기 ▲' : '분석 ▼'}
                        </span>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr key={`${eid}-analysis`}>
                        <td colSpan={12}>
                          <ModelAnalysisDrawer
                            embId={eid}
                            agg={agg}
                            questions={questions}
                            results={results}
                          />
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── 질문별 Top-1 유사도 ────────────────────────────────────────── */}
      <div className="bg-white rounded-xl border border-gray-200">
        <div className="px-5 py-4 border-b border-gray-100">
          <h4 className="text-sm font-semibold text-gray-700">질문별 Top-1 유사도</h4>
          <p className="text-xs text-gray-400 mt-0.5">행을 클릭하면 각 모델이 반환한 청크를 볼 수 있습니다</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="text-left py-2 px-5 text-gray-400 font-medium w-1/2">질문</th>
                {sorted.map((eid) => (
                  <th key={eid} className="text-center px-3 py-2 text-gray-500 font-medium whitespace-nowrap">
                    {shortId(eid)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {questions.map((q: TestQuestion) => {
                const isExpanded = expandedQuery === q.question
                const bestScore = Math.max(...sorted.map((eid) => results[q.question]?.[eid]?.[0]?.score ?? 0))
                return (
                  <>
                    <tr
                      key={q.id}
                      onClick={() => onExpandQuery(isExpanded ? null : q.question)}
                      className="border-b border-gray-50 hover:bg-gray-50 cursor-pointer"
                    >
                      <td className="py-2.5 px-5">
                        <div className="flex items-center gap-2">
                          <span className={`text-gray-300 transition-transform ${isExpanded ? 'rotate-90' : ''}`}>▶</span>
                          <span className="text-xs text-gray-400 shrink-0">[{q.category}]</span>
                          <span className="text-xs text-gray-700 line-clamp-1">{q.question}</span>
                        </div>
                      </td>
                      {sorted.map((eid) => {
                        const chunks = results[q.question]?.[eid] ?? []
                        const top1 = chunks[0]?.score ?? null
                        const isBest = top1 !== null && top1 === bestScore && sorted.length > 1
                        return (
                          <td key={eid} className="px-3 py-2.5 text-center">
                            {top1 !== null ? (
                              <span className={`inline-block px-2 py-0.5 rounded border font-bold tabular-nums
                                ${scoreColor(top1)} ${isBest ? 'ring-1 ring-indigo-400' : ''}`}>
                                {top1.toFixed(3)}
                              </span>
                            ) : (
                              <span className="text-gray-300">—</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                    {isExpanded && (
                      <tr key={`${q.id}-expanded`} className="bg-gray-50">
                        <td colSpan={sorted.length + 1} className="px-5 py-3">
                          <div
                            className="grid gap-3"
                            style={{ gridTemplateColumns: `repeat(${Math.min(sorted.length, 3)}, 1fr)` }}
                          >
                            {sorted.map((eid) => {
                              const chunks = results[q.question]?.[eid] ?? []
                              return (
                                <div key={eid} className="bg-white rounded-lg border border-gray-200 p-3">
                                  <div className="flex items-center gap-2 mb-2">
                                    <EmbeddingBadge provider={eid.split(':')[0] as 'local' | 'openai' | 'ollama'} />
                                    <span className="text-xs font-semibold text-gray-700">{shortId(eid)}</span>
                                  </div>
                                  {chunks.length === 0 ? (
                                    <p className="text-xs text-gray-400">결과 없음</p>
                                  ) : (
                                    <div className="space-y-1.5">
                                      {chunks.map((c, i) => (
                                        <div key={i} className="text-xs">
                                          <div className="flex items-center gap-1.5 mb-0.5">
                                            <span className="text-gray-400 w-4">#{i + 1}</span>
                                            <div className="flex-1 h-1 bg-gray-100 rounded-full overflow-hidden">
                                              <div className={`h-full rounded-full ${c.score >= 0.7 ? 'bg-emerald-400' : c.score >= 0.5 ? 'bg-amber-400' : 'bg-red-300'}`}
                                                style={{ width: `${c.score * 100}%` }} />
                                            </div>
                                            <span className={`font-bold w-10 text-right ${scoreColor(c.score)}`}>{c.score.toFixed(3)}</span>
                                          </div>
                                          <p className="text-gray-600 line-clamp-2 leading-relaxed pl-5">{c.text}</p>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function EmbeddingTab({ docs }: { docs: DocumentInfo[] }) {
  const [mode, setMode] = useState<EmbMode>('auto')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [similarityThreshold, setSimilarityThreshold] = useState(0.0)
  const [selEmb, setSelEmb] = useState<string[]>([])
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null)

  const { data: embModels = [] } = useQuery({ queryKey: ['embeddingModels'], queryFn: listEmbeddingModels })
  const { data: questions = [] } = useQuery({ queryKey: ['questions'], queryFn: () => listQuestions() })

  const indexedEmbIds = new Set(docs.flatMap((d) => Object.keys(d.processed_embeddings)))

  const retrieveMut = useMutation({ mutationFn: debugRetrieve })
  const batchMut = useMutation({ mutationFn: debugBatchRetrieve })

  const toggleEmb = (id: string) =>
    setSelEmb((p) => p.includes(id) ? p.filter((x) => x !== id) : [...p, id])

  const handleManualSearch = () => {
    if (!query.trim() || selEmb.length === 0) return
    retrieveMut.mutate({ query, embedding_model_ids: selEmb, top_k: topK, similarity_threshold: similarityThreshold })
  }

  const handleAutoEval = () => {
    if (questions.length === 0 || selEmb.length === 0) return
    batchMut.mutate({
      queries: questions.map((q: TestQuestion) => q.question),
      embedding_model_ids: selEmb,
      top_k: topK,
      similarity_threshold: similarityThreshold,
    })
  }

  const manualResults = retrieveMut.data?.results ?? {}
  const manualModelIds = Object.keys(manualResults)

  const batchResults = batchMut.data?.results ?? {}
  const batchAggregates = batchMut.data?.aggregates ?? {}
  const batchModelIds = Object.keys(batchAggregates)

  // Shared settings panel
  const settingsPanel = (
    <div className="bg-white rounded-xl border border-gray-200 p-5 mb-5">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <label className="text-xs font-medium text-gray-500 block mb-1.5">
            반환 청크 수 <span className="text-indigo-600 font-bold">{topK}</span>
          </label>
          <input type="range" min={1} max={10} value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full accent-indigo-600" />
        </div>
        <div>
          <label className="text-xs font-medium text-gray-500 block mb-1.5">
            유사도 임계값 <span className="text-indigo-600 font-bold">
              {similarityThreshold === 0 ? '없음' : similarityThreshold.toFixed(2)}
            </span>
          </label>
          <input type="range" min={0} max={0.8} step={0.05} value={similarityThreshold}
            onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
            className="w-full accent-indigo-600" />
        </div>
        <div className="col-span-2">
          <label className="text-xs font-medium text-gray-500 block mb-1.5">비교할 임베딩 모델</label>
          <div className="flex flex-wrap gap-1.5">
            {embModels.map((m: EmbeddingModelInfo) => {
              const hasData = indexedEmbIds.has(m.id)
              const checked = selEmb.includes(m.id)
              return (
                <button
                  key={m.id}
                  onClick={() => hasData && toggleEmb(m.id)}
                  disabled={!hasData}
                  title={!hasData ? '문서가 이 모델로 벡터화되지 않았습니다' : m.name}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors
                    ${!hasData ? 'opacity-30 cursor-not-allowed border-gray-200 text-gray-400' :
                      checked
                        ? 'border-indigo-500 bg-indigo-600 text-white'
                        : 'border-gray-200 text-gray-600 hover:border-indigo-300 hover:bg-indigo-50'}`}
                >
                  {shortId(m.id)}
                </button>
              )
            })}
          </div>
          {indexedEmbIds.size === 0 && (
            <p className="text-xs text-amber-500 mt-1">문서 관리 페이지에서 벡터화 후 검색이 가능합니다</p>
          )}
        </div>
      </div>
    </div>
  )

  return (
    <div className="max-w-6xl mx-auto px-6 py-6">
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-gray-500">
          LLM 없이 임베딩 검색만 실행해 각 모델의 검색 품질을 비교합니다.
        </p>
        {/* Mode toggle */}
        <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5 shrink-0">
          {([['auto', `자동 평가 (${questions.length}개 질문)`], ['manual', '수동 쿼리']] as const).map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors
                ${mode === m ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {settingsPanel}

      {/* ── Auto mode ─────────────────────────────────────────────────── */}
      {mode === 'auto' && (
        <div>
          {questions.length === 0 ? (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-center">
              <p className="text-sm text-amber-700">테스트 질문 페이지에서 질문을 먼저 등록해주세요.</p>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-4 mb-5">
                <button
                  onClick={handleAutoEval}
                  disabled={selEmb.length === 0 || batchMut.isPending}
                  className="px-5 py-2 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm"
                >
                  {batchMut.isPending
                    ? `평가 중... (${questions.length}개 질문)`
                    : `테스트 질문 ${questions.length}개 자동 평가`}
                </button>
                {selEmb.length === 0 && (
                  <p className="text-xs text-gray-400">위에서 임베딩 모델을 선택해주세요</p>
                )}
                {batchMut.data && (
                  <span className="text-xs text-gray-400 ml-auto">
                    {batchModelIds.length}개 모델 · {questions.length}개 질문 · top-{batchMut.data.top_k}
                  </span>
                )}
              </div>

              {batchMut.isPending && (
                <div className="flex justify-center py-12"><LoadingSpinner size="lg" /></div>
              )}

              {batchMut.isError && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-600 text-sm">
                  평가 오류: {String(batchMut.error)}
                </div>
              )}

              {batchModelIds.length > 0 && !batchMut.isPending && (
                <EmbedMetricsPanel
                  modelIds={batchModelIds}
                  aggregates={batchAggregates}
                  results={batchResults}
                  questions={questions as TestQuestion[]}
                  topK={batchMut.data?.top_k ?? 5}
                  expandedQuery={expandedQuery}
                  onExpandQuery={setExpandedQuery}
                />
              )}
            </>
          )}
        </div>
      )}

      {/* ── Manual mode ───────────────────────────────────────────────── */}
      {mode === 'manual' && (
        <div>
          <div className="flex gap-3 mb-5">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleManualSearch()}
              placeholder="검색 쿼리 입력 (예: 배송 기간이 얼마나 걸리나요?)"
              className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            {/* Quick-fill from test questions */}
            {questions.length > 0 && (
              <select
                defaultValue=""
                onChange={(e) => { if (e.target.value) setQuery(e.target.value) }}
                className="border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="" disabled>질문에서 불러오기</option>
                {questions.map((q: TestQuestion) => (
                  <option key={q.id} value={q.question}>{q.question.slice(0, 40)}{q.question.length > 40 ? '…' : ''}</option>
                ))}
              </select>
            )}
            <button
              onClick={handleManualSearch}
              disabled={!query.trim() || selEmb.length === 0 || retrieveMut.isPending}
              className="px-5 py-2 bg-indigo-600 text-white font-medium rounded-lg text-sm hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              {retrieveMut.isPending ? '검색 중...' : '검색'}
            </button>
          </div>

          {retrieveMut.isPending && (
            <div className="flex justify-center py-12"><LoadingSpinner size="lg" /></div>
          )}
          {retrieveMut.isError && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-600 text-sm">
              검색 오류: {String(retrieveMut.error)}
            </div>
          )}

          {manualModelIds.length > 0 && !retrieveMut.isPending && (
            <div>
              <p className="text-xs text-gray-400 mb-3">
                쿼리: <span className="font-medium text-gray-600">"{retrieveMut.data?.query}"</span>
                &nbsp;· top-{retrieveMut.data?.top_k}
              </p>
              <div
                className="grid gap-4"
                style={{ gridTemplateColumns: `repeat(${Math.min(manualModelIds.length, 3)}, 1fr)` }}
              >
                {manualModelIds.map((embId) => {
                  const chunks = manualResults[embId]
                  const avgScore = chunks.length
                    ? chunks.reduce((s, c) => s + c.score, 0) / chunks.length
                    : 0
                  return (
                    <div key={embId} className="bg-white rounded-xl border border-gray-200 flex flex-col">
                      <div className="px-4 py-3 border-b border-gray-100">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-semibold text-gray-800">{shortId(embId)}</span>
                          <EmbeddingBadge provider={embId.split(':')[0] as 'local' | 'openai' | 'ollama'} />
                        </div>
                        <div className="flex items-center gap-3 text-xs text-gray-400">
                          <span>{chunks.length}청크 반환</span>
                          {chunks.length > 0 && (
                            <span className={`px-1.5 py-0.5 rounded border text-xs font-medium ${scoreColor(avgScore)}`}>
                              평균 {avgScore.toFixed(3)}
                            </span>
                          )}
                        </div>
                      </div>
                      {chunks.length === 0 ? (
                        <div className="flex-1 flex items-center justify-center py-8 text-gray-400 text-sm">결과 없음</div>
                      ) : (
                        <div className="flex-1 p-3 space-y-2 overflow-auto max-h-[600px]">
                          {chunks.map((chunk, i) => (
                            <div key={i} className="border border-gray-100 rounded-lg p-3">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-xs font-bold text-gray-400">#{i + 1}</span>
                                <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full rounded-full transition-all ${
                                      chunk.score >= 0.7 ? 'bg-emerald-400' :
                                      chunk.score >= 0.45 ? 'bg-amber-400' : 'bg-red-400'
                                    }`}
                                    style={{ width: `${chunk.score * 100}%` }}
                                  />
                                </div>
                                <span className={`text-xs font-bold px-1.5 py-0.5 rounded border tabular-nums ${scoreColor(chunk.score)}`}>
                                  {chunk.score.toFixed(3)}
                                </span>
                              </div>
                              <p className="text-xs text-gray-600 leading-relaxed">{chunk.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>

              {manualModelIds.length > 1 && (
                <div className="mt-5 bg-white rounded-xl border border-gray-200 p-5">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">순위별 유사도 비교</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-gray-100">
                          <th className="text-left py-2 pr-4 text-gray-400 font-medium">순위</th>
                          {manualModelIds.map((eid) => (
                            <th key={eid} className="text-center px-3 py-2 text-gray-600 font-medium whitespace-nowrap">
                              {shortId(eid)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: Math.max(...manualModelIds.map((eid) => manualResults[eid].length)) }).map((_, rank) => (
                          <tr key={rank} className="border-b border-gray-50">
                            <td className="py-2 pr-4 text-gray-400">#{rank + 1}</td>
                            {manualModelIds.map((eid) => {
                              const chunk = manualResults[eid][rank]
                              if (!chunk) return <td key={eid} className="px-3 py-2 text-center text-gray-300">—</td>
                              return (
                                <td key={eid} className="px-3 py-2 text-center">
                                  <span className={`inline-block px-2 py-0.5 rounded border font-bold tabular-nums ${scoreColor(chunk.score)}`}>
                                    {chunk.score.toFixed(3)}
                                  </span>
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                        <tr className="bg-gray-50">
                          <td className="py-2 pr-4 text-gray-500 font-medium">평균</td>
                          {manualModelIds.map((eid) => {
                            const chunks = manualResults[eid]
                            const avg = chunks.length ? chunks.reduce((s, c) => s + c.score, 0) / chunks.length : 0
                            return (
                              <td key={eid} className="px-3 py-2 text-center">
                                <span className={`inline-block px-2 py-0.5 rounded border font-bold tabular-nums ${scoreColor(avg)}`}>
                                  {avg.toFixed(3)}
                                </span>
                              </td>
                            )
                          })}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── LLM Tab ───────────────────────────────────────────────────────────────────

function LlmTab() {
  const navigate = useNavigate()

  const [selEmb, setSelEmb] = useState<string[]>([])
  const [selLLM, setSelLLM] = useState<string[]>([])
  const [ragEnabled, setRagEnabled] = useState(true)
  const [topK, setTopK] = useState(3)
  const [similarityThreshold, setSimilarityThreshold] = useState(0.0)
  const [runName, setRunName] = useState('')

  const [runId, setRunId] = useState<string | null>(null)
  const [runPhase, setRunPhase] = useState<'config' | 'running' | 'done'>('config')
  const [liveResults, setLiveResults] = useState<TestResult[]>([])
  const [totalTests, setTotalTests] = useState(0)

  const { data: embModels = [] } = useQuery({ queryKey: ['embeddingModels'], queryFn: listEmbeddingModels })
  const { data: llmModels = [] } = useQuery({ queryKey: ['llmModels'], queryFn: listLLMModels })
  const { data: questions = [] } = useQuery({ queryKey: ['questions'], queryFn: () => listQuestions() })
  const { data: docs = [] } = useQuery({ queryKey: ['documents'], queryFn: listDocuments })

  // Judge model is always the primary (first) selected LLM
  const judgeModel = selLLM[0] ?? ''

  const indexedEmbIds = new Set(
    (docs as DocumentInfo[]).flatMap((d) => Object.keys(d.processed_embeddings))
  )

  const createMut = useMutation({ mutationFn: createRun })

  const toggleEmb = (id: string) =>
    setSelEmb((p) => p.includes(id) ? p.filter((x) => x !== id) : [...p, id])
  const toggleLLM = (id: string) =>
    setSelLLM((p) => p.includes(id) ? p.filter((x) => x !== id) : [...p, id])

  const nCombinations = ragEnabled ? selEmb.length * selLLM.length : selLLM.length
  const nTotal = nCombinations * questions.length

  const handleStart = async () => {
    const run = await createMut.mutateAsync({
      embedding_model_ids: ragEnabled ? selEmb : [],
      llm_model_ids: selLLM,
      question_ids: null,
      rag_enabled: ragEnabled,
      top_k: topK,
      similarity_threshold: similarityThreshold,
      search_strategy: 'semantic',
      judge_model: judgeModel,
      run_name: runName || null,
    })
    setRunId(run.id)
    setTotalTests(run.total_tests)
    setRunPhase('running')
    setLiveResults([])
  }

  useEffect(() => {
    if (!runId || runPhase !== 'running') return
    const es = new EventSource(`/api/runs/${runId}/stream`)
    es.addEventListener('result', (e) => setLiveResults((p) => [...p, JSON.parse(e.data as string)]))
    es.addEventListener('complete', () => { setRunPhase('done'); es.close() })
    es.onerror = () => { setRunPhase('done'); es.close() }
    return () => es.close()
  }, [runId, runPhase])

  // ── Config view ──────────────────────────────────────────────────────────────
  if (runPhase === 'config') {
    const canStart = selLLM.length > 0 && (!ragEnabled || selEmb.length > 0)

    return (
      <div className="max-w-5xl mx-auto px-6 py-6">
        <div className="mb-4">
          <p className="text-sm text-gray-500">
            임베딩 × LLM 조합을 선택해 자동 평가를 실행합니다. Judge 모델이 각 답변을 0–10점으로 채점합니다.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-5 mb-5">
          {/* Embedding models */}
          {ragEnabled && (
            <div className="bg-white rounded-xl border border-gray-200">
              <div className="px-5 py-4 border-b border-gray-100">
                <h2 className="font-semibold text-gray-800 text-sm">임베딩 모델</h2>
                <p className="text-xs text-gray-400 mt-0.5">RAG 검색 — 문서가 벡터화된 모델만 가능</p>
              </div>
              <div className="p-4 space-y-2">
                {embModels.map((m: EmbeddingModelInfo) => {
                  const hasData = indexedEmbIds.has(m.id)
                  const checked = selEmb.includes(m.id)
                  const disabled = !hasData
                  return (
                    <label key={m.id}
                      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all select-none
                        ${disabled ? 'opacity-40 cursor-not-allowed border-gray-100 bg-gray-50' :
                          checked ? 'border-indigo-300 bg-indigo-50' : 'border-gray-200 hover:border-gray-300'}`}>
                      <input type="checkbox" checked={checked} disabled={disabled}
                        onChange={() => !disabled && toggleEmb(m.id)}
                        className="mt-0.5 accent-indigo-600 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-sm font-medium text-gray-800">{m.name}</span>
                          <EmbeddingBadge provider={m.provider} />
                        </div>
                        <div className="text-xs text-gray-400 mt-0.5">
                          {m.dimensions}d
                          {hasData
                            ? <span className="ml-2 text-emerald-600">· 인덱싱됨</span>
                            : <span className="ml-2 text-amber-500">· 문서 벡터화 필요</span>
                          }
                        </div>
                      </div>
                    </label>
                  )
                })}
              </div>
            </div>
          )}

          {/* LLM models */}
          <div className={`bg-white rounded-xl border border-gray-200 ${!ragEnabled ? 'col-span-2' : ''}`}>
            <div className="px-5 py-4 border-b border-gray-100">
              <h2 className="font-semibold text-gray-800 text-sm">LLM 모델</h2>
              <p className="text-xs text-gray-400 mt-0.5">답변 생성 — 여러 개 선택 시 비교</p>
            </div>
            <div className="p-4 space-y-2">
              {llmModels.map((m: LLMModelInfo) => {
                const checked = selLLM.includes(m.id)
                return (
                  <label key={m.id}
                    className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all select-none
                      ${!m.available ? 'opacity-40 cursor-not-allowed border-gray-100 bg-gray-50' :
                        checked ? 'border-indigo-300 bg-indigo-50' : 'border-gray-200 hover:border-gray-300'}`}>
                    <input type="checkbox" checked={checked} disabled={!m.available}
                      onChange={() => m.available && toggleLLM(m.id)}
                      className="mt-0.5 accent-indigo-600 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-medium text-gray-800">{m.name}</span>
                        <LLMBadge provider={m.provider} />
                      </div>
                      {m.description && <p className="text-xs text-gray-400 mt-0.5">{m.description}</p>}
                    </div>
                  </label>
                )
              })}
            </div>
          </div>
        </div>

        {/* Settings + launch */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-5">
            {/* RAG toggle */}
            <div className="col-span-2 md:col-span-1">
              <label className="text-xs font-medium text-gray-500 block mb-2">RAG</label>
              <button type="button" onClick={() => setRagEnabled((v) => !v)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors w-full
                  ${ragEnabled ? 'border-indigo-300 bg-indigo-50 text-indigo-700' : 'border-gray-200 text-gray-500 hover:bg-gray-50'}`}>
                <div className={`w-8 h-4 rounded-full relative transition-colors ${ragEnabled ? 'bg-indigo-500' : 'bg-gray-300'}`}>
                  <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${ragEnabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                </div>
                {ragEnabled ? '켜짐' : '꺼짐'}
              </button>
            </div>

            {ragEnabled && (
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-2">
                  검색 청크 수 <span className="text-indigo-600 font-bold">{topK}</span>
                </label>
                <input type="range" min={1} max={10} value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full accent-indigo-600" />
                <div className="flex justify-between text-xs text-gray-400 mt-1"><span>1</span><span>10</span></div>
              </div>
            )}

            {ragEnabled && (
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-2">
                  유사도 임계값 <span className="text-indigo-600 font-bold">{similarityThreshold === 0 ? '없음' : similarityThreshold.toFixed(2)}</span>
                </label>
                <input type="range" min={0} max={0.8} step={0.05} value={similarityThreshold}
                  onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
                  className="w-full accent-indigo-600" />
                <div className="flex justify-between text-xs text-gray-400 mt-1"><span>0</span><span>0.8</span></div>
              </div>
            )}

            <div>
              <label className="text-xs font-medium text-gray-500 block mb-2">평가 모델 (Judge)</label>
              <div className="w-full border border-gray-100 bg-gray-50 rounded-lg px-3 py-2 text-sm text-gray-600">
                {judgeModel
                  ? llmModels.find((m: LLMModelInfo) => m.id === judgeModel)?.name ?? judgeModel
                  : <span className="text-gray-400">LLM 모델을 선택하면 자동 설정됩니다</span>
                }
              </div>
              <p className="text-xs text-gray-400 mt-1">첫 번째로 선택한 LLM이 Judge로 사용됩니다</p>
            </div>
          </div>

          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="text-xs font-medium text-gray-500 block mb-1.5">실행 이름 (선택)</label>
              <input value={runName} onChange={(e) => setRunName(e.target.value)}
                placeholder="예: bge-m3 vs nomic / Haiku vs Sonnet"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" />
            </div>
            <div className="shrink-0 text-right">
              {canStart && (
                <div className="text-xs text-gray-400 mb-1.5">
                  {ragEnabled
                    ? `${selEmb.length}임베딩 × ${selLLM.length}LLM × ${questions.length}질문 = ${nTotal}건`
                    : `${selLLM.length}LLM × ${questions.length}질문 = ${nTotal}건`
                  }
                </div>
              )}
              <button
                onClick={handleStart}
                disabled={!canStart || createMut.isPending}
                className="px-6 py-2 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm"
              >
                {createMut.isPending ? '시작 중...' : '▶ 테스트 시작'}
              </button>
              {!canStart && (
                <p className="text-xs text-gray-400 mt-1">
                  {selLLM.length === 0 ? 'LLM을 선택해주세요' : '임베딩 모델을 선택해주세요'}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  // ── Running / done view ──────────────────────────────────────────────────────
  const progress = totalTests > 0 ? liveResults.length / totalTests : 0
  const pairMap: Record<string, TestResult[]> = {}
  for (const r of liveResults) {
    const key = `${r.embedding_model_id ?? 'none'}||${r.llm_model_id}`
    pairMap[key] = pairMap[key] ?? []
    pairMap[key].push(r)
  }
  const pairKeys = Object.keys(pairMap)

  return (
    <div className="max-w-6xl mx-auto px-6 py-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-base font-semibold text-gray-900">
            {runPhase === 'running' ? '실행 중...' : '완료'}
          </h2>
          <p className="text-sm text-gray-400 mt-0.5">{liveResults.length} / {totalTests}건</p>
        </div>
        {runPhase === 'done' && (
          <button onClick={() => navigate(`/results/${runId}`)}
            className="px-5 py-2 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 text-sm">
            결과 상세 보기 →
          </button>
        )}
      </div>

      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden mb-6">
        <div className="h-full bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${progress * 100}%` }} />
      </div>

      <div className="grid gap-3"
        style={{ gridTemplateColumns: `repeat(${Math.min(pairKeys.length || 1, 3)}, 1fr)` }}>
        {pairKeys.map((key) => {
          const [embId, llmId] = key.split('||')
          const rs = pairMap[key]
          const scored = rs.filter((r) => r.scores)
          const avgOverall = scored.length
            ? scored.reduce((s, r) => s + (r.scores?.overall ?? 0), 0) / scored.length
            : 0
          const avgLatency = rs.reduce((s, r) => s + r.latency_ms, 0) / rs.length

          return (
            <div key={key} className="bg-white border border-gray-200 rounded-xl p-4">
              {ragEnabled && (
                <>
                  <div className="text-xs text-gray-400 mb-0.5">임베딩</div>
                  <div className="text-sm font-medium text-gray-700 truncate mb-2">{shortId(embId === 'none' ? null : embId)}</div>
                </>
              )}
              <div className="text-xs text-gray-400 mb-0.5">LLM</div>
              <div className="text-sm font-medium text-gray-700 truncate mb-3">{shortId(llmId)}</div>
              <div className="border-t border-gray-100 pt-3">
                <div className="text-xs text-gray-400 mb-1">{rs.length}건 완료</div>
                {scored.length > 0 && (
                  <>
                    <ScoreBar score={avgOverall} label="종합" />
                    <div className="text-xs text-gray-400 mt-1.5">{avgLatency.toFixed(0)}ms</div>
                  </>
                )}
              </div>
              {rs.slice(-1).map((r) => (
                <div key={r.id} className="mt-2 text-xs text-gray-400 bg-gray-50 rounded px-2 py-1.5 truncate">
                  {r.error
                    ? <span className="text-red-400">{r.error.slice(0, 50)}</span>
                    : r.response.slice(0, 60) + '…'}
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function RunTestPage() {
  const [activeTab, setActiveTab] = useState<Tab>('embedding')
  const { data: docs = [] } = useQuery({ queryKey: ['documents'], queryFn: listDocuments })

  return (
    <div className="min-h-screen">
      {/* Tab bar */}
      <div className="bg-white border-b border-gray-200 sticky top-12 z-10">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex gap-1 py-2">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex flex-col px-4 py-2 rounded-lg text-left transition-colors
                  ${activeTab === tab.id
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-500 hover:bg-gray-100 hover:text-gray-800'}`}
              >
                <span className="text-sm font-semibold leading-tight">{tab.label}</span>
                <span className={`text-xs mt-0.5 ${activeTab === tab.id ? 'text-indigo-200' : 'text-gray-400'}`}>
                  {tab.desc}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab content */}
      {activeTab === 'embedding' && <EmbeddingTab docs={docs as DocumentInfo[]} />}
      {activeTab === 'llm'       && <LlmTab />}
    </div>
  )
}
