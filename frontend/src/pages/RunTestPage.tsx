import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  listEmbeddingModels, listLLMModels, listQuestions,
  createRun, listDocuments, getDocumentText, debugRetrieve, debugBatchRetrieve,
} from '../lib/api'
import type { DocumentInfo, EmbeddingModelInfo, LLMModelInfo, TestQuestion, TestResult } from '../types'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { LLMBadge, EmbeddingBadge } from '../components/common/ModelBadge'
import ScoreBar from '../components/common/ScoreBar'

// ── Helpers ───────────────────────────────────────────────────────────────────

// Judge quality tiers — used to surface a warning when a weak model is selected
const JUDGE_QUALITY: Record<string, 'high' | 'medium' | 'low'> = {
  'claude-sonnet-4-6': 'high',
  'claude-opus-4-6': 'high',
  'gemini-2.5-pro': 'high',
  'gemini-2.5-flash': 'high',
  'claude-haiku-4-5-20251001': 'medium',
  'gemini-2.0-flash': 'medium',
}

function shortId(id: string | null) {
  return (id ?? 'no-rag')
    .replace('claude-', 'C-').replace('-20251001', '')
    .replace('gemini-', 'G-')
    .replace('local:', '').replace('openai:', '').replace('ollama:', '')
    .split(':')[0]
    .replace('all-MiniLM-L6-v2', 'MiniLM')
}

function scoreColor(score: number) {
  if (score >= 0.7) return 'bg-emerald-100 text-emerald-700 border-emerald-200'
  if (score >= 0.45) return 'bg-amber-100 text-amber-700 border-amber-200'
  return 'bg-red-100 text-red-600 border-red-200'
}

// ── Tab type ──────────────────────────────────────────────────────────────────

type Tab = 'ocr' | 'embedding' | 'llm'

const TABS: { id: Tab; label: string; desc: string }[] = [
  { id: 'ocr',       label: '① OCR 검증',    desc: '문서에서 추출된 텍스트와 청크를 확인' },
  { id: 'embedding', label: '② 임베딩 검색',  desc: 'LLM 없이 검색 품질만 비교' },
  { id: 'llm',       label: '③ LLM 비교',     desc: '전체 파이프라인 자동 평가' },
]

// Chunking presets (shared with SetupPage)
const CHUNK_PRESETS = [
  { label: '소형 300/50',   chunkSize: 300,  overlap: 50 },
  { label: '표준 600/100',  chunkSize: 600,  overlap: 100 },
  { label: '대형 1000/150', chunkSize: 1000, overlap: 150 },
]

// ── OCR helpers ───────────────────────────────────────────────────────────────

/** Extract meaningful Korean/English tokens from a question string. */
function extractTokens(text: string): string[] {
  // Split on spaces + punctuation, keep tokens ≥ 2 chars
  return text
    .split(/[\s,?!.。：:·\-\/()[\]{}'"]+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2)
}

/** Count how many tokens from the question appear in fullText (case-insensitive). */
function coverageScore(question: string, fullText: string): { matched: string[]; missing: string[]; ratio: number } {
  const tokens = extractTokens(question)
  if (tokens.length === 0) return { matched: [], missing: [], ratio: 0 }
  const lower = fullText.toLowerCase()
  const matched = tokens.filter((t) => lower.includes(t.toLowerCase()))
  const missing = tokens.filter((t) => !lower.includes(t.toLowerCase()))
  return { matched, missing, ratio: matched.length / tokens.length }
}

// ── OCR Tab ───────────────────────────────────────────────────────────────────

function OcrTab({ docs }: { docs: DocumentInfo[] }) {
  const [selDocId, setSelDocId] = useState<string | null>(null)
  const [showAllChunks, setShowAllChunks] = useState(false)
  const [viewMode, setViewMode] = useState<'preview' | 'chunks' | 'coverage'>('preview')
  const [chunkSize, setChunkSize] = useState(600)
  const [overlap, setOverlap] = useState(100)

  const { data: textData, isLoading, error } = useQuery({
    queryKey: ['docText', selDocId, chunkSize, overlap],
    queryFn: () => getDocumentText(selDocId!, chunkSize, overlap),
    enabled: !!selDocId,
    staleTime: 30_000,
  })

  const { data: questions = [] } = useQuery({
    queryKey: ['questions'],
    queryFn: listQuestions,
  })

  const selectedDoc = docs.find((d) => d.id === selDocId)

  // Compute coverage for all questions once we have the full text
  const fullText = textData
    ? (textData.chunks.map((c) => c.text).join('\n') || textData.text_preview)
    : ''

  const coverageResults = questions.map((q: TestQuestion) => ({
    question: q,
    ...coverageScore(q.question, fullText),
  }))

  const avgCoverage = coverageResults.length
    ? coverageResults.reduce((s, r) => s + r.ratio, 0) / coverageResults.length
    : 0

  const highCount  = coverageResults.filter((r) => r.ratio >= 0.7).length
  const midCount   = coverageResults.filter((r) => r.ratio >= 0.3 && r.ratio < 0.7).length
  const lowCount   = coverageResults.filter((r) => r.ratio < 0.3).length

  return (
    <div className="max-w-5xl mx-auto px-6 py-6">
      <div className="mb-4">
        <p className="text-sm text-gray-500">
          PDF·DOCX에서 추출된 원문 텍스트를 확인하고, 테스트 질문들이 문서에서 얼마나 커버되는지 자동으로 평가합니다.
        </p>
      </div>

      {/* Chunking strategy panel */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 mb-5">
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-xs font-semibold text-gray-500 shrink-0">청킹 전략</span>

          {/* Presets */}
          <div className="flex gap-1">
            {CHUNK_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => { setChunkSize(p.chunkSize); setOverlap(p.overlap); setShowAllChunks(false) }}
                className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors
                  ${chunkSize === p.chunkSize && overlap === p.overlap
                    ? 'bg-indigo-600 text-white border-indigo-600'
                    : 'border-gray-200 text-gray-600 hover:border-indigo-300 hover:bg-indigo-50'}`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Fine-tune */}
          <div className="flex items-center gap-3 ml-2 flex-1 min-w-64">
            <div className="flex items-center gap-2 flex-1">
              <span className="text-xs text-gray-400 shrink-0 w-14">청크 크기</span>
              <input type="range" min={100} max={1200} step={50} value={chunkSize}
                onChange={(e) => { setChunkSize(Number(e.target.value)); setShowAllChunks(false) }}
                className="flex-1 accent-indigo-600" />
              <span className="text-xs font-bold text-indigo-600 w-12 text-right">{chunkSize}자</span>
            </div>
            <div className="flex items-center gap-2 flex-1">
              <span className="text-xs text-gray-400 shrink-0 w-12">오버랩</span>
              <input type="range" min={0} max={300} step={25} value={overlap}
                onChange={(e) => { setOverlap(Number(e.target.value)); setShowAllChunks(false) }}
                className="flex-1 accent-indigo-600" />
              <span className="text-xs font-bold text-indigo-600 w-12 text-right">{overlap}자</span>
            </div>
          </div>

          {/* Live stats if data available */}
          {textData && (
            <div className="flex gap-3 text-xs text-gray-400 ml-auto shrink-0">
              <span>청크 <b className="text-gray-700">{textData.chunk_count}</b>개</span>
              <span>평균 <b className="text-gray-700">{textData.avg_chunk_size}</b>자</span>
              <span>범위 <b className="text-gray-700">{textData.min_chunk_size}~{textData.max_chunk_size}</b>자</span>
            </div>
          )}
        </div>
        <p className="text-xs text-gray-400 mt-2">
          청크 크기↑ → 문맥 풍부·노이즈↑ &nbsp;|&nbsp; 오버랩↑ → 경계 손실 방지·청크 수↑ &nbsp;|&nbsp;
          <span className="text-amber-500">벡터화할 때도 같은 설정을 준비 탭에서 맞춰주세요</span>
        </p>
      </div>

      <div className="grid grid-cols-3 gap-5">
        {/* Document selector */}
        <div className="bg-white rounded-xl border border-gray-200">
          <div className="px-4 py-3 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-700">문서 선택</h3>
          </div>
          <ul className="divide-y divide-gray-50">
            {docs.length === 0 ? (
              <li className="px-4 py-6 text-center text-gray-400 text-sm">문서 없음</li>
            ) : (
              docs.map((doc) => {
                const totalChunks = Object.values(doc.processed_embeddings).reduce((a, b) => a + b, 0)
                return (
                  <li
                    key={doc.id}
                    onClick={() => { setSelDocId(doc.id); setShowAllChunks(false) }}
                    className={`px-4 py-3 cursor-pointer transition-colors ${
                      selDocId === doc.id
                        ? 'bg-indigo-50 border-l-2 border-l-indigo-500'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`shrink-0 px-1.5 py-0.5 rounded text-xs font-bold
                        ${doc.file_type === 'pdf' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>
                        {doc.file_type.toUpperCase()}
                      </span>
                      <span className="text-sm text-gray-800 truncate">{doc.filename}</span>
                    </div>
                    {totalChunks > 0 && (
                      <div className="text-xs text-gray-400 mt-1 ml-7">
                        {`${Object.keys(doc.processed_embeddings).length}개 모델 · 최대 ${Math.max(...Object.values(doc.processed_embeddings))}청크`}
                      </div>
                    )}
                  </li>
                )
              })
            )}
          </ul>
        </div>

        {/* Text / chunk / coverage viewer */}
        <div className="col-span-2 bg-white rounded-xl border border-gray-200 flex flex-col">
          {!selDocId ? (
            <div className="flex-1 flex items-center justify-center text-gray-400 text-sm py-20">
              왼쪽에서 문서를 선택하세요
            </div>
          ) : isLoading ? (
            <div className="flex-1 flex items-center justify-center py-20">
              <div className="text-center">
                <LoadingSpinner />
                <p className="text-sm text-gray-400 mt-3">텍스트 추출 중...</p>
              </div>
            </div>
          ) : error ? (
            <div className="flex-1 flex items-center justify-center py-20">
              <p className="text-sm text-red-500">추출 실패: {String(error)}</p>
            </div>
          ) : textData ? (
            <>
              {/* Stats bar */}
              <div className="px-5 py-3 border-b border-gray-100 flex items-center gap-5 flex-wrap">
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-800 tabular-nums">
                    {textData.char_count.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-400">총 문자</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-800 tabular-nums">{textData.chunk_count}</div>
                  <div className="text-xs text-gray-400">청크 수</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-800 tabular-nums">{textData.avg_chunk_size}</div>
                  <div className="text-xs text-gray-400">평균 크기</div>
                </div>
                <div className="text-center">
                  <div className="text-sm font-bold text-gray-500 tabular-nums">
                    {textData.min_chunk_size}~{textData.max_chunk_size}
                  </div>
                  <div className="text-xs text-gray-400">크기 범위</div>
                </div>
                {questions.length > 0 && (
                  <div className="text-center">
                    <div className={`text-lg font-bold tabular-nums ${
                      avgCoverage >= 0.6 ? 'text-emerald-600' : avgCoverage >= 0.35 ? 'text-amber-600' : 'text-red-500'
                    }`}>
                      {(avgCoverage * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-400">질문 커버리지</div>
                  </div>
                )}
                <div className="ml-auto flex gap-1">
                  {(['preview', 'chunks', 'coverage'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setViewMode(m)}
                      className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors
                        ${viewMode === m
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                    >
                      {m === 'preview' ? '원문' : m === 'chunks' ? '청크' : `질문 커버리지 (${questions.length})`}
                    </button>
                  ))}
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-auto p-5">
                {viewMode === 'preview' && (
                  <div>
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed font-sans">
                      {textData.text_preview}
                    </pre>
                    {textData.char_count > 3000 && (
                      <p className="text-xs text-gray-400 mt-3 italic">
                        (앞 3,000자 표시 중 · 전체 {textData.char_count.toLocaleString()}자)
                      </p>
                    )}
                  </div>
                )}

                {viewMode === 'chunks' && (
                  <div className="space-y-2">
                    {(showAllChunks ? textData.chunks : textData.chunks.slice(0, 10)).map((chunk) => (
                      <div key={chunk.index}
                        className="border border-gray-100 rounded-lg p-3 hover:border-indigo-200 transition-colors">
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">
                            #{chunk.index + 1}
                          </span>
                          <span className="text-xs text-gray-400">{chunk.char_count}자</span>
                          <div className="ml-auto h-1 w-20 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-400 rounded-full"
                              style={{ width: `${Math.min(100, (chunk.char_count / 600) * 100)}%` }}
                            />
                          </div>
                        </div>
                        <p className="text-xs text-gray-600 leading-relaxed line-clamp-4">{chunk.text}</p>
                      </div>
                    ))}
                    {textData.chunks.length > 10 && !showAllChunks && (
                      <button
                        onClick={() => setShowAllChunks(true)}
                        className="w-full py-2 text-xs text-indigo-600 border border-dashed border-indigo-200 rounded-lg hover:bg-indigo-50 transition-colors"
                      >
                        +{textData.chunks.length - 10}개 더 보기
                      </button>
                    )}
                  </div>
                )}

                {viewMode === 'coverage' && (
                  <div>
                    {questions.length === 0 ? (
                      <p className="text-sm text-gray-400 text-center py-8">
                        준비 탭에서 테스트 질문을 먼저 등록해주세요.
                      </p>
                    ) : (
                      <>
                        {/* Summary bar */}
                        <div className="flex gap-4 mb-4 p-3 bg-gray-50 rounded-lg text-xs">
                          <span className="text-emerald-600 font-semibold">높음(≥70%) {highCount}개</span>
                          <span className="text-amber-600 font-semibold">중간(30~70%) {midCount}개</span>
                          <span className="text-red-500 font-semibold">낮음(&lt;30%) {lowCount}개</span>
                          <span className="ml-auto text-gray-500">
                            평균 커버리지: <b className="text-gray-700">{(avgCoverage * 100).toFixed(1)}%</b>
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mb-3">
                          각 질문의 핵심 키워드가 추출된 문서 텍스트에 포함되어 있는지 확인합니다.
                          커버리지가 낮으면 OCR 추출이 부족하거나 해당 내용이 문서에 없는 것입니다.
                        </p>
                        <div className="space-y-2">
                          {coverageResults.map(({ question: q, matched, missing, ratio }) => (
                            <div key={q.id}
                              className={`border rounded-lg p-3 transition-colors ${
                                ratio >= 0.7 ? 'border-emerald-200 bg-emerald-50/50' :
                                ratio >= 0.3 ? 'border-amber-200 bg-amber-50/50' :
                                'border-red-200 bg-red-50/50'
                              }`}>
                              <div className="flex items-start gap-3">
                                {/* Coverage bar */}
                                <div className="shrink-0 flex flex-col items-center pt-0.5">
                                  <span className={`text-sm font-bold tabular-nums ${
                                    ratio >= 0.7 ? 'text-emerald-600' :
                                    ratio >= 0.3 ? 'text-amber-600' : 'text-red-500'
                                  }`}>
                                    {(ratio * 100).toFixed(0)}%
                                  </span>
                                  <div className="w-8 h-1 bg-gray-200 rounded-full mt-1 overflow-hidden">
                                    <div
                                      className={`h-full rounded-full ${
                                        ratio >= 0.7 ? 'bg-emerald-400' :
                                        ratio >= 0.3 ? 'bg-amber-400' : 'bg-red-400'
                                      }`}
                                      style={{ width: `${ratio * 100}%` }}
                                    />
                                  </div>
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs text-gray-400 shrink-0">{q.category}</span>
                                  </div>
                                  <p className="text-xs text-gray-700 font-medium mb-1.5">{q.question}</p>
                                  <div className="flex flex-wrap gap-1">
                                    {matched.map((t) => (
                                      <span key={t} className="px-1.5 py-0.5 bg-emerald-100 text-emerald-700 text-xs rounded border border-emerald-200">
                                        {t}
                                      </span>
                                    ))}
                                    {missing.map((t) => (
                                      <span key={t} className="px-1.5 py-0.5 bg-gray-100 text-gray-400 text-xs rounded border border-gray-200 line-through">
                                        {t}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </>
          ) : null}
        </div>
      </div>

      {/* Vectorization status */}
      {selectedDoc && Object.keys(selectedDoc.processed_embeddings).length > 0 && (
        <div className="mt-4 bg-white rounded-xl border border-gray-200 p-4">
          <h4 className="text-xs font-semibold text-gray-500 mb-3">임베딩 모델별 벡터화 현황</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(selectedDoc.processed_embeddings).map(([eid, count]) => (
              <div key={eid}
                className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 border border-emerald-200 rounded-full text-xs text-emerald-700">
                <span className="font-medium">{shortId(eid)}</span>
                <span className="text-emerald-500">{count}청크</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Embedding Tab ─────────────────────────────────────────────────────────────

type EmbMode = 'manual' | 'auto'

function EmbeddingTab({ docs }: { docs: DocumentInfo[] }) {
  const [mode, setMode] = useState<EmbMode>('auto')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [similarityThreshold, setSimilarityThreshold] = useState(0.0)
  const [selEmb, setSelEmb] = useState<string[]>([])
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null)

  const { data: embModels = [] } = useQuery({ queryKey: ['embeddingModels'], queryFn: listEmbeddingModels })
  const { data: questions = [] } = useQuery({ queryKey: ['questions'], queryFn: listQuestions })

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
            <p className="text-xs text-amber-500 mt-1">준비 탭에서 문서를 벡터화해야 검색이 가능합니다</p>
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
              <p className="text-sm text-amber-700">준비 탭에서 테스트 질문을 먼저 등록해주세요.</p>
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
                <div className="space-y-5">
                  {/* Aggregate summary table */}
                  <div className="bg-white rounded-xl border border-gray-200 p-5">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">모델별 검색 성능 요약</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-gray-100 text-xs text-gray-400">
                            <th className="text-left py-2 pr-4 font-medium">모델</th>
                            <th className="text-center px-4 py-2 font-medium">평균 Top-1 유사도</th>
                            <th className="text-center px-4 py-2 font-medium">평균 전체 유사도</th>
                            <th className="text-center px-4 py-2 font-medium">결과 있는 질문</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchModelIds
                            .sort((a, b) => (batchAggregates[b]?.avg_top1_score ?? 0) - (batchAggregates[a]?.avg_top1_score ?? 0))
                            .map((eid, rank) => {
                              const agg = batchAggregates[eid]
                              return (
                                <tr key={eid} className="border-b border-gray-50 hover:bg-gray-50">
                                  <td className="py-2.5 pr-4">
                                    <div className="flex items-center gap-2">
                                      <span className="text-xs font-bold text-gray-400 w-4">{rank + 1}</span>
                                      <EmbeddingBadge provider={eid.split(':')[0] as 'local' | 'openai' | 'ollama'} />
                                      <span className="text-sm font-medium text-gray-700">{shortId(eid)}</span>
                                    </div>
                                  </td>
                                  <td className="px-4 py-2.5 text-center">
                                    <div className="flex items-center gap-2 justify-center">
                                      <div className="w-20 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                        <div
                                          className={`h-full rounded-full ${
                                            agg.avg_top1_score >= 0.7 ? 'bg-emerald-400' :
                                            agg.avg_top1_score >= 0.45 ? 'bg-amber-400' : 'bg-red-400'
                                          }`}
                                          style={{ width: `${agg.avg_top1_score * 100}%` }}
                                        />
                                      </div>
                                      <span className={`text-sm font-bold tabular-nums ${scoreColor(agg.avg_top1_score)}`}>
                                        {agg.avg_top1_score.toFixed(3)}
                                      </span>
                                    </div>
                                  </td>
                                  <td className="px-4 py-2.5 text-center">
                                    <span className={`text-sm font-bold tabular-nums ${scoreColor(agg.avg_score)}`}>
                                      {agg.avg_score.toFixed(3)}
                                    </span>
                                  </td>
                                  <td className="px-4 py-2.5 text-center">
                                    <span className="text-sm text-gray-700">
                                      {agg.hit_queries} / {agg.total_queries}
                                    </span>
                                    <span className={`ml-1.5 text-xs font-medium ${scoreColor(agg.hit_rate)}`}>
                                      ({(agg.hit_rate * 100).toFixed(0)}%)
                                    </span>
                                  </td>
                                </tr>
                              )
                            })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Per-question breakdown */}
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
                            {batchModelIds.map((eid) => (
                              <th key={eid} className="text-center px-3 py-2 text-gray-500 font-medium whitespace-nowrap">
                                {shortId(eid)}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {questions.map((q: TestQuestion) => {
                            const isExpanded = expandedQuery === q.question
                            return (
                              <>
                                <tr
                                  key={q.id}
                                  onClick={() => setExpandedQuery(isExpanded ? null : q.question)}
                                  className="border-b border-gray-50 hover:bg-gray-50 cursor-pointer"
                                >
                                  <td className="py-2.5 px-5">
                                    <div className="flex items-center gap-2">
                                      <span className={`text-gray-300 transition-transform ${isExpanded ? 'rotate-90' : ''}`}>▶</span>
                                      <span className="text-xs text-gray-400 shrink-0">[{q.category}]</span>
                                      <span className="text-xs text-gray-700 line-clamp-1">{q.question}</span>
                                    </div>
                                  </td>
                                  {batchModelIds.map((eid) => {
                                    const chunks = batchResults[q.question]?.[eid] ?? []
                                    const top1 = chunks[0]?.score ?? null
                                    return (
                                      <td key={eid} className="px-3 py-2.5 text-center">
                                        {top1 !== null ? (
                                          <span className={`inline-block px-2 py-0.5 rounded border font-bold tabular-nums ${scoreColor(top1)}`}>
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
                                    <td colSpan={batchModelIds.length + 1} className="px-5 py-3">
                                      <div
                                        className="grid gap-3"
                                        style={{ gridTemplateColumns: `repeat(${Math.min(batchModelIds.length, 3)}, 1fr)` }}
                                      >
                                        {batchModelIds.map((eid) => {
                                          const chunks = batchResults[q.question]?.[eid] ?? []
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
                                                  {chunks.slice(0, 3).map((c, i) => (
                                                    <div key={i} className="text-xs">
                                                      <div className="flex items-center gap-1.5 mb-0.5">
                                                        <span className="text-gray-400">#{i + 1}</span>
                                                        <span className={`font-bold ${scoreColor(c.score)}`}>{c.score.toFixed(3)}</span>
                                                      </div>
                                                      <p className="text-gray-600 line-clamp-2 leading-relaxed">{c.text}</p>
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
  const [judgeModel, setJudgeModel] = useState('claude-haiku-4-5-20251001')
  const [runName, setRunName] = useState('')

  const [runId, setRunId] = useState<string | null>(null)
  const [runPhase, setRunPhase] = useState<'config' | 'running' | 'done'>('config')
  const [liveResults, setLiveResults] = useState<TestResult[]>([])
  const [totalTests, setTotalTests] = useState(0)

  const { data: embModels = [] } = useQuery({ queryKey: ['embeddingModels'], queryFn: listEmbeddingModels })
  const { data: llmModels = [] } = useQuery({ queryKey: ['llmModels'], queryFn: listLLMModels })
  const { data: questions = [] } = useQuery({ queryKey: ['questions'], queryFn: listQuestions })
  const { data: docs = [] } = useQuery({ queryKey: ['documents'], queryFn: listDocuments })

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
              <select value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white">
                {llmModels.filter((m: LLMModelInfo) => m.available).map((m: LLMModelInfo) => {
                  const tier = JUDGE_QUALITY[m.id]
                  const suffix = tier === 'high' ? ' ★' : tier === 'medium' ? '' : ' (정확도 낮을 수 있음)'
                  return <option key={m.id} value={m.id}>{m.name}{suffix}</option>
                })}
              </select>
              {(() => {
                const tier = JUDGE_QUALITY[judgeModel]
                if (tier === 'high' || tier === 'medium') return null
                return (
                  <p className="text-xs text-amber-600 mt-1">
                    소형 모델은 JSON 출력이 불안정할 수 있어 일부 평가가 null로 기록될 수 있습니다.
                  </p>
                )
              })()}
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
  const [activeTab, setActiveTab] = useState<Tab>('ocr')
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
      {activeTab === 'ocr'       && <OcrTab docs={docs as DocumentInfo[]} />}
      {activeTab === 'embedding' && <EmbeddingTab docs={docs as DocumentInfo[]} />}
      {activeTab === 'llm'       && <LlmTab />}
    </div>
  )
}
