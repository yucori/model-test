/**
 * PipelinePage — 4-stage RAG pipeline analysis
 *
 * Stage 1: 파싱  — compare parsers, pick best
 * Stage 2: 청킹  — compare chunking strategies, pick best
 * Stage 3: 검색  — vectorize + compare semantic/BM25/hybrid
 * Stage 4: 생성  — compare LLMs with locked pipeline, see evidence
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  uploadDocument, listDocuments,
  compareParsersFn, compareChunks, compareSearch,
  listEmbeddingModels, listLLMModels, listQuestions,
  vectorizeDocuments, createRun, getRunResults,
} from '../lib/api'
import type { ParserCompareResult } from '../lib/api'
import type {
  DocumentInfo,
  TestResult, RetrievedChunk,
  ChunkVariantStats, SearchQueryResult,
} from '../types'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { formatBytes } from '../lib/utils'

// ─────────────────────────────────────────────────────────────────────────────
// Types & constants
// ─────────────────────────────────────────────────────────────────────────────

type StageId = 1 | 2 | 3 | 4

interface PipelineLock {
  docId: string
  filename: string
  parser: string
  chunkStrategy: string
  chunkSize: number
  overlap: number
  chunkLabel: string
  embModelId: string
  searchStrategy: string
  runId: string | null
}

const STAGES: { id: StageId; label: string; sub: string; icon: string }[] = [
  { id: 1, label: '파싱',   sub: '파서 선택',       icon: '📄' },
  { id: 2, label: '청킹',   sub: '청킹 전략 선택',  icon: '✂️' },
  { id: 3, label: '검색',   sub: '검색 전략 선택',  icon: '🔍' },
  { id: 4, label: '생성',   sub: 'LLM 비교',        icon: '🤖' },
]

const CHUNK_PRESETS = [
  { strategy: 'structure',   chunk_size: 800,  overlap: 0,   label: '구조단위-800' },
  { strategy: 'structure',   chunk_size: 1200, overlap: 0,   label: '구조단위-1200' },
  { strategy: 'paragraph',   chunk_size: 400,  overlap: 50,  label: '문단-400' },
  { strategy: 'paragraph',   chunk_size: 600,  overlap: 100, label: '문단-600' },
  { strategy: 'paragraph',   chunk_size: 1000, overlap: 150, label: '문단-1000' },
  { strategy: 'sentence',    chunk_size: 300,  overlap: 0,   label: '문장-300' },
  { strategy: 'sentence',    chunk_size: 500,  overlap: 0,   label: '문장-500' },
  { strategy: 'fixed',       chunk_size: 256,  overlap: 50,  label: '고정-256' },
  { strategy: 'fixed',       chunk_size: 512,  overlap: 100, label: '고정-512' },
  { strategy: 'semantic',    chunk_size: 600,  overlap: 0,   label: '시멘틱-600' },
  { strategy: 'parent_child', chunk_size: 300, overlap: 0,  label: 'ParentChild-300' },
]

const SEARCH_STRATEGIES = [
  { id: 'semantic',    label: 'Semantic',    desc: '벡터 유사도 검색. 의미 기반.' },
  { id: 'bm25',        label: 'BM25',        desc: '키워드 검색. 정확한 단어 매칭에 강함.' },
  { id: 'hybrid_rrf',  label: 'Hybrid RRF',  desc: 'BM25 + Semantic 융합 (Reciprocal Rank Fusion).' },
]

function shortId(id: string | null | undefined) {
  if (!id) return '—'
  const noPrefix = id.replace(/^(local|openai|ollama|hf):/, '')
  const name = noPrefix.includes('/') ? noPrefix.split('/').pop()! : noPrefix.split(':')[0]
  return name
    .replace('all-MiniLM-L6-v2', 'MiniLM')
    .replace('nomic-embed-text-v1.5', 'nomic-v1.5')
    .replace('ko-sroberta-multitask', 'ko-sroberta')
    .replace('jina-embeddings-v4', 'jina-v4')
    .replace('qwen3-embedding-4b', 'qwen3-4b')
    .replace('embeddinggemma-300m', 'gemma-300m')
    .replace('multilingual-e5-small', 'me5-small')
    .replace('-20251001', '')
}

function scoreColor(v: number, max = 10) {
  const pct = v / max
  if (pct >= 0.75) return 'text-emerald-600 bg-emerald-50'
  if (pct >= 0.5)  return 'text-amber-600 bg-amber-50'
  return 'text-red-500 bg-red-50'
}

// ─────────────────────────────────────────────────────────────────────────────
// Stepper rail
// ─────────────────────────────────────────────────────────────────────────────

function StepperRail({
  active, maxReached, lock,
  onSelect,
}: {
  active: StageId
  maxReached: StageId
  lock: Partial<PipelineLock>
  onSelect: (id: StageId) => void
}) {
  return (
    <div className="bg-white border-b border-slate-200 sticky top-12 z-10 shadow-sm">
      <div className="max-w-6xl mx-auto px-6 py-3">
        <div className="flex items-center gap-2">
          {STAGES.map((stage, i) => {
            const done = stage.id < maxReached || (stage.id < active)
            const current = stage.id === active
            const locked = stage.id > maxReached
            return (
              <div key={stage.id} className="flex items-center flex-1 last:flex-none">
                <button
                  disabled={locked}
                  onClick={() => !locked && onSelect(stage.id)}
                  className={`flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg transition-all w-full
                    ${current ? 'bg-indigo-50 ring-2 ring-indigo-400' : ''}
                    ${done ? 'cursor-pointer hover:bg-slate-50' : ''}
                    ${locked ? 'opacity-40 cursor-not-allowed' : ''}`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ring-2
                    ${done    ? 'bg-emerald-500 text-white ring-emerald-300' :
                      current ? 'bg-indigo-600 text-white ring-indigo-300' :
                                'bg-slate-100 text-slate-400 ring-slate-200'}`}>
                    {done ? '✓' : stage.icon}
                  </div>
                  <span className={`text-xs font-semibold ${current ? 'text-indigo-700' : done ? 'text-emerald-700' : 'text-slate-400'}`}>
                    {stage.label}
                  </span>
                </button>
                {i < STAGES.length - 1 && (
                  <div className={`h-0.5 flex-1 mx-1 rounded transition-colors
                    ${stage.id < maxReached ? 'bg-emerald-400' : 'bg-slate-200'}`} />
                )}
              </div>
            )
          })}
        </div>
        {/* Lock summary */}
        {maxReached > 1 && (
          <div className="flex gap-4 mt-2 text-xs text-slate-500 flex-wrap">
            {lock.filename && <span>📄 <b>{lock.filename}</b></span>}
            {lock.parser && <span>파서: <b>{lock.parser}</b></span>}
            {lock.chunkLabel && <span>청킹: <b>{lock.chunkLabel}</b></span>}
            {lock.embModelId && <span>임베딩: <b>{shortId(lock.embModelId)}</b></span>}
            {lock.searchStrategy && <span>검색: <b>{{semantic:'Semantic',bm25:'BM25',hybrid_rrf:'Hybrid RRF'}[lock.searchStrategy] ?? lock.searchStrategy}</b></span>}
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — Parser comparison
// ─────────────────────────────────────────────────────────────────────────────

function Stage1Parser({
  onConfirm,
}: {
  onConfirm: (docId: string, filename: string, parser: string) => void
}) {
  const qc = useQueryClient()
  const [selectedDoc, setSelectedDoc] = useState<DocumentInfo | null>(null)
  const [compareResult, setCompareResult] = useState<any>(null)
  const [selectedParser, setSelectedParser] = useState<string | null>(null)
  const [previewParser, setPreviewParser] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const { data: docs = [] } = useQuery({ queryKey: ['documents'], queryFn: listDocuments })

  const uploadMut = useMutation({
    mutationFn: uploadDocument,
    onSuccess: (doc) => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      setSelectedDoc(doc)
    },
  })

  const compareMut = useMutation({
    mutationFn: ({ docId }: { docId: string }) => compareParsersFn(docId),
    onSuccess: (data) => {
      setCompareResult(data)
      if (data.recommendation) {
        setSelectedParser(data.recommendation.parser_id)
        setPreviewParser(data.recommendation.parser_id)
      }
    },
  })

  const handleFile = useCallback((file: File) => {
    uploadMut.mutate(file)
  }, [])

  useEffect(() => {
    if (selectedDoc) compareMut.mutate({ docId: selectedDoc.id })
  }, [selectedDoc?.id])

  const parserResult = (pid: string): ParserCompareResult | undefined =>
    compareResult?.results?.find((r: ParserCompareResult) => r.parser_id === pid)

  const currentPreview = previewParser ? parserResult(previewParser) : null

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-800">Stage 1 — 파싱 전략 비교</h2>
        <p className="text-sm text-slate-500 mt-1">
          문서를 업로드하고, 파서별로 추출 품질을 비교하여 최적 파서를 선택합니다.
        </p>
      </div>

      {/* Upload + doc list */}
      <div className="grid grid-cols-2 gap-4">
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault(); setDragOver(false)
            const f = e.dataTransfer.files[0]
            if (f) handleFile(f)
          }}
          onClick={() => fileRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors
            ${dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-slate-300 hover:border-indigo-300 hover:bg-slate-50'}`}
        >
          <input ref={fileRef} type="file" className="hidden" accept=".pdf,.docx,.doc"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
          {uploadMut.isPending ? (
            <LoadingSpinner size="sm" />
          ) : (
            <>
              <div className="text-3xl mb-2">📂</div>
              <p className="text-sm font-medium text-slate-600">PDF / DOCX 드래그 또는 클릭</p>
            </>
          )}
        </div>

        {/* Existing docs */}
        <div className="border rounded-xl overflow-hidden">
          <div className="bg-slate-50 px-4 py-2 text-xs font-semibold text-slate-500 border-b">기존 문서</div>
          <div className="max-h-48 overflow-y-auto divide-y">
            {docs.length === 0 && (
              <p className="text-sm text-slate-400 p-4 text-center">업로드된 문서 없음</p>
            )}
            {docs.map((doc) => (
              <button
                key={doc.id}
                onClick={() => setSelectedDoc(doc)}
                className={`w-full text-left px-4 py-2.5 text-sm hover:bg-indigo-50 transition-colors
                  ${selectedDoc?.id === doc.id ? 'bg-indigo-50 font-semibold text-indigo-700' : 'text-slate-700'}`}
              >
                <span className="truncate block">{doc.filename}</span>
                <span className="text-xs text-slate-400">{formatBytes(doc.size_bytes)}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Parser comparison table */}
      {compareMut.isPending && (
        <div className="flex items-center gap-3 py-8 justify-center">
          <LoadingSpinner size="sm" />
          <span className="text-slate-500">파서 비교 중...</span>
        </div>
      )}

      {compareResult && !compareMut.isPending && (
        <>
          {/* Recommendation banner */}
          {compareResult.recommendation && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-5 py-3 flex items-center gap-3">
              <span className="text-emerald-600 text-lg">✓</span>
              <div>
                <p className="font-semibold text-emerald-800">
                  추천: <span className="capitalize">{compareResult.recommendation.parser_id}</span>
                </p>
                <p className="text-xs text-emerald-600">{compareResult.recommendation.reasons.join(' · ')}</p>
              </div>
            </div>
          )}

          {/* Parser cards */}
          <div className="grid grid-cols-3 gap-4">
            {compareResult.results.map((r: ParserCompareResult) => (
              <div
                key={r.parser_id}
                className={`rounded-xl border-2 p-4 cursor-pointer transition-all
                  ${r.error ? 'opacity-50 cursor-not-allowed border-slate-200' :
                    selectedParser === r.parser_id ? 'border-indigo-500 bg-indigo-50 shadow-md' :
                    'border-slate-200 hover:border-indigo-300'}`}
                onClick={() => { if (!r.error) { setSelectedParser(r.parser_id); setPreviewParser(r.parser_id) } }}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="font-bold text-slate-800 capitalize">{r.parser_id}</span>
                  {compareResult.recommendation?.parser_id === r.parser_id && (
                    <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-semibold">추천</span>
                  )}
                  {selectedParser === r.parser_id && !compareResult.recommendation && (
                    <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-semibold">선택됨</span>
                  )}
                </div>
                {r.error ? (
                  <p className="text-xs text-red-500">{r.error}</p>
                ) : (
                  <div className="space-y-1 text-xs text-slate-600">
                    <div className="flex justify-between"><span>한글 단어</span><b>{r.korean_word_count.toLocaleString()}</b></div>
                    <div className="flex justify-between"><span>총 문자</span><b>{r.clean_char_count.toLocaleString()}</b></div>
                    <div className="flex justify-between"><span>청크 수</span><b>{r.chunk_count}</b></div>
                    <div className="flex justify-between"><span>구조 점수</span><b>{(r.structure_score * 100).toFixed(0)}%</b></div>
                    <div className="flex justify-between"><span>속도</span><b>{r.elapsed_ms}ms</b></div>
                    {r.score !== undefined && (
                      <div className="flex justify-between pt-1 border-t mt-1">
                        <span className="font-semibold">종합 점수</span>
                        <b className="text-indigo-700">{(r.score * 100).toFixed(1)}점</b>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Text preview tabs */}
          {currentPreview && !currentPreview.error && (
            <div className="border rounded-xl overflow-hidden">
              <div className="flex border-b bg-slate-50">
                {compareResult.results
                  .filter((r: ParserCompareResult) => !r.error && !r.unavailable)
                  .map((r: ParserCompareResult) => (
                    <button
                      key={r.parser_id}
                      onClick={() => setPreviewParser(r.parser_id)}
                      className={`px-4 py-2 text-sm font-medium border-r capitalize
                        ${previewParser === r.parser_id ? 'bg-white text-indigo-700 border-b-white' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                      {r.parser_id}
                    </button>
                  ))}
              </div>
              <pre className="text-xs p-4 overflow-auto max-h-64 whitespace-pre-wrap font-mono text-slate-700 bg-white leading-relaxed">
                {currentPreview.text_preview}
              </pre>
            </div>
          )}

          <div className="flex justify-end">
            <button
              disabled={!selectedParser}
              onClick={() => {
                if (selectedParser && selectedDoc)
                  onConfirm(selectedDoc.id, selectedDoc.filename, selectedParser)
              }}
              className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-semibold disabled:opacity-40 hover:bg-indigo-700 transition"
            >
              이 파서로 계속 →
            </button>
          </div>
        </>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — Chunking comparison
// ─────────────────────────────────────────────────────────────────────────────

function Stage2Chunk({
  docId, parser,
  onConfirm,
}: {
  docId: string
  parser: string
  onConfirm: (strategy: string, chunkSize: number, overlap: number, label: string) => void
}) {
  const [result, setResult] = useState<any>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [expandedSample, setExpandedSample] = useState<string | null>(null)

  const compareMut = useMutation({
    mutationFn: () => compareChunks({
      doc_id: docId,
      parser,
      variants: CHUNK_PRESETS,
    }),
    onSuccess: (data) => setResult(data),
  })

  useEffect(() => { compareMut.mutate() }, [docId, parser])

  const selectedVariant = result?.variants.find((v: ChunkVariantStats) => v.config.label === selected)

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-800">Stage 2 — 청킹 전략 비교</h2>
        <p className="text-sm text-slate-500 mt-1">
          같은 문서를 여러 청킹 전략으로 분할하여 청크 수, 크기 분포, 샘플을 비교합니다.
        </p>
      </div>

      {compareMut.isPending && (
        <div className="flex items-center gap-3 py-12 justify-center">
          <LoadingSpinner size="sm" />
          <span className="text-slate-500">청킹 전략 분석 중...</span>
        </div>
      )}

      {result && !compareMut.isPending && (
        <>
          {/* Grid of variant cards */}
          <div className="grid grid-cols-3 gap-4">
            {result.variants.map((v: ChunkVariantStats) => {
              const totalChunks = v.chunk_count || 1
              const buckets = v.size_buckets
              const isSelected = selected === v.config.label
              return (
                <div
                  key={v.config.label}
                  onClick={() => setSelected(isSelected ? null : v.config.label)}
                  className={`rounded-xl border-2 p-4 cursor-pointer transition-all
                    ${isSelected ? 'border-indigo-500 bg-indigo-50 shadow-md' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-slate-800">{v.config.label}</span>
                    {isSelected && <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">선택됨</span>}
                  </div>
                  <div className="space-y-1 text-xs text-slate-600 mb-3">
                    <div className="flex justify-between"><span>청크 수</span><b>{v.chunk_count}</b></div>
                    <div className="flex justify-between"><span>평균 크기</span><b>{v.avg_size}자</b></div>
                    <div className="flex justify-between"><span>중간값</span><b>{v.median_size}자</b></div>
                    <div className="flex justify-between"><span>범위</span><b>{v.min_size}–{v.max_size}자</b></div>
                    <div className="flex justify-between items-center pt-1 border-t border-slate-100">
                      <span>구조 경계 시작</span>
                      <span className={`font-bold ${v.structure_aligned_count > 0 ? 'text-emerald-600' : 'text-slate-400'}`}>
                        {v.structure_aligned_count}/{v.chunk_count}
                        {v.chunk_count > 0 && (
                          <span className="font-normal text-slate-400 ml-1">
                            ({Math.round(v.structure_aligned_count / v.chunk_count * 100)}%)
                          </span>
                        )}
                      </span>
                    </div>
                  </div>
                  {/* Size distribution mini-bar */}
                  <div className="space-y-1">
                    <p className="text-xs text-slate-400 font-medium">크기 분포</p>
                    {Object.entries(buckets).map(([label, count]) => (
                      <div key={label} className="flex items-center gap-1.5 text-xs">
                        <span className="w-16 text-slate-500 shrink-0">{label}</span>
                        <div className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                          <div
                            className="h-full bg-indigo-400 rounded-full transition-all"
                            style={{ width: `${Math.round(((count as number) / totalChunks) * 100)}%` }}
                          />
                        </div>
                        <span className="w-8 text-right text-slate-500">{count as number}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Sample chunks for selected variant */}
          {selectedVariant && (
            <div className="border rounded-xl overflow-hidden">
              <div className="bg-slate-50 px-4 py-2 border-b flex items-center justify-between">
                <span className="text-sm font-semibold text-slate-700">
                  샘플 청크 — {selectedVariant.config.label}
                </span>
                <span className="text-xs text-slate-400">(처음 5개)</span>
              </div>
              <div className="divide-y">
                {selectedVariant.sample_chunks.map((chunk: string, i: number) => {
                  const isStructured = /^#{2,3} /.test(chunk.trimStart())
                  return (
                  <div key={i} className={`p-4 ${isStructured ? 'bg-emerald-50' : ''}`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-400">청크 #{i + 1}</span>
                        {isStructured && (
                          <span className="text-xs bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded font-medium">
                            구조 경계
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-slate-400">{chunk.length}자</span>
                    </div>
                    <div
                      className={`text-xs text-slate-700 leading-relaxed whitespace-pre-wrap cursor-pointer
                        ${expandedSample !== `${selectedVariant.config.label}-${i}` ? 'line-clamp-4' : ''}`}
                      onClick={() => setExpandedSample(
                        expandedSample === `${selectedVariant.config.label}-${i}` ? null : `${selectedVariant.config.label}-${i}`
                      )}
                    >
                      {chunk}
                    </div>
                  </div>
                  )
                })}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-500">
              카드를 클릭해 선택 → 오른쪽 버튼으로 확정
            </p>
            <button
              disabled={!selected}
              onClick={() => {
                const v = result.variants.find((v: ChunkVariantStats) => v.config.label === selected)
                if (v) onConfirm(v.config.strategy, v.config.chunk_size, v.config.overlap, v.config.label)
              }}
              className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-semibold disabled:opacity-40 hover:bg-indigo-700 transition"
            >
              이 청킹 전략으로 계속 →
            </button>
          </div>
        </>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 3 — Search strategy comparison
// ─────────────────────────────────────────────────────────────────────────────

function Stage3Search({
  docId, parser, chunkStrategy, chunkSize, overlap,
  onConfirm,
}: {
  docId: string
  parser: string
  chunkStrategy: string
  chunkSize: number
  overlap: number
  onConfirm: (embModelId: string, searchStrategy: string) => void
}) {
  const qc = useQueryClient()
  const [selectedEmb, setSelectedEmb] = useState<string | null>(null)
  const [vectorized, setVectorized] = useState(false)
  const [vectorizing, setVectorizing] = useState(false)
  const [queries, setQueries] = useState<string[]>([''])
  const [searchResult, setSearchResult] = useState<any>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<string>('semantic')
  const [activeQueryIdx, setActiveQueryIdx] = useState(0)

  const { data: embModels = [] } = useQuery({ queryKey: ['embedding-models'], queryFn: listEmbeddingModels })
  const { data: questions = [] } = useQuery({ queryKey: ['questions'], queryFn: () => listQuestions() })

  const searchMut = useMutation({
    mutationFn: () => compareSearch({
      doc_id: docId,
      emb_model_id: selectedEmb!,
      queries: queries.filter(q => q.trim()),
      strategies: ['semantic', 'bm25', 'hybrid_rrf'],
      top_k: 5,
    }),
    onSuccess: setSearchResult,
  })

  const handleVectorize = async () => {
    if (!selectedEmb) return
    setVectorizing(true)
    try {
      await vectorizeDocuments([selectedEmb], [docId], parser, chunkSize, overlap, chunkStrategy)
      // Poll until done
      for (let i = 0; i < 60; i++) {
        await new Promise(r => setTimeout(r, 2000))
        const docs: any[] = await qc.fetchQuery({ queryKey: ['documents'], queryFn: () => import('../lib/api').then(m => m.listDocuments()) })
        const doc = docs.find(d => d.id === docId)
        if (doc && !doc.processing) break
      }
      setVectorized(true)
    } finally {
      setVectorizing(false)
    }
  }

  const addQuery = () => setQueries([...queries, ''])
  const removeQuery = (i: number) => setQueries(queries.filter((_, j) => j !== i))
  const setQuery = (i: number, v: string) => setQueries(queries.map((q, j) => j === i ? v : q))
  const addFromSuite = (q: string) => { if (!queries.includes(q)) setQueries([...queries.filter(x => x.trim()), q]) }

  // Group search results by (strategy, query)
  const resultMap: Record<string, Record<string, SearchQueryResult>> = {}
  if (searchResult) {
    for (const r of searchResult.query_results) {
      if (!resultMap[r.query]) resultMap[r.query] = {}
      resultMap[r.query][r.strategy] = r
    }
  }
  const queriesTested = searchResult?.queries_tested ?? []
  const currentQuery = queriesTested[activeQueryIdx] ?? ''

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-800">Stage 3 — 검색 전략 비교</h2>
        <p className="text-sm text-slate-500 mt-1">
          임베딩 모델로 벡터화 후, Semantic · BM25 · Hybrid RRF 검색을 테스트 쿼리로 비교합니다.
        </p>
      </div>

      {/* Step A: Select embedding model + vectorize */}
      <div className="bg-slate-50 rounded-xl p-5 space-y-4 border">
        <h3 className="font-semibold text-slate-700">A. 임베딩 모델 선택 + 벡터화</h3>
        <div className="grid grid-cols-2 gap-3">
          {embModels.filter(m => m.available).map(m => (
            <button
              key={m.id}
              onClick={() => setSelectedEmb(m.id)}
              className={`text-left p-3 rounded-lg border-2 transition-all
                ${selectedEmb === m.id ? 'border-indigo-500 bg-white shadow' : 'border-slate-200 hover:border-indigo-300'}`}
            >
              <div className="font-semibold text-sm">{m.name}</div>
              <div className="text-xs text-slate-500 mt-0.5">{m.dimensions}d · {m.description}</div>
            </button>
          ))}
        </div>
        <button
          disabled={!selectedEmb || vectorizing || vectorized}
          onClick={handleVectorize}
          className="px-5 py-2 bg-slate-800 text-white rounded-lg font-semibold disabled:opacity-40 hover:bg-slate-900 transition text-sm"
        >
          {vectorizing ? '벡터화 중...' : vectorized ? '✓ 벡터화 완료' : '벡터화 시작'}
        </button>
      </div>

      {/* Step B: Test queries */}
      <div className="bg-slate-50 rounded-xl p-5 space-y-4 border">
        <h3 className="font-semibold text-slate-700">B. 테스트 쿼리 설정</h3>
        <div className="space-y-2">
          {queries.map((q, i) => (
            <div key={i} className="flex gap-2">
              <input
                value={q}
                onChange={(e) => setQuery(i, e.target.value)}
                placeholder={`쿼리 ${i + 1}`}
                className="flex-1 border rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
              />
              {queries.length > 1 && (
                <button onClick={() => removeQuery(i)} className="text-red-400 hover:text-red-600 text-sm px-2">×</button>
              )}
            </div>
          ))}
          <button onClick={addQuery} className="text-xs text-indigo-600 hover:underline">+ 쿼리 추가</button>
        </div>
        {questions.length > 0 && (
          <div>
            <p className="text-xs text-slate-500 mb-2">테스트 질문에서 불러오기:</p>
            <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
              {questions.slice(0, 10).map(q => (
                <button
                  key={q.id}
                  onClick={() => addFromSuite(q.question)}
                  className="text-xs bg-white border border-slate-200 rounded-full px-3 py-1 hover:bg-indigo-50 hover:border-indigo-300 truncate max-w-xs"
                >
                  {q.question.length > 40 ? q.question.slice(0, 40) + '…' : q.question}
                </button>
              ))}
            </div>
          </div>
        )}
        <button
          disabled={!vectorized || !queries.some(q => q.trim())}
          onClick={() => searchMut.mutate()}
          className="px-5 py-2 bg-indigo-600 text-white rounded-lg font-semibold disabled:opacity-40 hover:bg-indigo-700 transition text-sm"
        >
          {searchMut.isPending ? '검색 중...' : '검색 전략 비교 실행'}
        </button>
      </div>

      {/* Results: per-query, 3-column strategy comparison */}
      {searchResult && !searchMut.isPending && (
        <>
          {/* Query tabs */}
          {queriesTested.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {queriesTested.map((q: string, i: number) => (
                <button
                  key={i}
                  onClick={() => setActiveQueryIdx(i)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border
                    ${activeQueryIdx === i ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-300'}`}
                >
                  {q.length > 30 ? q.slice(0, 30) + '…' : q}
                </button>
              ))}
            </div>
          )}

          {currentQuery && (
            <div>
              <p className="text-sm font-semibold text-slate-700 mb-3">
                쿼리: <span className="text-indigo-700">"{currentQuery}"</span>
              </p>
              <div className="grid grid-cols-3 gap-4">
                {SEARCH_STRATEGIES.map(s => {
                  const r: SearchQueryResult | undefined = resultMap[currentQuery]?.[s.id]
                  const isSelected = selectedStrategy === s.id
                  return (
                    <div
                      key={s.id}
                      onClick={() => setSelectedStrategy(s.id)}
                      className={`rounded-xl border-2 cursor-pointer transition-all
                        ${isSelected ? 'border-indigo-500 shadow-md' : 'border-slate-200 hover:border-indigo-300'}`}
                    >
                      <div className={`px-4 py-3 border-b flex items-center justify-between
                        ${isSelected ? 'bg-indigo-50' : 'bg-slate-50'}`}>
                        <div>
                          <span className="font-bold text-sm">{s.label}</span>
                          <p className="text-xs text-slate-500 mt-0.5">{s.desc}</p>
                        </div>
                        {r && (
                          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                            r.avg_score > 0.7 ? 'bg-emerald-100 text-emerald-700' :
                            r.avg_score > 0.3 ? 'bg-amber-100 text-amber-700' :
                            'bg-slate-100 text-slate-600'}`}>
                            avg {(r.avg_score * (r.avg_score > 1 ? 1 : 1)).toFixed(3)}
                          </span>
                        )}
                      </div>
                      <div className="divide-y max-h-80 overflow-y-auto">
                        {!r || r.chunks.length === 0 ? (
                          <p className="text-xs text-slate-400 p-4 text-center">결과 없음</p>
                        ) : r.chunks.map((chunk, ci) => (
                          <div key={ci} className="px-4 py-3">
                            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                              <span className="text-xs font-bold text-slate-400">#{ci + 1}</span>
                              {chunk.semantic_score != null && (
                                <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                                  chunk.semantic_score > 0.7 ? 'bg-emerald-100 text-emerald-700' :
                                  chunk.semantic_score > 0.4 ? 'bg-amber-100 text-amber-700' :
                                  'bg-red-50 text-red-600'}`}>
                                  sem {chunk.semantic_score.toFixed(3)}
                                </span>
                              )}
                              {chunk.bm25_score != null && (
                                <span className="text-xs px-1.5 py-0.5 rounded font-mono bg-blue-50 text-blue-700">
                                  bm25 {chunk.bm25_score.toFixed(2)}
                                </span>
                              )}
                              {s.id === 'hybrid_rrf' && (
                                <span className="text-xs px-1.5 py-0.5 rounded font-mono bg-purple-50 text-purple-700">
                                  rrf {chunk.final_score.toFixed(4)}
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-slate-700 line-clamp-4 leading-relaxed">
                              {chunk.content}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <p className="text-sm text-slate-500">전략을 클릭해 선택</p>
            <button
              disabled={!selectedEmb}
              onClick={() => onConfirm(selectedEmb!, selectedStrategy)}
              className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-semibold disabled:opacity-40 hover:bg-indigo-700 transition"
            >
              이 검색 전략으로 계속 →
            </button>
          </div>
        </>
      )}

      {/* Allow confirm even without comparison */}
      {!searchResult && vectorized && (
        <div className="flex justify-end pt-2">
          <button
            onClick={() => onConfirm(selectedEmb!, selectedStrategy)}
            className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 transition"
          >
            이 검색 전략으로 계속 →
          </button>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 4 — LLM generation comparison
// ─────────────────────────────────────────────────────────────────────────────

function RetrievedChunkBadge({ chunk, index }: { chunk: RetrievedChunk; index: number }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="border rounded-lg bg-slate-50 overflow-hidden text-xs">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-100 transition-colors"
      >
        <span className="font-bold text-slate-400 shrink-0">#{index + 1}</span>
        <span className="flex-1 truncate text-slate-600">{chunk.content.slice(0, 80)}…</span>
        <div className="flex gap-1 shrink-0">
          {chunk.semantic_score != null && (
            <span className={`px-1.5 py-0.5 rounded font-mono ${
              chunk.semantic_score > 0.7 ? 'bg-emerald-100 text-emerald-700' :
              chunk.semantic_score > 0.4 ? 'bg-amber-100 text-amber-700' :
              'bg-red-50 text-red-600'}`}>
              {chunk.semantic_score.toFixed(3)}
            </span>
          )}
          {chunk.bm25_score != null && (
            <span className="px-1.5 py-0.5 rounded font-mono bg-blue-50 text-blue-700">
              bm25
            </span>
          )}
        </div>
        <span className="text-slate-300">{expanded ? '▲' : '▼'}</span>
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t whitespace-pre-wrap text-slate-700 leading-relaxed">
          {chunk.content}
        </div>
      )}
    </div>
  )
}

function ResultCard({ result }: { result: TestResult }) {
  const [showReasoning, setShowReasoning] = useState(false)
  const [showChunks, setShowChunks] = useState(false)
  const scores = result.scores

  return (
    <div className="border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="bg-slate-50 px-4 py-2.5 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-slate-500">LLM</span>
          <span className="text-sm font-semibold text-slate-800">{shortId(result.llm_model_id)}</span>
          {result.embedding_model_id && (
            <>
              <span className="text-xs text-slate-300">·</span>
              <span className="text-xs text-slate-500">임베딩: {shortId(result.embedding_model_id)}</span>
            </>
          )}
        </div>
        <div className="flex gap-1.5">
          {scores && (['relevance','accuracy','helpfulness','korean_fluency','source_citation','overall'] as const).map(k => (
            <span key={k} className={`text-xs px-1.5 py-0.5 rounded font-bold ${scoreColor(scores[k])}`}>
              {k === 'overall' ? `★ ${scores[k].toFixed(1)}` : scores[k].toFixed(1)}
            </span>
          ))}
        </div>
      </div>

      {/* Response */}
      {result.error ? (
        <div className="px-4 py-3 text-sm text-red-600 bg-red-50">{result.error}</div>
      ) : (
        <div className="px-4 py-3 text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">
          {result.response}
        </div>
      )}

      {/* Retrieved chunks */}
      {result.retrieved_chunks.length > 0 && (
        <div className="border-t px-4 py-3">
          <button
            onClick={() => setShowChunks(!showChunks)}
            className="flex items-center gap-2 text-xs font-semibold text-slate-600 hover:text-indigo-700 mb-2"
          >
            <span>🔍 검색된 청크 ({result.retrieved_chunks.length}개)</span>
            <span className="text-slate-300">{showChunks ? '▲' : '▼'}</span>
          </button>
          {showChunks && (
            <div className="space-y-2 mt-2">
              {result.retrieved_chunks.map((chunk, i) => (
                <RetrievedChunkBadge key={i} chunk={chunk} index={i} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Judge reasoning */}
      {scores?.reasoning && (
        <div className="border-t bg-amber-50">
          <button
            onClick={() => setShowReasoning(!showReasoning)}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-xs font-semibold text-amber-700 hover:bg-amber-100 transition-colors"
          >
            <span>⚖️ Judge 평가 이유</span>
            <span className="ml-auto text-amber-400">{showReasoning ? '▲' : '▼'}</span>
          </button>
          {showReasoning && (
            <div className="px-4 pb-3 text-xs text-amber-900 leading-relaxed">
              {scores.reasoning}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Stage4LLM({
  lock,
}: {
  lock: PipelineLock
}) {
  const navigate = useNavigate()
  const [selectedLLMs, setSelectedLLMs] = useState<string[]>([])
  const [judgeModel, setJudgeModel] = useState('')
  const [runId, setRunId] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [progress, setProgress] = useState(0)
  const [total, setTotal] = useState(0)
  const [liveResults, setLiveResults] = useState<TestResult[]>([])
  const [expandedQ, setExpandedQ] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  const { data: llmModels = [] } = useQuery({ queryKey: ['llm-models'], queryFn: listLLMModels })
  const { data: questions = [] } = useQuery({ queryKey: ['questions'], queryFn: () => listQuestions() })
  const { data: finalResults } = useQuery({
    queryKey: ['run-results', runId],
    queryFn: () => getRunResults(runId!),
    enabled: !!runId && !streaming,
  })

  const results = streaming ? liveResults : (finalResults ?? liveResults)

  const toggleLLM = (id: string) =>
    setSelectedLLMs(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])

  const startRun = async () => {
    if (selectedLLMs.length === 0) return
    const run = await createRun({
      embedding_model_ids: [lock.embModelId],
      llm_model_ids: selectedLLMs,
      question_ids: null,
      rag_enabled: true,
      top_k: 5,
      similarity_threshold: 0.0,
      search_strategy: lock.searchStrategy as any,
      judge_model: judgeModel,
      run_name: `Pipeline Test — ${lock.filename}`,
    })
    setRunId(run.id)
    setStreaming(true)
    setLiveResults([])
    setProgress(0)
    setTotal(run.total_tests)

    const es = new EventSource(`/api/runs/${run.id}/stream`)
    eventSourceRef.current = es
    es.addEventListener('result', (e) => {
      const data: TestResult = JSON.parse(e.data)
      setLiveResults(prev => [...prev, data])
      setProgress(p => p + 1)
    })
    es.addEventListener('complete', () => {
      es.close()
      setStreaming(false)
    })
    es.onerror = () => { es.close(); setStreaming(false) }
  }

  // Group results by question
  const byQuestion: Record<string, TestResult[]> = {}
  for (const r of results) {
    if (!byQuestion[r.question]) byQuestion[r.question] = []
    byQuestion[r.question].push(r)
  }

  // Summary stats per LLM
  const llmStats: Record<string, { overall: number; count: number }> = {}
  for (const r of results) {
    if (!r.scores) continue
    if (!llmStats[r.llm_model_id]) llmStats[r.llm_model_id] = { overall: 0, count: 0 }
    llmStats[r.llm_model_id].overall += r.scores.overall
    llmStats[r.llm_model_id].count++
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-800">Stage 4 — LLM 생성 비교</h2>
        <p className="text-sm text-slate-500 mt-1">
          잠금된 파이프라인에서 여러 LLM 모델을 비교합니다. 검색된 청크와 Judge 평가 이유를 확인하세요.
        </p>
      </div>

      {!runId && (
        <div className="bg-slate-50 rounded-xl p-5 space-y-5 border">
          <div>
            <h3 className="font-semibold text-slate-700 mb-3">LLM 모델 선택</h3>
            <div className="grid grid-cols-2 gap-2">
              {llmModels.filter(m => m.available).map(m => (
                <button
                  key={m.id}
                  onClick={() => toggleLLM(m.id)}
                  className={`text-left p-3 rounded-lg border-2 text-sm transition-all
                    ${selectedLLMs.includes(m.id) ? 'border-indigo-500 bg-indigo-50 font-semibold' : 'border-slate-200 hover:border-indigo-300'}`}
                >
                  <div>{m.name}</div>
                  <div className="text-xs text-slate-400 mt-0.5 capitalize">{m.provider}</div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Judge 모델 (선택사항)</h3>
            <select
              value={judgeModel}
              onChange={e => setJudgeModel(e.target.value)}
              className="border rounded-lg px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-indigo-300"
            >
              <option value="">Judge 없음 (점수 채점 생략)</option>
              {llmModels.filter(m => m.available).map(m => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </div>

          <button
            disabled={selectedLLMs.length === 0}
            onClick={startRun}
            className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-semibold disabled:opacity-40 hover:bg-indigo-700 transition"
          >
            테스트 실행 ({selectedLLMs.length}개 LLM × {questions.length}개 질문)
          </button>
        </div>
      )}

      {/* Progress */}
      {streaming && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-indigo-800">테스트 진행 중...</span>
            <span className="text-sm text-indigo-600">{progress} / {total}</span>
          </div>
          <div className="w-full bg-indigo-200 rounded-full h-2">
            <div
              className="bg-indigo-600 h-2 rounded-full transition-all"
              style={{ width: total > 0 ? `${(progress / total) * 100}%` : '0%' }}
            />
          </div>
        </div>
      )}

      {/* LLM summary */}
      {Object.keys(llmStats).length > 0 && (
        <div>
          <h3 className="font-semibold text-slate-700 mb-3">LLM 요약 성적</h3>
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(llmStats)
              .sort((a, b) => b[1].overall / b[1].count - a[1].overall / a[1].count)
              .map(([llmId, s]) => (
                <div key={llmId} className="bg-white border rounded-xl p-4">
                  <div className="font-semibold text-slate-800 mb-1">{shortId(llmId)}</div>
                  <div className="text-3xl font-bold text-indigo-700 tabular-nums">
                    {(s.overall / s.count).toFixed(1)}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">avg overall · {s.count}개</div>
                </div>
              ))}
          </div>
          {runId && (
            <button
              onClick={() => navigate(`/results/${runId}`)}
              className="mt-3 text-sm text-indigo-600 hover:underline"
            >
              상세 결과 보기 →
            </button>
          )}
        </div>
      )}

      {/* Question-by-question drill down */}
      {Object.keys(byQuestion).length > 0 && (
        <div>
          <h3 className="font-semibold text-slate-700 mb-3">질문별 드릴다운</h3>
          <div className="space-y-3">
            {Object.entries(byQuestion).map(([question, qResults]) => (
              <div key={question} className="border rounded-xl overflow-hidden">
                <button
                  onClick={() => setExpandedQ(expandedQ === question ? null : question)}
                  className="w-full flex items-start gap-3 px-5 py-4 text-left bg-white hover:bg-slate-50 transition-colors"
                >
                  <span className="text-slate-300 mt-0.5">{expandedQ === question ? '▼' : '▶'}</span>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-slate-800">{question}</p>
                    <div className="flex gap-2 mt-1.5 flex-wrap">
                      {qResults.map(r => r.scores && (
                        <span key={r.llm_model_id} className={`text-xs px-2 py-0.5 rounded-full font-semibold ${scoreColor(r.scores.overall)}`}>
                          {shortId(r.llm_model_id)}: {r.scores.overall.toFixed(1)}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>

                {expandedQ === question && (
                  <div className="border-t p-4 space-y-4 bg-slate-50">
                    {/* Show retrieved chunks once (same for all LLMs since same embedding) */}
                    {qResults[0]?.retrieved_chunks.length > 0 && (
                      <div>
                        <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-2">
                          검색된 청크 ({qResults[0].retrieved_chunks.length}개)
                        </p>
                        <div className="space-y-1.5">
                          {qResults[0].retrieved_chunks.map((chunk, i) => (
                            <RetrievedChunkBadge key={i} chunk={chunk} index={i} />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Per-LLM responses */}
                    <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Math.min(qResults.length, 2)}, 1fr)` }}>
                      {qResults.map(r => (
                        <ResultCard key={r.llm_model_id} result={r} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main PipelinePage
// ─────────────────────────────────────────────────────────────────────────────

export default function PipelinePage() {
  const [activeStage, setActiveStage] = useState<StageId>(1)
  const [maxReached, setMaxReached] = useState<StageId>(1)
  const [lock, setLock] = useState<Partial<PipelineLock>>({})

  const advance = (stage: StageId) => {
    setActiveStage(stage)
    if (stage > maxReached) setMaxReached(stage)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <StepperRail
        active={activeStage}
        maxReached={maxReached}
        lock={lock}
        onSelect={(id) => setActiveStage(id)}
      />

      <div className="pb-16">
        {activeStage === 1 && (
          <Stage1Parser
            onConfirm={(docId, filename, parser) => {
              setLock(prev => ({ ...prev, docId, filename, parser }))
              advance(2)
            }}
          />
        )}
        {activeStage === 2 && lock.docId && lock.parser && (
          <Stage2Chunk
            docId={lock.docId}
            parser={lock.parser}
            onConfirm={(strategy, chunkSize, overlap, label) => {
              setLock(prev => ({ ...prev, chunkStrategy: strategy, chunkSize, overlap, chunkLabel: label }))
              advance(3)
            }}
          />
        )}
        {activeStage === 3 && lock.docId && lock.parser && lock.chunkStrategy && (
          <Stage3Search
            docId={lock.docId}
            parser={lock.parser}
            chunkStrategy={lock.chunkStrategy}
            chunkSize={lock.chunkSize ?? 600}
            overlap={lock.overlap ?? 0}
            onConfirm={(embModelId, searchStrategy) => {
              setLock(prev => ({ ...prev, embModelId, searchStrategy }))
              advance(4)
            }}
          />
        )}
        {activeStage === 4 && lock.docId && lock.embModelId && (
          <Stage4LLM
            lock={lock as PipelineLock}
          />
        )}

        {/* Guard messages for locked stages */}
        {activeStage === 2 && !lock.docId && (
          <div className="max-w-xl mx-auto mt-20 text-center text-slate-500">
            <p className="text-4xl mb-4">🔒</p>
            <p>먼저 Stage 1에서 문서와 파서를 선택해주세요.</p>
          </div>
        )}
        {activeStage === 3 && !lock.chunkStrategy && (
          <div className="max-w-xl mx-auto mt-20 text-center text-slate-500">
            <p className="text-4xl mb-4">🔒</p>
            <p>먼저 Stage 2에서 청킹 전략을 선택해주세요.</p>
          </div>
        )}
        {activeStage === 4 && !lock.embModelId && (
          <div className="max-w-xl mx-auto mt-20 text-center text-slate-500">
            <p className="text-4xl mb-4">🔒</p>
            <p>먼저 Stage 3에서 임베딩 모델과 검색 전략을 선택해주세요.</p>
          </div>
        )}
      </div>
    </div>
  )
}
