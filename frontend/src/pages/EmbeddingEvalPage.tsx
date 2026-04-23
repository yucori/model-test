import { useCallback, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listEmbeddingModels, listDocuments, listParsers,
  uploadDocument, vectorizeDocuments, deleteDocument, compareEmbeddings,
  previewChunks, generateTestCases,
} from '../lib/api'
import type { PreviewChunk, GeneratedTestCase } from '../lib/api'
import type { EmbTestCase, EmbModelResult, EmbCompareResult, DocumentInfo, ParserInfo } from '../types'

// ── Pre-loaded FarmOS CS test cases ──────────────────────────────────────────

const DEFAULT_TEST_CASES: EmbTestCase[] = [
  { id: 'tc-1', query: '무통장 입금은 주문 후 얼마 이내에 해야 하나요?',   expected_contains: '입금',    category: '결제'      },
  { id: 'tc-2', query: '배송비는 얼마이고 무료배송 기준이 어떻게 되나요?', expected_contains: '배송비',  category: '배송'      },
  { id: 'tc-3', query: '환불 처리 기간이 결제 수단별로 다른가요?',          expected_contains: '환불',    category: '환불'      },
  { id: 'tc-4', query: '신선 냉장 배송과 일반 배송 방식 차이가 뭔가요?',   expected_contains: '배송 방식', category: '배송'    },
  { id: 'tc-5', query: '고객센터 전화 운영 시간이 어떻게 되나요?',          expected_contains: '운영',    category: '고객서비스' },
  { id: 'tc-6', query: '회원 등급별 혜택이 어떻게 되나요?',                 expected_contains: '등급',    category: '회원'      },
  { id: 'tc-7', query: '상품 품질 불량 시 어떻게 처리되나요?',              expected_contains: '품질',    category: '품질보증'  },
  { id: 'tc-8', query: '교환 신청할 수 있는 사유가 뭐가 있나요?',           expected_contains: '교환',    category: '반품/교환' },
]

const CHUNK_PRESETS = [
  { label: '소형 300', chunkSize: 300, overlap: 50  },
  { label: '표준 600', chunkSize: 600, overlap: 100 },
  { label: '대형 1000', chunkSize: 1000, overlap: 150 },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function shortModelId(id: string) {
  return id
    .replace('hf:', '').replace('BAAI/', '').replace('intfloat/', '')
    .replace('jhgan/', '').replace('google/', '').replace('nomic-ai/', '')
    .replace('jinaai/', '').replace('Qwen/', '').replace('openai:', '')
    .replace('local:', '').replace('ollama:', '')
}

function pct(v: number) { return `${(v * 100).toFixed(0)}%` }

function HitBadge({ rank }: { rank: number | null }) {
  if (rank === null) return <span className="text-xs text-slate-400">✗</span>
  const color = rank === 1 ? 'text-emerald-600 font-semibold' : rank <= 3 ? 'text-sky-600 font-medium' : 'text-slate-500'
  return <span className={`text-xs ${color}`}>#{rank}</span>
}

function ScoreBar({ value }: { value: number }) {
  const w = Math.round(value * 100)
  const color = w >= 75 ? 'bg-emerald-500' : w >= 50 ? 'bg-sky-500' : w >= 25 ? 'bg-amber-400' : 'bg-red-400'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-slate-200 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${w}%` }} />
      </div>
      <span className="text-xs text-slate-600 w-8">{pct(value)}</span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function EmbeddingEvalPage() {
  const qc = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)

  // ── Setup state ───────────────────────────────────────────────────────────
  const [uploading, setUploading]         = useState(false)
  const [dragOver, setDragOver]           = useState(false)
  const [selPdfParser, setSelPdfParser]   = useState('pdfplumber')
  const [selDocxParser, setSelDocxParser] = useState('python-docx')
  const [chunkSize, setChunkSize]         = useState(600)
  const [overlap, setOverlap]             = useState(100)
  const [selModels, setSelModels]         = useState<string[]>([])

  // ── Eval state ────────────────────────────────────────────────────────────
  const [testCases, setTestCases]         = useState<EmbTestCase[]>(DEFAULT_TEST_CASES)
  const [topK, setTopK]                   = useState(5)
  const [result, setResult]               = useState<EmbCompareResult | null>(null)
  const [expandedCase, setExpandedCase]   = useState<string | null>(null)
  const [editingCase, setEditingCase]     = useState<string | null>(null)

  // ── Wizard state ──────────────────────────────────────────────────────────
  const [wizardQuery, setWizardQuery]         = useState('')
  const [wizardModel, setWizardModel]         = useState('')
  const [wizardChunks, setWizardChunks]       = useState<PreviewChunk[]>([])
  const [wizardSelected, setWizardSelected]   = useState<PreviewChunk | null>(null)
  const [wizardPhrase, setWizardPhrase]       = useState('')
  const [wizardCategory, setWizardCategory]   = useState('')

  // ── Auto-generate state ───────────────────────────────────────────────────
  const [genModel, setGenModel]               = useState('')
  const [genJudge, setGenJudge]               = useState('gemini-2.0-flash')
  const [genCases, setGenCases]               = useState<GeneratedTestCase[]>([])
  const [genSelected, setGenSelected]         = useState<Set<string>>(new Set())
  const [showGen, setShowGen]                 = useState(false)

  // ── Queries ───────────────────────────────────────────────────────────────
  const { data: allModels = [] } = useQuery({ queryKey: ['emb-models'], queryFn: listEmbeddingModels })
  const { data: parsers = [] }   = useQuery({ queryKey: ['parsers'],    queryFn: listParsers })
  const { data: docs = [] }      = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    refetchInterval: (q) =>
      (q.state.data as DocumentInfo[] | undefined)?.some((d) => d.processing) ? 2000 : 5000,
  })

  const invalidateDocs = () => qc.invalidateQueries({ queryKey: ['documents'] })

  // ── Mutations ─────────────────────────────────────────────────────────────
  const vectorizeMut = useMutation({
    mutationFn: () => vectorizeDocuments(selModels, undefined, selPdfParser, chunkSize, overlap, 'paragraph', selPdfParser, selDocxParser),
    onSuccess: invalidateDocs,
  })
  const deleteMut = useMutation({ mutationFn: deleteDocument, onSuccess: invalidateDocs })
  const evalMut   = useMutation({
    mutationFn: compareEmbeddings,
    onSuccess: (data) => setResult(data),
  })
  const genMut = useMutation({
    mutationFn: (modelId: string) => generateTestCases({
      emb_model_id: modelId,
      num_cases: 20,
      judge_model: genJudge,
    }),
    onSuccess: (data) => {
      setGenCases(data.cases)
      setGenSelected(new Set(data.cases.map((c) => c.id)))
      setShowGen(true)
    },
  })

  const previewMut = useMutation({
    mutationFn: ({ query, modelId }: { query: string; modelId: string }) =>
      previewChunks(query, modelId, 5),
    onSuccess: (data) => {
      setWizardChunks(data.chunks)
      setWizardSelected(null)
      setWizardPhrase('')
    },
  })

  // ── Derived ───────────────────────────────────────────────────────────────
  const typedDocs   = docs as DocumentInfo[]
  const anyProcessing = typedDocs.some((d) => d.processing)

  const vectorizedCount = (modelId: string) =>
    typedDocs.filter((d) => modelId in d.processed_embeddings).length

  // 선택된 모델 중 벡터화가 필요한 문서가 있는지
  const needsVectorize = selModels.length > 0 && typedDocs.some((d) =>
    !d.processing && selModels.some((eid) => !(eid in d.processed_embeddings))
  )

  // 평가 실행 가능 여부: 선택된 모델 중 하나라도 벡터화 완료
  const canEval = selModels.some((id) => vectorizedCount(id) > 0) && testCases.length > 0

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setUploading(true)
    for (const f of Array.from(files)) {
      await uploadDocument(f)
    }
    await invalidateDocs()
    setUploading(false)
  }, [])

  const toggleModel = (id: string) =>
    setSelModels((prev) => prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id])

  const updateCase = (id: string, field: keyof EmbTestCase, value: string) =>
    setTestCases((prev) => prev.map((tc) => tc.id === id ? { ...tc, [field]: value } : tc))

  const addCase = () => {
    const id = `tc-${Date.now()}`
    setTestCases((prev) => [...prev, { id, query: '', expected_contains: '', category: '' }])
    setEditingCase(id)
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-6 space-y-5">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div>
        <h1 className="text-xl font-bold text-slate-800">임베딩 모델 비교</h1>
        <p className="mt-1 text-sm text-slate-500">
          문서를 업로드하고 벡터화한 뒤, FarmOS CS 질문으로 Hit@K를 측정해 모델을 비교합니다.
        </p>
      </div>

      {/* ── Step 1: 문서 업로드 ─────────────────────────────────────────────── */}
      <section className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-700">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-600 text-white text-xs mr-2">1</span>
            문서 업로드
          </span>
          <span className="text-xs text-slate-400">{typedDocs.length}개 업로드됨</span>
        </div>

        <div className="p-4 space-y-3">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
            onClick={() => !uploading && fileRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-all
              ${dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'}`}
          >
            <input ref={fileRef} type="file" accept=".pdf,.docx,.doc" multiple className="hidden"
              onChange={(e) => handleFiles(e.target.files)} />
            {uploading ? (
              <p className="text-sm text-indigo-500">업로드 중...</p>
            ) : (
              <>
                <p className="text-sm text-slate-600 font-medium">클릭하거나 파일을 드래그</p>
                <p className="text-xs text-slate-400 mt-0.5">PDF, DOCX · 여러 파일 동시 가능</p>
              </>
            )}
          </div>

          {/* Doc list */}
          {typedDocs.length > 0 && (
            <ul className="divide-y divide-slate-50 border border-slate-100 rounded-lg overflow-hidden">
              {typedDocs.map((doc) => (
                <li key={doc.id} className="flex items-center gap-3 px-3 py-2 text-xs bg-white">
                  <span className={`shrink-0 px-1.5 py-0.5 rounded font-bold text-xs
                    ${doc.file_type === 'pdf' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>
                    {doc.file_type.toUpperCase()}
                  </span>
                  <span className="flex-1 truncate text-slate-700">{doc.filename}</span>
                  {doc.processing ? (
                    <span className="text-indigo-500">벡터화 중...</span>
                  ) : (
                    <span className="text-slate-400">
                      {Object.keys(doc.processed_embeddings).length > 0
                        ? `${Object.keys(doc.processed_embeddings).length}개 모델 완료`
                        : '미벡터화'}
                    </span>
                  )}
                  <button
                    onClick={() => { if (confirm(`"${doc.filename}" 삭제?`)) deleteMut.mutate(doc.id) }}
                    className="text-slate-300 hover:text-red-400 transition-colors"
                  >✕</button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      {/* ── Step 2: 청킹 & 벡터화 ───────────────────────────────────────────── */}
      <section className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
          <span className="text-sm font-semibold text-slate-700">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-600 text-white text-xs mr-2">2</span>
            청킹 설정 & 벡터화
          </span>
        </div>

        <div className="p-4 grid grid-cols-3 gap-6">
          {/* Parser */}
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-2">PDF 파서</p>
            <div className="space-y-1 mb-3">
              {(parsers as ParserInfo[]).filter((p) => p.available && p.file_types.includes('pdf')).map((p) => (
                <label key={p.id} className="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="emb-pdf-parser" checked={selPdfParser === p.id}
                    onChange={() => setSelPdfParser(p.id)} className="accent-indigo-600" />
                  <span className="text-sm text-slate-700">{p.name}</span>
                </label>
              ))}
            </div>
            <p className="text-xs font-semibold text-slate-500 mb-2">DOCX 파서</p>
            <div className="space-y-1">
              {(parsers as ParserInfo[]).filter((p) => p.available && p.file_types.includes('docx')).map((p) => (
                <label key={p.id} className="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="emb-docx-parser" checked={selDocxParser === p.id}
                    onChange={() => setSelDocxParser(p.id)} className="accent-indigo-600" />
                  <span className="text-sm text-slate-700">{p.name}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Chunk size */}
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-2">청크 크기</p>
            <div className="flex gap-1 mb-3">
              {CHUNK_PRESETS.map((p) => (
                <button key={p.label}
                  onClick={() => { setChunkSize(p.chunkSize); setOverlap(p.overlap) }}
                  className={`flex-1 py-1 rounded text-xs font-medium border transition-colors
                    ${chunkSize === p.chunkSize ? 'bg-indigo-600 text-white border-indigo-600' : 'border-slate-200 text-slate-600 hover:border-indigo-300'}`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <div className="space-y-2">
              <label className="text-xs text-slate-400 flex justify-between">
                <span>크기</span><span className="font-semibold text-indigo-600">{chunkSize}자</span>
              </label>
              <input type="range" min={100} max={1200} step={50} value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-full accent-indigo-600" />
              <label className="text-xs text-slate-400 flex justify-between">
                <span>오버랩</span><span className="font-semibold text-indigo-600">{overlap}자</span>
              </label>
              <input type="range" min={0} max={300} step={25} value={overlap}
                onChange={(e) => setOverlap(Number(e.target.value))}
                className="w-full accent-indigo-600" />
            </div>
          </div>

          {/* Models + vectorize */}
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-2">임베딩 모델</p>
            <div className="space-y-1.5 mb-3">
              {allModels.map((m) => {
                const count = vectorizedCount(m.id)
                return (
                  <label key={m.id} className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" className="accent-indigo-600 rounded"
                      checked={selModels.includes(m.id)}
                      onChange={() => toggleModel(m.id)} />
                    <span className="text-xs flex-1 text-slate-700">{shortModelId(m.id)}</span>
                    {count > 0 && (
                      <span className="text-xs text-emerald-600 bg-emerald-50 px-1 py-0.5 rounded">
                        ✓{count}
                      </span>
                    )}
                  </label>
                )
              })}
            </div>

            {anyProcessing ? (
              <div className="text-xs text-indigo-500 py-2 text-center">벡터화 중...</div>
            ) : (
              <button
                onClick={() => vectorizeMut.mutate()}
                disabled={selModels.length === 0 || typedDocs.length === 0 || !needsVectorize || vectorizeMut.isPending}
                className="w-full py-2 bg-indigo-600 text-white text-xs font-semibold rounded-lg
                           hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {vectorizeMut.isPending ? '시작 중...' :
                  !needsVectorize && selModels.length > 0 ? '✓ 벡터화 완료' :
                  selModels.length === 0 ? '모델을 선택하세요' :
                  typedDocs.length === 0 ? '문서를 먼저 업로드하세요' :
                  `벡터화 실행 (${selModels.length}개 모델)`}
              </button>
            )}
          </div>
        </div>
      </section>

      {/* ── Step 3: 테스트 케이스 + Wizard ─────────────────────────────────── */}
      <section className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-700">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-600 text-white text-xs mr-2">3</span>
            테스트 케이스 ({testCases.length}개)
          </span>
          <div className="flex items-center gap-2">
            {/* 자동 생성 */}
            <div className="flex items-center gap-1.5">
              <select
                className="text-xs px-2 py-1 border border-slate-200 rounded bg-white"
                value={genModel}
                onChange={(e) => setGenModel(e.target.value)}
              >
                <option value="">임베딩 모델</option>
                {allModels.filter((m) => vectorizedCount(m.id) > 0).map((m) => (
                  <option key={m.id} value={m.id}>{shortModelId(m.id)}</option>
                ))}
              </select>
              <select
                className="text-xs px-2 py-1 border border-slate-200 rounded bg-white"
                value={genJudge}
                onChange={(e) => setGenJudge(e.target.value)}
              >
                <optgroup label="Google">
                  <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                  <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                </optgroup>
                <optgroup label="OpenRouter">
                  <option value="openrouter:openai/gpt-5-nano">GPT-5 Nano</option>
                  <option value="openrouter:openai/gpt-4.1-nano">GPT-4.1 Nano</option>
                  <option value="openrouter:openai/gpt-4.1-mini">GPT-4.1 Mini</option>
                  <option value="openrouter:openai/gpt-4o-mini">GPT-4o Mini</option>
                </optgroup>
              </select>
              <button
                onClick={() => genModel && genMut.mutate(genModel)}
                disabled={!genModel || genMut.isPending}
                className="px-3 py-1 bg-indigo-600 text-white rounded text-xs font-medium
                           hover:bg-indigo-700 disabled:opacity-40 transition-colors"
              >
                {genMut.isPending ? '생성 중...' : '자동 생성'}
              </button>
            </div>
            <button onClick={addCase}
              className="px-3 py-1 bg-slate-100 text-slate-600 rounded text-xs font-medium hover:bg-slate-200 transition-colors">
              + 직접 추가
            </button>
          </div>
        </div>

        {/* 자동 생성 결과 패널 */}
        {showGen && genCases.length > 0 && (
          <div className="border-b border-slate-200 bg-indigo-50/50 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-indigo-700">
                자동 생성 결과 {genCases.length}개 — 추가할 케이스를 선택하세요
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setGenSelected(new Set(genCases.map((c) => c.id)))}
                  className="text-xs text-indigo-600 hover:underline"
                >전체 선택</button>
                <button
                  onClick={() => setGenSelected(new Set())}
                  className="text-xs text-slate-500 hover:underline"
                >전체 해제</button>
                <button
                  onClick={() => {
                    const toAdd = genCases.filter((c) => genSelected.has(c.id))
                    setTestCases((prev) => [
                      ...prev,
                      ...toAdd.map((c) => ({ ...c })),
                    ])
                    setShowGen(false)
                    setGenCases([])
                  }}
                  disabled={genSelected.size === 0}
                  className="px-3 py-1 bg-emerald-600 text-white rounded text-xs font-medium
                             hover:bg-emerald-700 disabled:opacity-40 transition-colors"
                >
                  {genSelected.size}개 추가
                </button>
                <button
                  onClick={() => setShowGen(false)}
                  className="text-xs text-slate-400 hover:text-slate-600"
                >✕</button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {genCases.map((c) => {
                const checked = genSelected.has(c.id)
                return (
                  <label
                    key={c.id}
                    className={`flex gap-2 p-2.5 rounded-lg border cursor-pointer transition-all
                      ${checked ? 'border-indigo-300 bg-white' : 'border-slate-200 bg-white/60 opacity-60'}`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => setGenSelected((prev) => {
                        const next = new Set(prev)
                        checked ? next.delete(c.id) : next.add(c.id)
                        return next
                      })}
                      className="mt-0.5 accent-indigo-600 shrink-0"
                    />
                    <div className="min-w-0">
                      <p className="text-xs text-slate-700 leading-relaxed">{c.question}</p>
                      <p className="text-xs mt-0.5">
                        <span className="font-mono text-indigo-600">{c.expected_contains}</span>
                        {c.category && <span className="text-slate-400"> · {c.category}</span>}
                      </p>
                    </div>
                  </label>
                )
              })}
            </div>
          </div>
        )}
        {/* 생성 중 */}
        {genMut.isPending && (
          <div className="px-4 py-4 border-b border-slate-100 bg-slate-50 text-center">
            <p className="text-sm text-indigo-600 font-medium">케이스 생성 중...</p>
            <p className="text-xs text-slate-400 mt-1">청크에서 질문과 기대 구문을 생성합니다. 20~40초 소요될 수 있습니다.</p>
          </div>
        )}
        {/* 에러 */}
        {genMut.isError && (
          <div className="px-4 py-3 bg-red-50 border-b border-red-200 text-xs text-red-600">
            생성 오류: {String((genMut.error as any)?.response?.data?.detail ?? genMut.error)}
          </div>
        )}
        {/* 성공했지만 결과 없음 */}
        {genMut.isSuccess && genCases.length === 0 && (
          <div className="px-4 py-3 bg-amber-50 border-b border-amber-200 text-xs text-amber-700">
            생성된 케이스가 없습니다. Gemini API 키가 설정되어 있는지 확인하거나 다른 모델을 선택해 보세요.
          </div>
        )}

        <div className="grid grid-cols-2 divide-x divide-slate-100">

          {/* 왼쪽: 케이스 목록 */}
          <div className="divide-y divide-slate-50">
            {testCases.length === 0 && (
              <p className="px-4 py-6 text-xs text-slate-400 text-center">
                오른쪽 wizard로 케이스를 추가하세요.
              </p>
            )}
            {testCases.map((tc, i) => (
              <div key={tc.id} className="px-4 py-2.5">
                {editingCase === tc.id ? (
                  <div className="space-y-1.5">
                    <input className="w-full text-xs px-2 py-1.5 border border-slate-200 rounded"
                      placeholder="질문" value={tc.query}
                      onChange={(e) => updateCase(tc.id, 'query', e.target.value)} />
                    <div className="flex gap-2">
                      <input className="flex-1 text-xs px-2 py-1.5 border border-slate-200 rounded"
                        placeholder="기대 구문" value={tc.expected_contains}
                        onChange={(e) => updateCase(tc.id, 'expected_contains', e.target.value)} />
                      <input className="w-24 text-xs px-2 py-1.5 border border-slate-200 rounded"
                        placeholder="카테고리" value={tc.category}
                        onChange={(e) => updateCase(tc.id, 'category', e.target.value)} />
                      <button onClick={() => setEditingCase(null)}
                        className="px-3 py-1 bg-indigo-600 text-white rounded text-xs">완료</button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 group">
                    <span className="text-xs text-slate-400 w-4 shrink-0">{i + 1}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-slate-700 truncate">{tc.query}</p>
                      <p className="text-xs text-slate-400">
                        <span className="font-mono text-indigo-600">{tc.expected_contains}</span>
                        {tc.category && ` · ${tc.category}`}
                      </p>
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => setEditingCase(tc.id)}
                        className="px-2 py-0.5 text-xs text-slate-500 hover:bg-slate-100 rounded">편집</button>
                      <button onClick={() => setTestCases((prev) => prev.filter((t) => t.id !== tc.id))}
                        className="px-2 py-0.5 text-xs text-red-400 hover:bg-red-50 rounded">삭제</button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* 오른쪽: Wizard */}
          <div className="p-4 space-y-3 bg-slate-50/50">
            <p className="text-xs font-semibold text-slate-500">청크 확인 후 케이스 추가</p>

            {/* 질문 + 모델 선택 */}
            <input
              className="w-full text-xs px-2.5 py-2 border border-slate-200 rounded-lg bg-white"
              placeholder="질문 입력 (예: 배송비 무료 기준이 뭔가요?)"
              value={wizardQuery}
              onChange={(e) => setWizardQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && wizardQuery.trim() && wizardModel)
                  previewMut.mutate({ query: wizardQuery, modelId: wizardModel })
              }}
            />
            <div className="flex gap-2">
              <select
                className="flex-1 text-xs px-2 py-1.5 border border-slate-200 rounded-lg bg-white"
                value={wizardModel}
                onChange={(e) => setWizardModel(e.target.value)}
              >
                <option value="">모델 선택</option>
                {allModels.filter((m) => vectorizedCount(m.id) > 0).map((m) => (
                  <option key={m.id} value={m.id}>{shortModelId(m.id)}</option>
                ))}
              </select>
              <button
                onClick={() => previewMut.mutate({ query: wizardQuery, modelId: wizardModel })}
                disabled={!wizardQuery.trim() || !wizardModel || previewMut.isPending}
                className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-xs font-medium
                           hover:bg-indigo-700 disabled:opacity-40 transition-colors"
              >
                {previewMut.isPending ? '검색 중...' : '검색'}
              </button>
            </div>

            {/* 청크 목록 */}
            {previewMut.isError && (
              <p className="text-xs text-red-500">오류: {String(previewMut.error)}</p>
            )}
            {wizardChunks.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs text-slate-400">정답 청크를 클릭하세요</p>
                {wizardChunks.map((chunk) => {
                  const isSelected = wizardSelected?.rank === chunk.rank
                  const displayText = chunk.matched_text || chunk.content
                  return (
                    <div
                      key={chunk.rank}
                      onClick={() => {
                        setWizardSelected(chunk)
                        setWizardPhrase(chunk.suggested_phrase)
                      }}
                      className={`p-2.5 rounded-lg border cursor-pointer transition-all text-xs
                        ${isSelected
                          ? 'border-indigo-400 bg-indigo-50'
                          : 'border-slate-200 bg-white hover:border-slate-300'}`}
                    >
                      <div className="flex justify-between mb-1">
                        <span className={`font-semibold ${isSelected ? 'text-indigo-600' : 'text-slate-500'}`}>
                          #{chunk.rank}
                        </span>
                        <span className="text-slate-400">{chunk.final_score.toFixed(3)}</span>
                      </div>
                      <p className="text-slate-600 leading-relaxed line-clamp-3 whitespace-pre-wrap">
                        {displayText.slice(0, 180)}{displayText.length > 180 ? '...' : ''}
                      </p>
                    </div>
                  )
                })}
              </div>
            )}

            {/* 선택된 청크 → 케이스 추가 */}
            {wizardSelected && (
              <div className="space-y-2 pt-1 border-t border-slate-200">
                <p className="text-xs text-slate-500">기대 구문 (이 텍스트가 청크에 있으면 히트)</p>
                <input
                  className="w-full text-xs px-2.5 py-2 border border-indigo-300 rounded-lg bg-white font-mono"
                  value={wizardPhrase}
                  onChange={(e) => setWizardPhrase(e.target.value)}
                  placeholder="기대 구문 입력"
                />
                <div className="flex gap-2">
                  <input
                    className="flex-1 text-xs px-2 py-1.5 border border-slate-200 rounded-lg bg-white"
                    placeholder="카테고리 (선택)"
                    value={wizardCategory}
                    onChange={(e) => setWizardCategory(e.target.value)}
                  />
                  <button
                    onClick={() => {
                      if (!wizardQuery.trim() || !wizardPhrase.trim()) return
                      const id = `tc-${Date.now()}`
                      setTestCases((prev) => [...prev, {
                        id,
                        query: wizardQuery,
                        expected_contains: wizardPhrase,
                        category: wizardCategory,
                      }])
                      setWizardQuery('')
                      setWizardChunks([])
                      setWizardSelected(null)
                      setWizardPhrase('')
                      setWizardCategory('')
                    }}
                    disabled={!wizardPhrase.trim()}
                    className="px-3 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-medium
                               hover:bg-emerald-700 disabled:opacity-40 transition-colors"
                  >
                    케이스 추가
                  </button>
                </div>
              </div>
            )}

            {wizardChunks.length === 0 && !previewMut.isPending && (
              <p className="text-xs text-slate-400 text-center py-4">
                질문을 입력하고 검색하면 실제 청크가 표시됩니다.
              </p>
            )}
          </div>
        </div>
      </section>

      {/* ── Step 4: 평가 실행 ───────────────────────────────────────────────── */}
      <section className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
          <span className="text-sm font-semibold text-slate-700">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-600 text-white text-xs mr-2">4</span>
            평가 실행
          </span>
        </div>
        <div className="p-4 flex items-center gap-6">
          <div>
            <p className="text-xs text-slate-500 mb-1.5">Top-K</p>
            <div className="flex gap-1">
              {[3, 5, 10].map((k) => (
                <button key={k} onClick={() => setTopK(k)}
                  className={`px-3 py-1.5 rounded text-sm font-medium transition-colors
                    ${topK === k ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>
                  {k}
                </button>
              ))}
            </div>
          </div>
          <div className="flex-1">
            {!canEval && (
              <p className="text-xs text-slate-400">
                {selModels.length === 0 ? '2단계에서 모델을 선택하세요.' :
                  anyProcessing ? '벡터화 완료를 기다리는 중...' :
                  '선택한 모델을 먼저 벡터화하세요.'}
              </p>
            )}
          </div>
          <button
            onClick={() => evalMut.mutate({ test_cases: testCases, emb_model_ids: selModels, top_k: topK })}
            disabled={!canEval || evalMut.isPending}
            className="px-6 py-2 bg-indigo-600 text-white rounded-lg text-sm font-semibold
                       hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {evalMut.isPending ? '평가 중...' : '평가 실행'}
          </button>
        </div>
        {evalMut.isError && (
          <div className="mx-4 mb-4 p-3 bg-red-50 border border-red-200 rounded text-xs text-red-700">
            오류: {String(evalMut.error)}
          </div>
        )}
      </section>

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      {result && (
        <>
          {/* Summary table */}
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
              <span className="text-sm font-semibold text-slate-700">
                결과 요약 — Top-{result.top_k} / {result.total_cases}개 케이스
              </span>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="text-left px-4 py-2 text-xs font-medium text-slate-500">모델</th>
                  <th className="px-4 py-2 text-xs font-medium text-slate-500 text-center">Hit@1</th>
                  <th className="px-4 py-2 text-xs font-medium text-slate-500 text-center">Hit@3</th>
                  <th className="px-4 py-2 text-xs font-medium text-slate-500 text-center">Hit@{result.top_k}</th>
                  <th className="px-4 py-2 text-xs font-medium text-slate-500">시각화</th>
                </tr>
              </thead>
              <tbody>
                {result.model_results.map((mr) => (
                  <tr key={mr.emb_model_id} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-700">
                      {shortModelId(mr.emb_model_id)}
                      {!mr.available && <span className="ml-1.5 px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded text-xs">미벡터화</span>}
                    </td>
                    {[mr.hit_at_1, mr.hit_at_3, mr.hit_at_k].map((v, i) => (
                      <td key={i} className="px-4 py-2.5 text-center">
                        <span className={`font-semibold ${v >= 0.7 ? 'text-emerald-600' : v >= 0.4 ? 'text-sky-600' : 'text-slate-500'}`}>
                          {pct(v)}
                        </span>
                      </td>
                    ))}
                    <td className="px-4 py-2.5"><ScoreBar value={mr.hit_at_1} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-case detail */}
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
              <span className="text-sm font-semibold text-slate-700">케이스별 적중 순위</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100">
                    <th className="text-left px-4 py-2 text-xs font-medium text-slate-500 w-8">#</th>
                    <th className="text-left px-4 py-2 text-xs font-medium text-slate-500">질문</th>
                    <th className="text-left px-4 py-2 text-xs font-medium text-slate-500">키워드</th>
                    {result.model_results.map((mr) => (
                      <th key={mr.emb_model_id} className="px-3 py-2 text-xs font-medium text-slate-500 text-center whitespace-nowrap">
                        {shortModelId(mr.emb_model_id)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {testCases.map((tc, i) => {
                    const isExpanded = expandedCase === tc.id
                    return (
                      <>
                        <tr key={tc.id}
                          className="border-b border-slate-50 hover:bg-slate-50 cursor-pointer"
                          onClick={() => setExpandedCase(isExpanded ? null : tc.id)}>
                          <td className="px-4 py-2.5 text-xs text-slate-400">{i + 1}</td>
                          <td className="px-4 py-2.5 max-w-xs">
                            <div className="flex items-start gap-1">
                              <span className="text-slate-300 text-xs mt-0.5">{isExpanded ? '▾' : '▸'}</span>
                              <span className="text-xs text-slate-700">{tc.query}</span>
                            </div>
                            {tc.category && <span className="ml-4 text-xs text-slate-400">{tc.category}</span>}
                          </td>
                          <td className="px-4 py-2.5 font-mono text-xs text-slate-500">{tc.expected_contains}</td>
                          {result.model_results.map((mr) => {
                            const tr = mr.test_results.find((r) => r.test_case_id === tc.id)
                            return (
                              <td key={mr.emb_model_id} className="px-3 py-2.5 text-center">
                                {mr.available
                                  ? <HitBadge rank={tr?.hit_rank ?? null} />
                                  : <span className="text-xs text-slate-300">—</span>}
                              </td>
                            )
                          })}
                        </tr>
                        {isExpanded && (
                          <tr key={`${tc.id}-exp`} className="bg-slate-50 border-b border-slate-100">
                            <td />
                            <td colSpan={2 + result.model_results.length} className="px-4 py-3">
                              <div className="grid gap-4"
                                style={{ gridTemplateColumns: `repeat(${result.model_results.length}, 1fr)` }}>
                                {result.model_results.map((mr) => {
                                  const tr = mr.test_results.find((r) => r.test_case_id === tc.id)
                                  if (!mr.available || !tr) return (
                                    <div key={mr.emb_model_id} className="text-xs text-slate-400">미벡터화</div>
                                  )
                                  return (
                                    <div key={mr.emb_model_id}>
                                      <p className="text-xs font-semibold text-slate-500 mb-1">{shortModelId(mr.emb_model_id)}</p>
                                      <div className="space-y-1.5">
                                        {tr.chunks.slice(0, 3).map((chunk, ci) => {
                                          const text = chunk.matched_text || chunk.content
                                          const isHit = tc.expected_contains && text.toLowerCase().includes(tc.expected_contains.toLowerCase())
                                          return (
                                            <div key={ci}
                                              className={`text-xs p-2 rounded border ${isHit ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-white'}`}>
                                              <div className="flex justify-between mb-0.5">
                                                <span className={`font-medium ${isHit ? 'text-emerald-700' : 'text-slate-500'}`}>
                                                  #{ci + 1}{isHit ? ' ✓ hit' : ''}
                                                </span>
                                                <span className="text-slate-400">{chunk.final_score.toFixed(3)}</span>
                                              </div>
                                              <p className="text-slate-600 leading-relaxed line-clamp-3">
                                                {text.slice(0, 200)}{text.length > 200 ? '...' : ''}
                                              </p>
                                            </div>
                                          )
                                        })}
                                      </div>
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
        </>
      )}
    </div>
  )
}
