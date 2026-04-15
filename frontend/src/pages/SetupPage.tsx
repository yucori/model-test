import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listEmbeddingModels, listDocuments, uploadDocument, vectorizeDocuments,
  deleteDocument, listQuestions, createQuestion, deleteQuestion,
} from '../lib/api'
import { formatBytes, formatDate } from '../lib/utils'
import LoadingSpinner from '../components/common/LoadingSpinner'
import type { DocumentInfo, EmbeddingModelInfo, TestQuestion } from '../types'

const CATEGORIES = ['배송', '반품/환불', '상품', '보관', '결제', '회원', '고객센터', '기타']
const STORAGE_KEY = 'rag_bench_selected_emb'

// Chunking presets for quick selection
const CHUNK_PRESETS = [
  { label: '소형 (300/50)',  chunkSize: 300,  overlap: 50,  desc: '짧은 문답, 세밀한 검색' },
  { label: '표준 (600/100)', chunkSize: 600,  overlap: 100, desc: '일반 CS 문서 (기본값)' },
  { label: '대형 (1000/150)', chunkSize: 1000, overlap: 150, desc: '긴 설명·정책 문서' },
]

function loadSavedEmb(): string[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]') } catch { return [] }
}

export default function SetupPage() {
  const qc = useQueryClient()

  // ── Embedding model selection (persisted) ────────────────────────────────────
  const [selEmb, setSelEmb] = useState<string[]>(loadSavedEmb)

  // ── Chunking strategy ────────────────────────────────────────────────────────
  const [chunkSize, setChunkSize] = useState(600)
  const [overlap, setOverlap] = useState(100)

  const { data: embModels = [] } = useQuery({
    queryKey: ['embeddingModels'],
    queryFn: listEmbeddingModels,
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selEmb))
  }, [selEmb])

  const toggleEmb = (id: string) =>
    setSelEmb((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])

  // ── Documents ────────────────────────────────────────────────────────────────
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)

  const { data: docs = [], isLoading: docsLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    refetchInterval: (query) => {
      const data = query.state.data as DocumentInfo[] | undefined
      return data?.some((d) => d.processing) ? 2000 : false
    },
  })

  const invalidateDocs = () => qc.invalidateQueries({ queryKey: ['documents'] })

  const uploadMut = useMutation({
    mutationFn: uploadDocument,
    onSuccess: invalidateDocs,
  })
  const vectorizeMut = useMutation({
    mutationFn: () => vectorizeDocuments(selEmb, undefined, chunkSize, overlap),
    onSuccess: () => {
      // Start polling
      invalidateDocs()
    },
  })
  const deleteMut = useMutation({ mutationFn: deleteDocument, onSuccess: invalidateDocs })

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files) return
    setUploading(true)
    for (const f of Array.from(files)) await uploadMut.mutateAsync(f)
    setUploading(false)
  }, [uploadMut])

  const anyProcessing = docs.some((d: DocumentInfo) => d.processing)

  // Docs that don't yet have all selected models vectorized
  const unvectorized = docs.filter((d: DocumentInfo) =>
    !d.processing && selEmb.some((eid) => !(eid in d.processed_embeddings))
  )
  const canVectorize = selEmb.length > 0 && unvectorized.length > 0

  // ── Test questions ────────────────────────────────────────────────────────────
  const [qCat, setQCat] = useState<string | null>(null)
  const [showQForm, setShowQForm] = useState(false)
  const [qForm, setQForm] = useState({ category: '배송', question: '' })

  const { data: questions = [], isLoading: qLoading } = useQuery({
    queryKey: ['questions', qCat],
    queryFn: () => listQuestions(qCat ?? undefined),
  })
  const createQMut = useMutation({
    mutationFn: createQuestion,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['questions'] })
      setShowQForm(false)
      setQForm({ category: '배송', question: '' })
    },
  })
  const deleteQMut = useMutation({
    mutationFn: deleteQuestion,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['questions'] }),
  })

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-5">

      {/* ── Top row: Documents + Embedding models ───────────────────────────── */}
      <div className="grid grid-cols-5 gap-5">

        {/* Documents (wider) */}
        <div className="col-span-3 bg-white rounded-xl border border-gray-200 flex flex-col">
          <div className="px-5 py-4 border-b border-gray-100">
            <h2 className="font-semibold text-gray-800 text-sm">문서</h2>
            <p className="text-xs text-gray-400 mt-0.5">PDF, DOCX — 업로드 후 오른쪽에서 임베딩 모델을 선택해 벡터화하세요</p>
          </div>

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
            onClick={() => !uploading && fileRef.current?.click()}
            className={`mx-4 mt-4 border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-all
              ${dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'}`}
          >
            <input ref={fileRef} type="file" accept=".pdf,.docx,.doc" multiple className="hidden"
              onChange={(e) => handleFiles(e.target.files)} />
            {uploading ? (
              <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
                <LoadingSpinner size="sm" /> 업로드 중...
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                클릭 또는 드래그하여 업로드
                <span className="block text-xs text-gray-400 mt-0.5">PDF, DOCX</span>
              </p>
            )}
          </div>

          {/* Document list */}
          <div className="flex-1 overflow-auto px-4 py-3 space-y-2">
            {docsLoading ? (
              <div className="flex justify-center py-6"><LoadingSpinner /></div>
            ) : docs.length === 0 ? (
              <p className="text-center text-gray-400 text-sm py-6">문서가 없습니다</p>
            ) : (
              docs.map((doc: DocumentInfo) => (
                <DocRow
                  key={doc.id}
                  doc={doc}
                  selEmb={selEmb}
                  embModels={embModels}
                  onDelete={() => { if (confirm(`"${doc.filename}" 삭제?`)) deleteMut.mutate(doc.id) }}
                />
              ))
            )}
          </div>
        </div>

        {/* Embedding models (narrower) */}
        <div className="col-span-2 bg-white rounded-xl border border-gray-200 flex flex-col">
          <div className="px-5 py-4 border-b border-gray-100">
            <h2 className="font-semibold text-gray-800 text-sm">임베딩 모델</h2>
            <p className="text-xs text-gray-400 mt-0.5">선택 후 아래 버튼으로 벡터화 시작</p>
          </div>

          <div className="p-4 space-y-2 flex-1">
            {embModels.map((m: EmbeddingModelInfo) => {
              const checked = selEmb.includes(m.id)
              return (
                <label
                  key={m.id}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all select-none
                    ${!m.available ? 'opacity-40 cursor-not-allowed border-gray-100 bg-gray-50' :
                      checked ? 'border-indigo-300 bg-indigo-50' : 'border-gray-200 hover:border-gray-300'}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!m.available}
                    onChange={() => m.available && toggleEmb(m.id)}
                    className="mt-0.5 accent-indigo-600 shrink-0"
                  />
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-gray-800 leading-tight">{m.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">
                      {m.dimensions}d
                      {!m.available && <span className="ml-1 text-amber-500">· 사용 불가</span>}
                    </div>
                  </div>
                </label>
              )
            })}
          </div>

          {/* Chunking strategy */}
          <div className="px-4 py-3 border-t border-gray-100 space-y-3">
            <div>
              <p className="text-xs font-medium text-gray-500 mb-1.5">청킹 전략</p>
              <div className="flex gap-1">
                {CHUNK_PRESETS.map((p) => (
                  <button
                    key={p.label}
                    onClick={() => { setChunkSize(p.chunkSize); setOverlap(p.overlap) }}
                    title={p.desc}
                    className={`flex-1 py-1 rounded text-xs font-medium border transition-colors
                      ${chunkSize === p.chunkSize && overlap === p.overlap
                        ? 'bg-indigo-600 text-white border-indigo-600'
                        : 'border-gray-200 text-gray-600 hover:border-indigo-300 hover:bg-indigo-50'}`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-400 flex justify-between">
                  <span>청크 크기</span>
                  <span className="font-bold text-indigo-600">{chunkSize}자</span>
                </label>
                <input type="range" min={100} max={1200} step={50} value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="w-full accent-indigo-600 mt-1" />
              </div>
              <div>
                <label className="text-xs text-gray-400 flex justify-between">
                  <span>오버랩</span>
                  <span className="font-bold text-indigo-600">{overlap}자</span>
                </label>
                <input type="range" min={0} max={300} step={25} value={overlap}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                  className="w-full accent-indigo-600 mt-1" />
              </div>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              청크 크기↑ → 문맥 풍부, 노이즈↑ &nbsp;|&nbsp; 오버랩↑ → 경계 손실 방지, 청크 수↑
            </p>
          </div>

          {/* Vectorize button */}
          <div className="px-4 py-4 border-t border-gray-100">
            {anyProcessing ? (
              <div className="flex items-center justify-center gap-2 py-2 text-sm text-indigo-500">
                <LoadingSpinner size="sm" /> 벡터화 중...
              </div>
            ) : (
              <button
                onClick={() => vectorizeMut.mutate()}
                disabled={!canVectorize || vectorizeMut.isPending}
                className="w-full py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {vectorizeMut.isPending ? '시작 중...' : `벡터화 시작 (${selEmb.length}개 모델)`}
              </button>
            )}
            {!canVectorize && !anyProcessing && (
              <p className="text-xs text-gray-400 text-center mt-2">
                {selEmb.length === 0
                  ? '임베딩 모델을 선택해주세요'
                  : docs.length === 0
                    ? '문서를 먼저 업로드해주세요'
                    : '모든 문서가 이미 처리됨'}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* ── Test questions ────────────────────────────────────────────────────── */}
      <div className="bg-white rounded-xl border border-gray-200">
        <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-gray-800 text-sm">테스트 질문</h2>
            <p className="text-xs text-gray-400 mt-0.5">{questions.length}개</p>
          </div>
          <button
            onClick={() => setShowQForm((v) => !v)}
            className="px-3 py-1.5 bg-gray-100 text-gray-700 text-xs font-medium rounded-lg hover:bg-gray-200 transition-colors"
          >
            {showQForm ? '취소' : '+ 추가'}
          </button>
        </div>

        {showQForm && (
          <form
            onSubmit={(e) => {
              e.preventDefault()
              createQMut.mutate({ category: qForm.category, question: qForm.question, expected_topics: [], reference_answer: null })
            }}
            className="px-5 py-4 border-b border-gray-100 bg-gray-50 flex gap-3 items-end"
          >
            <div>
              <label className="block text-xs text-gray-500 mb-1">카테고리</label>
              <select value={qForm.category} onChange={(e) => setQForm({ ...qForm, category: e.target.value })}
                className="border border-gray-200 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white">
                {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-xs text-gray-500 mb-1">질문 내용</label>
              <input
                required
                value={qForm.question}
                onChange={(e) => setQForm({ ...qForm, question: e.target.value })}
                placeholder="고객 질문을 입력하세요..."
                className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <button type="submit" disabled={createQMut.isPending}
              className="px-4 py-1.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 shrink-0">
              추가
            </button>
          </form>
        )}

        <div className="px-5 py-3 flex gap-1.5 flex-wrap border-b border-gray-100">
          {[{ label: '전체', val: null }, ...CATEGORIES.map((c) => ({ label: c, val: c }))].map(({ label, val }) => (
            <button key={label} onClick={() => setQCat(val)}
              className={`px-2.5 py-1 rounded-full text-xs transition-colors
                ${qCat === val ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
              {label}
            </button>
          ))}
        </div>

        {qLoading ? (
          <div className="flex justify-center py-8"><LoadingSpinner /></div>
        ) : questions.length === 0 ? (
          <p className="text-center text-gray-400 text-sm py-8">질문이 없습니다</p>
        ) : (
          <ul className="divide-y divide-gray-50">
            {questions.map((q: TestQuestion) => (
              <li key={q.id} className="px-5 py-3 flex items-start gap-3">
                <span className="shrink-0 px-2 py-0.5 bg-gray-100 text-gray-500 rounded text-xs mt-0.5">
                  {q.category}
                </span>
                <p className="flex-1 text-sm text-gray-700">{q.question}</p>
                <button
                  onClick={() => { if (confirm('삭제할까요?')) deleteQMut.mutate(q.id) }}
                  className="shrink-0 p-1 text-gray-300 hover:text-red-400 rounded transition-colors"
                >
                  <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                    <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

// ── Document row ──────────────────────────────────────────────────────────────

function DocRow({ doc, selEmb, embModels, onDelete }: {
  doc: DocumentInfo
  selEmb: string[]
  embModels: EmbeddingModelInfo[]
  onDelete: () => void
}) {
  const shortName = (id: string) =>
    id.replace('ollama:', '').replace('local:', '').replace('openai:', '')
      .split(':')[0].replace('all-MiniLM-L6-v2', 'MiniLM')

  const processedIds = Object.keys(doc.processed_embeddings)
  const pendingEmbs = selEmb.filter((eid) => !processedIds.includes(eid))

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg border border-gray-100 bg-gray-50">
      <div className={`shrink-0 w-8 h-8 rounded-md flex items-center justify-center text-xs font-bold
        ${doc.file_type === 'pdf' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>
        {doc.file_type.toUpperCase()}
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-800 truncate">{doc.filename}</div>
        <div className="text-xs text-gray-400">{formatBytes(doc.size_bytes)} · {formatDate(doc.uploaded_at)}</div>

        {doc.processing ? (
          <div className="flex items-center gap-1 mt-1.5 text-xs text-indigo-500">
            <LoadingSpinner size="sm" /> 벡터화 중...
          </div>
        ) : doc.processing_error ? (
          <div className="mt-1 text-xs text-red-500">{doc.processing_error}</div>
        ) : (
          <div className="flex gap-1 mt-1.5 flex-wrap">
            {processedIds.length === 0 && Object.keys(doc.embed_errors ?? {}).length === 0 ? (
              <span className="text-xs text-gray-400">벡터화 안 됨</span>
            ) : (
              <>
                {processedIds.map((eid) => (
                  <span key={eid}
                    className={`px-1.5 py-0.5 rounded text-xs border
                      ${selEmb.includes(eid)
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-gray-100 text-gray-500 border-gray-200'}`}>
                    {shortName(eid)} · {doc.processed_embeddings[eid]}청크
                  </span>
                ))}
                {Object.entries(doc.embed_errors ?? {}).map(([eid, err]) => (
                  <span key={eid} title={err}
                    className="px-1.5 py-0.5 rounded text-xs border border-red-200 bg-red-50 text-red-600 cursor-help">
                    {shortName(eid)} · 실패
                  </span>
                ))}
              </>
            )}
            {pendingEmbs.length > 0 && !doc.processing && (
              <span className="px-1.5 py-0.5 rounded text-xs border border-dashed border-amber-300 text-amber-500 bg-amber-50">
                +{pendingEmbs.length}개 대기
              </span>
            )}
          </div>
        )}
      </div>

      <button onClick={onDelete}
        className="shrink-0 p-1.5 text-gray-300 hover:text-red-400 rounded transition-colors">
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
          <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
        </svg>
      </button>
    </div>
  )
}
