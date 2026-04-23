import { useCallback, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listDocuments, uploadDocument, vectorizeDocuments, deleteDocument,
  listEmbeddingModels, listParsers, getDocumentText,
} from '../lib/api'
import type { DocumentInfo, EmbeddingModelInfo, ParserInfo } from '../types'
import { formatBytes, formatDate } from '../lib/utils'
import LoadingSpinner from '../components/common/LoadingSpinner'

const CHUNK_PRESETS = [
  { label: '소형 300/50',   chunkSize: 300,  overlap: 50  },
  { label: '표준 600/100',  chunkSize: 600,  overlap: 100 },
  { label: '대형 1000/150', chunkSize: 1000, overlap: 150 },
]

const STORAGE_KEY_EMB    = 'docs_sel_emb'
const STORAGE_KEY_PARSER = 'docs_sel_parser'

function loadSaved<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) ?? '') ?? fallback } catch { return fallback }
}

export default function DocumentsPage() {
  const qc = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver]   = useState(false)
  const [uploading, setUploading] = useState(false)

  // ── Config state (persisted) ──────────────────────────────────────────────
  const [selPdfParser, setSelPdfParser]   = useState<string>(() => loadSaved(STORAGE_KEY_PARSER, 'pdfplumber'))
  const [selDocxParser, setSelDocxParser] = useState<string>('python-docx')
  const [selEmb, setSelEmb]               = useState<string[]>(() => loadSaved(STORAGE_KEY_EMB, []))
  const [chunkSize, setChunkSize] = useState(600)
  const [overlap, setOverlap]     = useState(100)

  // ── OCR comparison state ──────────────────────────────────────────────────
  const [cmpDocId, setCmpDocId]       = useState<string | null>(null)
  const [cmpParsers, setCmpParsers]   = useState<string[]>([])

  // ── Queries ───────────────────────────────────────────────────────────────
  const { data: docs = [], isLoading: docsLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    refetchInterval: (q) =>
      (q.state.data as DocumentInfo[] | undefined)?.some((d) => d.processing) ? 2000 : false,
  })
  const { data: embModels = [] } = useQuery({ queryKey: ['embeddingModels'], queryFn: listEmbeddingModels })
  const { data: parsers = [] }   = useQuery({ queryKey: ['parsers'], queryFn: listParsers })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['documents'] })

  const uploadMut    = useMutation({ mutationFn: uploadDocument, onSuccess: invalidate })
  const vectorizeMut = useMutation({
    mutationFn: () => vectorizeDocuments(selEmb, undefined, selPdfParser, chunkSize, overlap, 'paragraph', selPdfParser, selDocxParser),
    onSuccess: invalidate,
  })
  const deleteMut = useMutation({ mutationFn: deleteDocument, onSuccess: invalidate })

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files) return
    setUploading(true)
    for (const f of Array.from(files)) await uploadMut.mutateAsync(f)
    setUploading(false)
  }, [uploadMut])

  const toggleEmb = (id: string) => {
    const next = selEmb.includes(id) ? selEmb.filter((x) => x !== id) : [...selEmb, id]
    setSelEmb(next)
    localStorage.setItem(STORAGE_KEY_EMB, JSON.stringify(next))
  }
  const handlePdfParserChange = (id: string) => {
    setSelPdfParser(id)
    localStorage.setItem(STORAGE_KEY_PARSER, JSON.stringify(id))
  }

  const anyProcessing = docs.some((d: DocumentInfo) => d.processing)
  const unvectorized  = docs.filter((d: DocumentInfo) =>
    !d.processing && selEmb.some((eid) => !(eid in d.processed_embeddings))
  )
  const canVectorize = selEmb.length > 0 && unvectorized.length > 0

  const handleOpenComparison = (docId: string) => {
    setCmpDocId(docId)
    // default: compare all available parsers
    setCmpParsers(parsers.filter((p: ParserInfo) => p.available).map((p: ParserInfo) => p.id))
  }

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">문서 파싱 & 임베딩</h1>
        <p className="text-slate-400 text-sm mt-1">
          문서를 업로드하고 파서와 임베딩 모델을 선택한 뒤 벡터화합니다.
        </p>
      </div>

      {/* ── Top grid: upload / parser / embedding ───────────────────────── */}
      <div className="grid grid-cols-12 gap-5">

        {/* Upload zone (5 cols) */}
        <div className="col-span-5 flex flex-col gap-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
            onClick={() => !uploading && fileRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all
              ${dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'}`}
          >
            <input ref={fileRef} type="file" accept=".pdf,.docx,.doc" multiple className="hidden"
              onChange={(e) => handleFiles(e.target.files)} />
            {uploading ? (
              <div className="flex flex-col items-center gap-2">
                <LoadingSpinner />
                <span className="text-slate-400 text-sm">업로드 중...</span>
              </div>
            ) : (
              <>
                <div className="w-10 h-10 bg-slate-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-slate-400">
                    <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-slate-600 text-sm font-medium">클릭하거나 파일을 드래그</p>
                <p className="text-slate-400 text-xs mt-0.5">PDF, DOCX · 여러 파일 동시 가능</p>
              </>
            )}
          </div>

          {anyProcessing && (
            <div className="flex items-center gap-2 px-4 py-3 bg-indigo-50 border border-indigo-200 rounded-xl text-indigo-700 text-sm">
              <LoadingSpinner size="sm" /> 벡터화 처리 중...
            </div>
          )}
        </div>

        {/* Parser selection (3 cols) */}
        <div className="col-span-3 bg-white border border-slate-200 rounded-xl p-4 flex flex-col gap-4">
          {parsers.length === 0 ? <LoadingSpinner size="sm" /> : (
            <>
              {/* PDF parsers */}
              <div>
                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">PDF 파서</h2>
                <div className="space-y-1.5">
                  {(parsers as ParserInfo[]).filter((p) => p.file_types.includes('pdf')).map((p) => (
                    <label key={p.id}
                      className={`flex items-start gap-2.5 p-2 rounded-lg border cursor-pointer transition-all select-none
                        ${!p.available ? 'opacity-40 cursor-not-allowed border-slate-100 bg-slate-50' :
                          selPdfParser === p.id ? 'border-indigo-300 bg-indigo-50' : 'border-slate-200 hover:border-slate-300'}`}
                    >
                      <input type="radio" name="pdf-parser" value={p.id}
                        checked={selPdfParser === p.id} disabled={!p.available}
                        onChange={() => p.available && handlePdfParserChange(p.id)}
                        className="mt-0.5 accent-indigo-600 shrink-0" />
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-slate-800 leading-tight">{p.name}</div>
                        <div className="text-xs text-slate-400 mt-0.5 leading-snug">{p.description}</div>
                        {!p.available && <div className="text-xs text-amber-500 mt-1">설치 필요</div>}
                      </div>
                    </label>
                  ))}
                </div>
              </div>
              {/* DOCX parsers */}
              <div>
                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">DOCX 파서</h2>
                <div className="space-y-1.5">
                  {(parsers as ParserInfo[]).filter((p) => p.file_types.includes('docx')).map((p) => (
                    <label key={p.id}
                      className={`flex items-start gap-2.5 p-2 rounded-lg border cursor-pointer transition-all select-none
                        ${!p.available ? 'opacity-40 cursor-not-allowed border-slate-100 bg-slate-50' :
                          selDocxParser === p.id ? 'border-indigo-300 bg-indigo-50' : 'border-slate-200 hover:border-slate-300'}`}
                    >
                      <input type="radio" name="docx-parser" value={p.id}
                        checked={selDocxParser === p.id} disabled={!p.available}
                        onChange={() => p.available && setSelDocxParser(p.id)}
                        className="mt-0.5 accent-indigo-600 shrink-0" />
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-slate-800 leading-tight">{p.name}</div>
                        <div className="text-xs text-slate-400 mt-0.5 leading-snug">{p.description}</div>
                        {!p.available && <div className="text-xs text-amber-500 mt-1">설치 필요</div>}
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Embedding model + chunking + vectorize (4 cols) */}
        <div className="col-span-4 bg-white border border-slate-200 rounded-xl p-4 flex flex-col gap-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-700 mb-2">임베딩 모델</h2>
            <div className="space-y-1.5">
              {embModels.map((m: EmbeddingModelInfo) => {
                const checked = selEmb.includes(m.id)
                return (
                  <label key={m.id}
                    className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer transition-all select-none text-sm
                      ${!m.available ? 'opacity-40 cursor-not-allowed border-slate-100 bg-slate-50' :
                        checked ? 'border-indigo-300 bg-indigo-50' : 'border-slate-200 hover:border-slate-300'}`}
                  >
                    <input type="checkbox" checked={checked} disabled={!m.available}
                      onChange={() => m.available && toggleEmb(m.id)}
                      className="accent-indigo-600 shrink-0" />
                    <span className="font-medium text-slate-700 truncate">{m.name}</span>
                    <span className="text-xs text-slate-400 ml-auto shrink-0">{m.dimensions}d</span>
                  </label>
                )
              })}
            </div>
          </div>

          {/* Chunking */}
          <div>
            <h2 className="text-sm font-semibold text-slate-700 mb-2">청킹 전략</h2>
            <div className="flex gap-1 mb-2">
              {CHUNK_PRESETS.map((p) => (
                <button key={p.label}
                  onClick={() => { setChunkSize(p.chunkSize); setOverlap(p.overlap) }}
                  className={`flex-1 py-1 rounded text-xs font-medium border transition-colors
                    ${chunkSize === p.chunkSize && overlap === p.overlap
                      ? 'bg-indigo-600 text-white border-indigo-600'
                      : 'border-slate-200 text-slate-600 hover:border-indigo-300 hover:bg-indigo-50'}`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-slate-400 flex justify-between mb-1">
                  <span>청크 크기</span>
                  <span className="font-bold text-indigo-600">{chunkSize}자</span>
                </label>
                <input type="range" min={100} max={1200} step={50} value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="w-full accent-indigo-600" />
              </div>
              <div>
                <label className="text-xs text-slate-400 flex justify-between mb-1">
                  <span>오버랩</span>
                  <span className="font-bold text-indigo-600">{overlap}자</span>
                </label>
                <input type="range" min={0} max={300} step={25} value={overlap}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                  className="w-full accent-indigo-600" />
              </div>
            </div>
          </div>

          {/* Vectorize button */}
          <div className="mt-auto">
            {anyProcessing ? (
              <div className="flex items-center justify-center gap-2 py-2 text-sm text-indigo-500">
                <LoadingSpinner size="sm" /> 벡터화 중...
              </div>
            ) : (
              <button
                onClick={() => vectorizeMut.mutate()}
                disabled={!canVectorize || vectorizeMut.isPending}
                className="w-full py-2 bg-indigo-600 text-white text-sm font-semibold rounded-lg hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {vectorizeMut.isPending
                  ? '시작 중...'
                  : canVectorize
                    ? `벡터화 시작 (${selEmb.length}개 모델)`
                    : selEmb.length === 0 ? '임베딩 모델을 선택해주세요'
                    : docs.length === 0 ? '문서를 먼저 업로드해주세요'
                    : '모든 문서가 이미 처리됨'}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Document list ────────────────────────────────────────────────── */}
      <div className="bg-white border border-slate-200 rounded-xl">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-semibold text-slate-800 text-sm">문서 목록 ({docs.length})</h2>
          <span className="text-xs text-slate-400">클릭하면 파서별 OCR 결과를 비교합니다</span>
        </div>
        {docsLoading ? (
          <div className="flex justify-center py-10"><LoadingSpinner /></div>
        ) : docs.length === 0 ? (
          <p className="text-center text-slate-400 text-sm py-10">업로드된 문서가 없습니다</p>
        ) : (
          <ul className="divide-y divide-slate-50">
            {docs.map((doc: DocumentInfo) => (
              <DocRow
                key={doc.id}
                doc={doc}
                selEmb={selEmb}
                embModels={embModels}
                active={cmpDocId === doc.id}
                onClick={() => handleOpenComparison(doc.id)}
                onDelete={() => { if (confirm(`"${doc.filename}" 삭제?`)) deleteMut.mutate(doc.id) }}
              />
            ))}
          </ul>
        )}
      </div>

      {/* ── OCR comparison panel ─────────────────────────────────────────── */}
      {cmpDocId && (
        <OcrComparisonPanel
          docId={cmpDocId}
          docName={docs.find((d: DocumentInfo) => d.id === cmpDocId)?.filename ?? ''}
          parsers={parsers}
          activeParsers={cmpParsers}
          onToggleParser={(id) =>
            setCmpParsers((prev) =>
              prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
            )
          }
          chunkSize={chunkSize}
          overlap={overlap}
          onClose={() => setCmpDocId(null)}
        />
      )}
    </div>
  )
}

// ── Document row ──────────────────────────────────────────────────────────────

function DocRow({ doc, selEmb, embModels, active, onClick, onDelete }: {
  doc: DocumentInfo
  selEmb: string[]
  embModels: EmbeddingModelInfo[]
  active: boolean
  onClick: () => void
  onDelete: () => void
}) {
  const shortName = (id: string) =>
    id.replace(/^(ollama|local|openai):/, '').split(':')[0]
      .replace('all-MiniLM-L6-v2', 'MiniLM')

  const processedIds = Object.keys(doc.processed_embeddings)

  return (
    <li
      onClick={onClick}
      className={`px-5 py-3.5 flex items-start gap-3 cursor-pointer transition-colors
        ${active ? 'bg-indigo-50 border-l-2 border-l-indigo-500' : 'hover:bg-slate-50'}`}
    >
      <div className={`shrink-0 w-8 h-8 rounded-md flex items-center justify-center text-xs font-bold
        ${doc.file_type === 'pdf' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>
        {doc.file_type.toUpperCase()}
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-slate-800 truncate">{doc.filename}</div>
        <div className="text-xs text-slate-400 mt-0.5">{formatBytes(doc.size_bytes)} · {formatDate(doc.uploaded_at)}</div>

        {doc.processing ? (
          <div className="flex items-center gap-1 mt-1.5 text-xs text-indigo-500">
            <LoadingSpinner size="sm" /> 벡터화 중...
          </div>
        ) : doc.processing_error ? (
          <div className="mt-1 text-xs text-red-500">{doc.processing_error}</div>
        ) : (
          <div className="flex gap-1 mt-1.5 flex-wrap">
            {processedIds.length === 0 && Object.keys(doc.embed_errors ?? {}).length === 0 ? (
              <span className="text-xs text-slate-400">벡터화 안 됨</span>
            ) : (
              <>
                {processedIds.map((eid) => (
                  <span key={eid}
                    className={`px-1.5 py-0.5 rounded text-xs border
                      ${selEmb.includes(eid)
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-slate-100 text-slate-500 border-slate-200'}`}>
                    {shortName(eid)} · {doc.processed_embeddings[eid]}청크
                  </span>
                ))}
                {Object.entries(doc.embed_errors ?? {}).map(([eid, err]) => (
                  <span key={eid} title={err}
                    className="px-1.5 py-0.5 rounded text-xs border border-red-200 bg-red-50 text-red-600 cursor-help">
                    {shortName(eid)} · 실패
                  </span>
                ))}
                {selEmb
                  .filter((eid) => !processedIds.includes(eid) && !(eid in (doc.embed_errors ?? {})))
                  .length > 0 && (
                  <span className="px-1.5 py-0.5 rounded text-xs border border-dashed border-amber-300 text-amber-500 bg-amber-50">
                    +{selEmb.filter((eid) => !processedIds.includes(eid) && !(eid in (doc.embed_errors ?? {}))).length}개 대기
                  </span>
                )}
              </>
            )}
          </div>
        )}
      </div>

      <button
        onClick={(e) => { e.stopPropagation(); onDelete() }}
        className="shrink-0 p-1.5 text-slate-300 hover:text-red-400 rounded transition-colors"
      >
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
          <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
        </svg>
      </button>
    </li>
  )
}

// ── OCR comparison panel ──────────────────────────────────────────────────────

function OcrComparisonPanel({ docId, docName, parsers, activeParsers, onToggleParser, chunkSize, overlap, onClose }: {
  docId: string
  docName: string
  parsers: ParserInfo[]
  activeParsers: string[]
  onToggleParser: (id: string) => void
  chunkSize: number
  overlap: number
  onClose: () => void
}) {
  const [viewMode, setViewMode] = useState<'text' | 'chunks'>('text')

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-slate-100 flex items-center gap-3">
        <h3 className="font-semibold text-slate-800 text-sm flex-1 truncate">
          파서 비교 — {docName}
        </h3>
        {/* Parser toggles */}
        <div className="flex gap-1">
          {parsers.filter((p) => p.available).map((p) => (
            <button
              key={p.id}
              onClick={() => onToggleParser(p.id)}
              className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors
                ${activeParsers.includes(p.id)
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'border-slate-200 text-slate-500 hover:border-indigo-300'}`}
            >
              {p.name}
            </button>
          ))}
        </div>
        {/* View mode */}
        <div className="flex gap-0.5 bg-slate-100 rounded-lg p-0.5">
          {(['text', 'chunks'] as const).map((m) => (
            <button key={m} onClick={() => setViewMode(m)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors
                ${viewMode === m ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>
              {m === 'text' ? '원문' : '청크'}
            </button>
          ))}
        </div>
        <button onClick={onClose}
          className="p-1 text-slate-400 hover:text-slate-600 rounded transition-colors">
          <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
            <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
          </svg>
        </button>
      </div>

      {/* Columns */}
      {activeParsers.length === 0 ? (
        <p className="text-center text-slate-400 text-sm py-10">위에서 비교할 파서를 선택하세요</p>
      ) : (
        <div className={`grid divide-x divide-slate-100`}
          style={{ gridTemplateColumns: `repeat(${activeParsers.length}, 1fr)` }}>
          {activeParsers.map((parserId) => (
            <ParserColumn
              key={parserId}
              docId={docId}
              parserId={parserId}
              parserName={parsers.find((p) => p.id === parserId)?.name ?? parserId}
              chunkSize={chunkSize}
              overlap={overlap}
              viewMode={viewMode}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Single parser column ──────────────────────────────────────────────────────

function ParserColumn({ docId, parserId, parserName, chunkSize, overlap, viewMode }: {
  docId: string
  parserId: string
  parserName: string
  chunkSize: number
  overlap: number
  viewMode: 'text' | 'chunks'
}) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['docText', docId, parserId, chunkSize, overlap],
    queryFn: () => getDocumentText(docId, parserId, chunkSize, overlap),
    staleTime: 60_000,
  })

  return (
    <div className="flex flex-col min-h-0">
      {/* Column header with stats */}
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
        <div className="text-xs font-semibold text-slate-600 mb-2">{parserName}</div>
        {data && (
          <div className="flex gap-3 text-xs text-slate-500">
            <span><b className="text-slate-700">{data.char_count.toLocaleString()}</b>자</span>
            <span><b className="text-slate-700">{data.chunk_count}</b>청크</span>
            <span>평균 <b className="text-slate-700">{data.avg_chunk_size}</b>자</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4 overflow-auto max-h-96 text-xs text-slate-700 leading-relaxed">
        {isLoading && (
          <div className="flex flex-col items-center gap-2 py-8 text-slate-400">
            <LoadingSpinner />
            <span>추출 중...</span>
          </div>
        )}
        {isError && (
          <div className="text-red-500 text-xs py-4">
            오류: {String((error as Error)?.message ?? error)}
          </div>
        )}
        {data && viewMode === 'text' && (
          <pre className="whitespace-pre-wrap font-sans">{data.text_preview}</pre>
        )}
        {data && viewMode === 'chunks' && (
          <div className="space-y-2">
            {data.chunks.map((c) => (
              <div key={c.index} className="border border-slate-100 rounded p-2 bg-slate-50">
                <div className="text-xs text-slate-400 mb-1">#{c.index + 1} · {c.char_count}자</div>
                <p className="whitespace-pre-wrap">{c.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
