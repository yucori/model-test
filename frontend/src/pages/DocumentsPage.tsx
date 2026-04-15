import { useCallback, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listDocuments, uploadDocument, reprocessDocument, deleteDocument, listEmbeddingModels } from '../lib/api'
import { formatBytes, formatDate } from '../lib/utils'
import LoadingSpinner from '../components/common/LoadingSpinner'
import type { DocumentInfo, EmbeddingModelInfo } from '../types'

export default function DocumentsPage() {
  const qc = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)

  const { data: docs = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    // Keep polling while any document is still processing
    refetchInterval: (query) => {
      const data = query.state.data as DocumentInfo[] | undefined
      return data?.some((d) => d.processing) ? 2000 : false
    },
  })

  const { data: embModels = [] } = useQuery({
    queryKey: ['embeddingModels'],
    queryFn: listEmbeddingModels,
  })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['documents'] })

  const uploadMut = useMutation({ mutationFn: uploadDocument, onSuccess: invalidate })
  const reprocessMut = useMutation({ mutationFn: reprocessDocument, onSuccess: invalidate })
  const deleteMut = useMutation({ mutationFn: deleteDocument, onSuccess: invalidate })

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files) return
    setUploading(true)
    for (const f of Array.from(files)) await uploadMut.mutateAsync(f)
    setUploading(false)
  }, [uploadMut])

  const anyProcessing = docs.some((d: DocumentInfo) => d.processing)

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-slate-900 mb-1">문서 관리</h1>
      <p className="text-slate-400 text-sm mb-8">
        파일을 업로드하면 사용 가능한 모든 임베딩 모델로 자동 벡터화됩니다.
      </p>

      {/* Upload zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
        onClick={() => fileRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all mb-6
          ${dragOver ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 hover:border-indigo-400 hover:bg-slate-50'}`}
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
            <div className="w-12 h-12 bg-slate-100 rounded-xl flex items-center justify-center mx-auto mb-3">
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6 text-slate-400">
                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="text-slate-600 font-medium">클릭하거나 파일을 드래그하여 업로드</p>
            <p className="text-slate-400 text-xs mt-1">PDF, DOCX · 여러 파일 동시 가능</p>
            <p className="text-slate-400 text-xs mt-0.5">
              업로드 즉시 {embModels.filter((m: EmbeddingModelInfo) => m.available).length}개 임베딩 모델로 자동 벡터화됩니다
            </p>
          </>
        )}
      </div>

      {/* Processing notice */}
      {anyProcessing && (
        <div className="flex items-center gap-3 p-4 bg-indigo-50 border border-indigo-200 rounded-xl mb-5">
          <LoadingSpinner size="sm" />
          <span className="text-indigo-700 text-sm">벡터화 처리 중... 자동으로 업데이트됩니다.</span>
        </div>
      )}

      {/* Document list */}
      <div className="bg-white border border-slate-200 rounded-2xl">
        <div className="px-6 py-4 border-b border-slate-100">
          <h2 className="font-semibold text-slate-800">업로드된 문서 ({docs.length})</h2>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12"><LoadingSpinner /></div>
        ) : docs.length === 0 ? (
          <div className="text-center text-slate-400 py-12 text-sm">문서가 없습니다. 위에서 업로드해주세요.</div>
        ) : (
          <ul className="divide-y divide-slate-50">
            {docs.map((doc: DocumentInfo) => (
              <DocumentRow
                key={doc.id}
                doc={doc}
                embModels={embModels}
                onReprocess={() => reprocessMut.mutate(doc.id)}
                onDelete={() => { if (confirm(`"${doc.filename}" 삭제?`)) deleteMut.mutate(doc.id) }}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function DocumentRow({ doc, embModels, onReprocess, onDelete }: {
  doc: DocumentInfo
  embModels: EmbeddingModelInfo[]
  onReprocess: () => void
  onDelete: () => void
}) {
  const available = embModels.filter((m) => m.available)
  const processedIds = Object.keys(doc.processed_embeddings)
  const allDone = available.length > 0 && available.every((m) => processedIds.includes(m.id))
  const hasNewModels = !doc.processing && available.some((m) => !processedIds.includes(m.id))

  return (
    <li className="px-6 py-5">
      <div className="flex items-start gap-4">
        {/* File type badge */}
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center text-xs font-bold shrink-0
          ${doc.file_type === 'pdf' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>
          {doc.file_type.toUpperCase()}
        </div>

        <div className="flex-1 min-w-0">
          <div className="font-medium text-slate-800 truncate">{doc.filename}</div>
          <div className="text-xs text-slate-400 mt-0.5">
            {formatBytes(doc.size_bytes)} · {formatDate(doc.uploaded_at)}
          </div>

          {/* Processing / error state */}
          {doc.processing && (
            <div className="flex items-center gap-1.5 mt-2 text-xs text-indigo-600">
              <LoadingSpinner size="sm" />
              <span>임베딩 모델 벡터화 중...</span>
            </div>
          )}
          {doc.processing_error && !doc.processing && (
            <div className="mt-2 text-xs text-red-500 bg-red-50 rounded px-2 py-1">
              오류: {doc.processing_error}
            </div>
          )}

          {/* Per-model embedding status */}
          {!doc.processing && (
            <div className="flex gap-2 mt-2 flex-wrap">
              {embModels.map((m: EmbeddingModelInfo) => {
                const chunks = doc.processed_embeddings[m.id]
                const done = chunks !== undefined
                return (
                  <div key={m.id} className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border
                    ${done ? 'bg-emerald-50 border-emerald-200 text-emerald-700' :
                      m.available ? 'bg-slate-50 border-slate-200 text-slate-400' :
                                    'bg-slate-50 border-slate-100 text-slate-300'}`}>
                    {done ? '✓' : '○'}
                    <span className="truncate max-w-28">
                      {m.id.replace('local:', '').replace('openai:', '')
                        .replace('ollama:', '').replace('all-MiniLM-L6-v2', 'MiniLM')
                        .split(':')[0]}
                    </span>
                    {done && <span className="font-medium">{chunks}청크</span>}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Re-process button — only shown if new embedding models appeared after upload */}
          {hasNewModels && (
            <button
              onClick={onReprocess}
              className="px-3 py-1.5 bg-slate-100 text-slate-600 text-xs font-medium rounded-lg hover:bg-indigo-50 hover:text-indigo-700 transition-colors"
              title="새로 추가된 임베딩 모델로 재처리"
            >
              재처리
            </button>
          )}
          {allDone && !hasNewModels && (
            <span className="text-xs text-emerald-600 font-medium">완료</span>
          )}
          <button
            onClick={onDelete}
            className="p-1.5 text-slate-300 hover:text-red-400 hover:bg-red-50 rounded-lg transition-colors"
          >
            <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
              <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
    </li>
  )
}
