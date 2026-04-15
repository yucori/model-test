import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listQuestions, createQuestion, deleteQuestion } from '../lib/api'
import type { TestQuestion } from '../types'
import LoadingSpinner from '../components/common/LoadingSpinner'

const CATEGORIES = ['배송', '반품/환불', '상품', '보관', '결제', '회원', '고객센터', '기타']

export default function TestSuitePage() {
  const qc = useQueryClient()
  const [selectedCat, setSelectedCat] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ category: '배송', question: '', topics: '' })

  const { data: questions = [], isLoading } = useQuery({
    queryKey: ['questions', selectedCat],
    queryFn: () => listQuestions(selectedCat ?? undefined),
  })
  const createMut = useMutation({
    mutationFn: createQuestion,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['questions'] })
      setShowForm(false)
      setForm({ category: '배송', question: '', topics: '' })
    },
  })
  const deleteMut = useMutation({
    mutationFn: deleteQuestion,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['questions'] }),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMut.mutate({
      category: form.category,
      question: form.question,
      expected_topics: form.topics.split(',').map((t) => t.trim()).filter(Boolean),
      reference_answer: null,
    })
  }

  const grouped = questions.reduce<Record<string, TestQuestion[]>>((acc, q) => {
    acc[q.category] = acc[q.category] ?? []
    acc[q.category].push(q)
    return acc
  }, {})

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-2xl font-bold text-slate-900">테스트 질문 관리</h1>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors"
        >
          + 질문 추가
        </button>
      </div>
      <p className="text-slate-400 text-sm mb-6">총 {questions.length}개 질문</p>

      {/* Add form */}
      {showForm && (
        <form onSubmit={handleSubmit}
          className="bg-white border border-slate-200 rounded-2xl p-6 mb-6">
          <h2 className="font-semibold text-slate-800 mb-4">새 질문 추가</h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">카테고리</label>
              <select value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
                {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">예상 주제 (쉼표 구분)</label>
              <input value={form.topics} onChange={(e) => setForm({ ...form, topics: e.target.value })}
                placeholder="배송 기간, 영업일..."
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" />
            </div>
          </div>
          <div className="mb-4">
            <label className="block text-xs font-medium text-slate-500 mb-1">질문 내용</label>
            <textarea required value={form.question}
              onChange={(e) => setForm({ ...form, question: e.target.value })}
              placeholder="고객이 물어볼 질문을 입력하세요..."
              rows={3}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500" />
          </div>
          <div className="flex gap-2">
            <button type="submit" disabled={createMut.isPending}
              className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50">
              {createMut.isPending ? '추가 중...' : '추가'}
            </button>
            <button type="button" onClick={() => setShowForm(false)}
              className="px-4 py-2 bg-slate-100 text-slate-600 text-sm rounded-lg hover:bg-slate-200">
              취소
            </button>
          </div>
        </form>
      )}

      {/* Category filter */}
      <div className="flex gap-2 flex-wrap mb-6">
        {[{ label: '전체', val: null }, ...CATEGORIES.map((c) => ({ label: c, val: c }))].map(({ label, val }) => (
          <button key={label} onClick={() => setSelectedCat(val)}
            className={`px-3 py-1 rounded-full text-sm transition-colors
              ${selectedCat === val ? 'bg-indigo-600 text-white' : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'}`}>
            {label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12"><LoadingSpinner /></div>
      ) : (
        <div className="space-y-4">
          {Object.entries(grouped).map(([cat, qs]) => (
            <div key={cat} className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
              <div className="px-6 py-3 bg-slate-50 border-b border-slate-100 flex items-center gap-2">
                <span className="font-semibold text-slate-700 text-sm">{cat}</span>
                <span className="text-xs text-slate-400 bg-white border border-slate-200 rounded-full px-2 py-0.5">
                  {qs.length}
                </span>
              </div>
              <ul className="divide-y divide-slate-50">
                {qs.map((q) => (
                  <li key={q.id} className="px-6 py-4 flex items-start gap-4">
                    <div className="flex-1">
                      <p className="text-sm text-slate-800">{q.question}</p>
                      {q.expected_topics.length > 0 && (
                        <div className="flex gap-1 mt-2 flex-wrap">
                          {q.expected_topics.map((t) => (
                            <span key={t} className="px-2 py-0.5 bg-slate-100 text-slate-500 rounded text-xs">{t}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => { if (confirm('이 질문을 삭제할까요?')) deleteMut.mutate(q.id) }}
                      className="shrink-0 p-1 text-slate-300 hover:text-red-400 transition-colors rounded"
                    >
                      <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
