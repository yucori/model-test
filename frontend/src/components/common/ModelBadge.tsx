import type { LLMProvider } from '../../types'

const LLM_STYLES: Record<LLMProvider, string> = {
  anthropic: 'bg-orange-50 text-orange-700 ring-orange-200',
  openai: 'bg-sky-50 text-sky-700 ring-sky-200',
  ollama: 'bg-violet-50 text-violet-700 ring-violet-200',
  google: 'bg-green-50 text-green-700 ring-green-200',
}

const LLM_LABELS: Record<LLMProvider, string> = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  ollama: 'Ollama',
  google: 'Google',
}

const EMB_STYLES: Record<string, string> = {
  local: 'bg-teal-50 text-teal-700 ring-teal-200',
  openai: 'bg-sky-50 text-sky-700 ring-sky-200',
  ollama: 'bg-violet-50 text-violet-700 ring-violet-200',
}

export function LLMBadge({ provider }: { provider: LLMProvider }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ring-1 ${LLM_STYLES[provider]}`}>
      {LLM_LABELS[provider]}
    </span>
  )
}

export function EmbeddingBadge({ provider }: { provider: string }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ring-1 ${EMB_STYLES[provider] ?? 'bg-slate-50 text-slate-600 ring-slate-200'}`}>
      {provider === 'local' ? '로컬' : provider}
    </span>
  )
}
