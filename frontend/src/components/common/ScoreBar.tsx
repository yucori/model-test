function barColor(score: number) {
  if (score >= 8) return 'bg-emerald-500'
  if (score >= 6) return 'bg-amber-400'
  if (score >= 4) return 'bg-orange-400'
  return 'bg-red-400'
}

export default function ScoreBar({
  score,
  max = 10,
  label,
  showValue = true,
}: {
  score: number
  max?: number
  label?: string
  showValue?: boolean
}) {
  const pct = Math.min((score / max) * 100, 100)
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>{label}</span>
          {showValue && (
            <span className="font-semibold tabular-nums text-slate-700">
              {score.toFixed(1)}
            </span>
          )}
        </div>
      )}
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor(score)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
