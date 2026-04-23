import { NavLink, Outlet } from 'react-router-dom'

const NAV = [
  { to: '/',               label: '파이프라인',    end: true },
  { to: '/embedding-eval', label: '임베딩 비교',   end: false },
  { to: '/results',        label: '결과 분석',     end: false },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="max-w-5xl mx-auto px-6 h-12 flex items-center gap-8">
          <span className="text-sm font-bold text-slate-800 tracking-tight select-none">
            RAG Bench
          </span>
          <nav className="flex gap-1">
            {NAV.map(({ to, label, end }) => (
              <NavLink key={to} to={to} end={end}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-indigo-600 text-white'
                      : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  )
}
