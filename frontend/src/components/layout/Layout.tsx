import { NavLink, Outlet } from 'react-router-dom'

const TABS = [
  { to: '/', label: '준비', end: true },
  { to: '/run', label: '실행' },
  { to: '/results', label: '결과' },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Top nav */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 h-12 flex items-center gap-8">
          <span className="text-sm font-semibold text-gray-800 tracking-tight select-none">
            RAG Bench
          </span>
          <nav className="flex gap-1">
            {TABS.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-gray-100 text-gray-900'
                      : 'text-gray-500 hover:text-gray-800 hover:bg-gray-50'
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
