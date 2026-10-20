export default function Sidebar({ items, activeTab, onTabChange, open, onToggle }) {
  return (
    <aside className={`fixed inset-y-0 left-0 z-20 flex w-full max-w-xs flex-col border-r border-slate-200/80 bg-slate-50/95 py-6 px-5 shadow-xl shadow-slate-900/5 backdrop-blur-xl dark:border-slate-800/80 dark:bg-slate-950/95 lg:static lg:z-auto lg:w-72 lg:shadow-none ${open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'} transition-transform duration-300`}>
      <div className="flex items-center justify-between gap-3 pb-6 lg:hidden">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Telecom Analytics</p>
          <h2 className="text-xl font-semibold text-slate-950 dark:text-white">Customer Retention</h2>
        </div>
        <button
          className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-slate-200 text-slate-600 transition hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          onClick={onToggle}
          aria-label="Close sidebar"
        >
          ×
        </button>
      </div>

      <div className="space-y-6">
        <div className="space-y-3 rounded-3xl border border-slate-200 bg-white/90 p-5 shadow-sm dark:border-slate-700 dark:bg-slate-950/80">
          <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Workspace</p>
          <div>
            <p className="text-lg font-semibold text-slate-950 dark:text-white">Customer Retention Intelligence</p>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Telecom churn analysis, customer segments, and model evaluation.</p>
          </div>
        </div>

        <div className="space-y-2">
          {items.map((item) => {
            const active = activeTab === item.key
            return (
              <button
                key={item.key}
                onClick={() => onTabChange(item.key)}
                className={`group flex w-full items-center justify-between rounded-3xl px-4 py-4 text-left transition ${active ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/10' : 'bg-white text-slate-700 hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800'}`}
              >
                <span className="flex items-center gap-3">
                  <span className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl ${active ? 'bg-white/15 text-white' : 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-200'}`}>
                    {item.icon}
                  </span>
                  <span className="flex flex-col">
                    <span className="font-semibold">{item.label}</span>
                    <span className="text-xs text-slate-500 dark:text-slate-400">{item.description}</span>
                  </span>
                </span>
                <span className={`inline-flex h-10 w-10 items-center justify-center rounded-full ${active ? 'bg-white/10 text-white' : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-300'}`}>
                  →
                </span>
              </button>
            )
          })}
        </div>

        <div className="rounded-3xl border border-slate-200 bg-slate-100/90 p-5 text-sm text-slate-700 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-300">
          <p className="uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Quick actions</p>
          <p className="mt-3 leading-6">Use this panel to jump to the churn insights, payments, contracts, or ML pages in one click.</p>
        </div>
      </div>
    </aside>
  )
}
