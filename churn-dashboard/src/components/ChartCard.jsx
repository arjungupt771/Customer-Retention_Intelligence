export default function ChartCard({ title, description, badge, children }) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">{title}</p>
          <p className="mt-2 text-lg font-semibold text-slate-950 dark:text-white">{description}</p>
        </div>
        {badge ? (
          <span className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-200">{badge}</span>
        ) : null}
      </div>
      <div className="mt-6 min-h-[260px]">{children}</div>
    </div>
  )
}
