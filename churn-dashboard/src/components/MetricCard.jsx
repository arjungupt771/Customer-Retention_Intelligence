export default function MetricCard({ title, value, label, accent, children }) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md dark:border-slate-800 dark:bg-slate-950/80">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">{title}</p>
          <p className="mt-3 text-3xl font-semibold text-slate-950 dark:text-white">{value}</p>
        </div>
        {accent ? (
          <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200">{accent}</span>
        ) : null}
      </div>
      {label ? <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">{label}</p> : null}
      {children}
    </div>
  )
}
