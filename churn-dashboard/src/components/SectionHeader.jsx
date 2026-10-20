export default function SectionHeader({ title, subtitle, breadcrumbs = [], actions }) {
  return (
    <div className="mb-6 rounded-3xl border border-slate-200 bg-slate-50/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">
            {breadcrumbs.map((crumb, index) => (
              <span key={crumb} className="inline-flex items-center gap-2">
                {index > 0 && <span>›</span>}
                <span>{crumb}</span>
              </span>
            ))}
          </div>
          <div>
            <h1 className="text-3xl font-semibold text-slate-950 dark:text-white">{title}</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-400">{subtitle}</p>
          </div>
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-3">{actions}</div> : null}
      </div>
    </div>
  )
}
