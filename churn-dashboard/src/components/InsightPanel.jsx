export default function InsightPanel({ insights }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {insights.map((item) => (
        <div key={item.title} className="rounded-3xl border border-slate-200 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
          <p className="text-sm font-semibold text-slate-950 dark:text-white">{item.title}</p>
          <p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-400">{item.description}</p>
        </div>
      ))}
    </div>
  )
}
