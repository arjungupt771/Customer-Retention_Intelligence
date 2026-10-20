export default function LoadingSkeleton({ rows = 4 }) {
  return (
    <div className="space-y-4">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="animate-pulse rounded-3xl border border-slate-200 bg-slate-200/70 p-6 dark:border-slate-700 dark:bg-slate-800/80">
          <div className="h-5 w-3/4 rounded-full bg-slate-300 dark:bg-slate-700" />
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="h-28 rounded-3xl bg-slate-300 dark:bg-slate-700" />
            <div className="h-28 rounded-3xl bg-slate-300 dark:bg-slate-700" />
          </div>
        </div>
      ))}
    </div>
  )
}
