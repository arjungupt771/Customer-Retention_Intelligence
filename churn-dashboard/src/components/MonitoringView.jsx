import { useEffect, useState } from 'react'
import axios from 'axios'
import { Activity, AlertTriangle, CheckCircle2, RefreshCw } from 'lucide-react'
import MetricCard from './MetricCard.jsx'

export default function MonitoringView() {
  const [driftReport, setDriftReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const fetchDrift = async () => {
    setRefreshing(true)
    try {
      const res = await axios.get('/api/monitoring/drift')
      setDriftReport(res.data)
    } catch (err) {
      console.error('Drift monitoring error:', err)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    let isMounted = true
    axios.get('/api/monitoring/drift')
      .then((res) => {
        if (isMounted) setDriftReport(res.data)
      })
      .catch((err) => console.error('Drift monitoring error:', err))
      .finally(() => {
        if (isMounted) setLoading(false)
      })
    return () => {
      isMounted = false
    }
  }, [])

  if (loading) {
    return (
      <div className="py-20 text-center text-sm text-slate-500 animate-pulse">
        Comparing inference cohorts against baseline reference profile…
      </div>
    )
  }

  const featureDrift = driftReport?.feature_drift || {}
  const predDrift = driftReport?.prediction_drift || {}
  const overall = driftReport?.overall_status || 'No Significant Drift'

  const overallBadgeColor =
    overall === 'High Drift'
      ? 'border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200'
      : overall === 'Moderate Drift'
      ? 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-200'
      : 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-200'

  return (
    <div className="space-y-8">
      {/* Overall Drift Status Banner */}
      <div className="flex flex-col gap-4 rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-wide ${overallBadgeColor}`}>
              {overall === 'No Significant Drift' ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
              {overall}
            </span>
            <h3 className="text-lg font-bold text-slate-950 dark:text-white">Dataset & Prediction Drift Monitor</h3>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Automated Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests comparing production inputs against reference baseline.
          </p>
        </div>

        <button
          onClick={fetchDrift}
          disabled={refreshing}
          className="inline-flex items-center gap-2 rounded-2xl bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-800 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Re-evaluating…' : 'Refresh Drift Status'}
        </button>
      </div>

      {/* KPI Cards: Drift Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Drifted Features"
          value={`${driftReport?.drifted_features_count ?? 0} / ${driftReport?.total_features ?? 19}`}
          caption={`${driftReport?.drifted_features_pct ?? 0}% feature shift`}
          icon={<Activity className="h-5 w-5 text-indigo-500" />}
        />
        <MetricCard
          label="Prediction Distribution PSI"
          value={predDrift.psi?.toFixed(3) ?? '—'}
          caption={`Severity: ${predDrift.severity ?? 'Low'}`}
          icon={<AlertTriangle className="h-5 w-5 text-amber-500" />}
        />
        <MetricCard
          label="Average Churn Probability"
          value={`${((predDrift.current_avg_probability ?? 0) * 100).toFixed(1)}%`}
          caption={`Baseline: ${((predDrift.reference_avg_probability ?? 0) * 100).toFixed(1)}%`}
          icon={<CheckCircle2 className="h-5 w-5 text-emerald-500" />}
        />
        <MetricCard
          label="High-Risk Cohort Proportion"
          value={`${predDrift.current_high_risk_pct ?? '—'}%`}
          caption={`Baseline: ${predDrift.reference_high_risk_pct ?? '—'}%`}
          icon={<Activity className="h-5 w-5 text-violet-500" />}
        />
      </div>

      {/* Feature Drift Detailed Table */}
      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
        <h4 className="text-base font-semibold text-slate-950 dark:text-white">Feature Stability & Drift Table</h4>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Green (&lt; 0.10) indicates stable distributions; Amber (0.10–0.20) indicates moderate shift; Red (&ge; 0.20) flags significant drift.
        </p>

        <div className="mt-5 overflow-x-auto">
          <table className="w-full text-left text-sm font-sans">
            <thead className="border-b border-slate-200 text-xs uppercase text-slate-400 dark:border-slate-800">
              <tr>
                <th className="pb-3 font-semibold">Feature</th>
                <th className="pb-3 font-semibold">Type</th>
                <th className="pb-3 font-semibold">PSI Score</th>
                <th className="pb-3 font-semibold">Drift Severity</th>
                <th className="pb-3 font-semibold">KS Statistic / p-value</th>
                <th className="pb-3 font-semibold">Baseline vs Current Mean</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-xs">
              {Object.entries(featureDrift).map(([name, feat]) => {
                const isHigh = feat.psi >= 0.20
                const isMod = feat.psi >= 0.10 && feat.psi < 0.20
                const badgeColor = isHigh
                  ? 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                  : isMod
                  ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300'
                  : 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'

                return (
                  <tr key={name} className="hover:bg-slate-50/60 dark:hover:bg-slate-900/40">
                    <td className="py-3 font-sans font-bold text-slate-900 dark:text-white">{name}</td>
                    <td className="py-3 font-sans capitalize text-slate-500">{feat.type}</td>
                    <td className="py-3 font-bold text-slate-800 dark:text-slate-200">{feat.psi?.toFixed(3)}</td>
                    <td className="py-3">
                      <span className={`inline-flex rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${badgeColor}`}>
                        {feat.severity}
                      </span>
                    </td>
                    <td className="py-3 text-slate-600 dark:text-slate-400">
                      {feat.type === 'numeric'
                        ? `KS: ${feat.ks_statistic} (p=${feat.ks_p_value})`
                        : '— (Categorical)'}
                    </td>
                    <td className="py-3 text-slate-600 dark:text-slate-400">
                      {feat.type === 'numeric'
                        ? `${feat.reference_mean} → ${feat.current_mean}`
                        : 'Frequency proportions monitored'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
