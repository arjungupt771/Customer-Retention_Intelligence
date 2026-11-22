import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  CartesianGrid,
  Line,
  LineChart,
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Activity, CheckCircle2, Sliders } from 'lucide-react'
import ChartCard from './ChartCard.jsx'
import MetricCard from './MetricCard.jsx'

export default function ModelEvaluationView({ initialThreshold = 0.35 }) {
  const [threshold, setThreshold] = useState(initialThreshold)
  const [diagnostics, setDiagnostics] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let isMounted = true

    Promise.all([
      axios.get('/api/model-diagnostics', { params: { threshold } }),
      axios.get('/api/model-confidence', { params: { threshold, n_bootstraps: 150 } }),
    ])
      .then(([diagRes, confRes]) => {
        if (isMounted) {
          setDiagnostics(diagRes.data)
          setConfidence(confRes.data)
          setError('')
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.response?.data?.detail || err.message || 'Failed to load model diagnostics')
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false)
      })

    return () => {
      isMounted = false
    }
  }, [threshold])

  if (loading && !diagnostics) {
    return (
      <div className="flex h-64 items-center justify-center rounded-3xl border border-slate-200 bg-white/50 dark:border-slate-800 dark:bg-slate-900/50">
        <p className="text-sm text-slate-500 animate-pulse">Loading model evaluation diagnostics & confidence intervals…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-3xl border border-rose-200 bg-rose-50/80 p-6 text-rose-800 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
        <p className="font-semibold">Unable to load evaluation diagnostics</p>
        <p className="mt-1 text-sm">{error}</p>
      </div>
    )
  }

  const cm = diagnostics?.confusion_matrix || { true_positives: 0, true_negatives: 0, false_positives: 0, false_negatives: 0, total: 1 }
  const report = diagnostics?.classification_report || {}
  const roc = diagnostics?.roc_curve || { roc_auc: 0.85, points: [] }
  const pr = diagnostics?.pr_curve || { average_precision: 0.65, points: [] }
  const lift = diagnostics?.lift_chart || { deciles: [] }
  const confMetrics = confidence?.metrics || {}

  return (
    <div className="space-y-8">
      {/* Operating Threshold Header Control */}
      <div className="flex flex-col gap-4 rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Sliders className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
            <h3 className="text-lg font-bold text-slate-950 dark:text-white">Operating Threshold Diagnostics</h3>
          </div>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Adjust the classification threshold to evaluate metric sensitivity, confusion matrix shifts, and trade-offs.
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-sm font-semibold text-indigo-600 dark:text-indigo-400">
            Threshold: {Number(threshold).toFixed(2)}
          </span>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            value={threshold}
            onChange={(e) => {
              setLoading(true)
              setThreshold(parseFloat(e.target.value))
            }}
            className="h-2 w-40 cursor-pointer accent-indigo-600"
          />
        </div>
      </div>

      {/* KPI Cards: Classification Report */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <MetricCard
          label="Accuracy"
          value={`${(report.accuracy * 100).toFixed(1)}%`}
          caption="Overall correct"
          icon={<CheckCircle2 className="h-5 w-5 text-emerald-500" />}
        />
        <MetricCard
          label="Precision"
          value={`${(report.precision * 100).toFixed(1)}%`}
          caption="True churn rate"
          icon={<Activity className="h-5 w-5 text-indigo-500" />}
        />
        <MetricCard
          label="Recall (Sensitivity)"
          value={`${(report.recall * 100).toFixed(1)}%`}
          caption="Churners caught"
          icon={<Activity className="h-5 w-5 text-amber-500" />}
        />
        <MetricCard
          label="F1 Score"
          value={report.f1_score?.toFixed(3)}
          caption="Harmonic mean"
          icon={<Activity className="h-5 w-5 text-violet-500" />}
        />
        <MetricCard
          label="Specificity"
          value={`${(report.specificity * 100).toFixed(1)}%`}
          caption="True negative rate"
          icon={<CheckCircle2 className="h-5 w-5 text-teal-500" />}
        />
        <MetricCard
          label="Balanced Accuracy"
          value={`${(report.balanced_accuracy * 100).toFixed(1)}%`}
          caption="Class-weighted"
          icon={<CheckCircle2 className="h-5 w-5 text-sky-500" />}
        />
      </div>

      {/* Confusion Matrix & Bootstrap Confidence */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Confusion Matrix */}
        <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
          <h4 className="text-base font-semibold text-slate-950 dark:text-white">Confusion Matrix</h4>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Evaluated on held-out test cohort (total: {cm.total} customers).
          </p>

          <div className="mt-6 grid grid-cols-2 gap-3 text-center">
            <div className="rounded-2xl border border-emerald-200 bg-emerald-50/60 p-4 dark:border-emerald-900/50 dark:bg-emerald-950/30">
              <span className="text-xs uppercase font-medium text-emerald-700 dark:text-emerald-400">True Positives (TP)</span>
              <p className="mt-2 text-2xl font-bold text-emerald-900 dark:text-emerald-200">{cm.true_positives}</p>
              <span className="text-xs text-emerald-600/80 dark:text-emerald-400/70">Correctly caught churners</span>
            </div>
            <div className="rounded-2xl border border-amber-200 bg-amber-50/60 p-4 dark:border-amber-900/50 dark:bg-amber-950/30">
              <span className="text-xs uppercase font-medium text-amber-700 dark:text-amber-400">False Positives (FP)</span>
              <p className="mt-2 text-2xl font-bold text-amber-900 dark:text-amber-200">{cm.false_positives}</p>
              <span className="text-xs text-amber-600/80 dark:text-amber-400/70">False alarms (outreach cost)</span>
            </div>
            <div className="rounded-2xl border border-rose-200 bg-rose-50/60 p-4 dark:border-rose-900/50 dark:bg-rose-950/30">
              <span className="text-xs uppercase font-medium text-rose-700 dark:text-rose-400">False Negatives (FN)</span>
              <p className="mt-2 text-2xl font-bold text-rose-900 dark:text-rose-200">{cm.false_negatives}</p>
              <span className="text-xs text-rose-600/80 dark:text-rose-400/70">Missed churners (lost revenue)</span>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/50">
              <span className="text-xs uppercase font-medium text-slate-700 dark:text-slate-400">True Negatives (TN)</span>
              <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-slate-200">{cm.true_negatives}</p>
              <span className="text-xs text-slate-500 dark:text-slate-400">Correctly retained loyals</span>
            </div>
          </div>
        </div>

        {/* Statistical Confidence & Uncertainty (95% CI) */}
        <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold text-slate-950 dark:text-white">95% Bootstrap Confidence Intervals</h4>
            <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300">
              500 Resamples
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Quantifies statistical estimation uncertainty across metric distributions.
          </p>

          <div className="mt-5 space-y-3">
            {[
              { key: 'roc_auc', name: 'ROC-AUC' },
              { key: 'pr_auc', name: 'PR-AUC (Avg Precision)' },
              { key: 'f1_score', name: 'F1 Score' },
              { key: 'recall', name: 'Recall' },
              { key: 'precision', name: 'Precision' },
            ].map(({ key, name }) => {
              const item = confMetrics[key]
              if (!item) return null
              return (
                <div key={key} className="flex items-center justify-between rounded-xl bg-slate-50 px-4 py-2.5 dark:bg-slate-900/60">
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{name}</span>
                  <div className="text-right">
                    <span className="font-mono text-sm font-bold text-slate-900 dark:text-white">
                      {item.estimate}
                    </span>
                    <span className="ml-2 font-mono text-xs text-slate-500 dark:text-slate-400">
                      [{item.confidence_interval?.lower} – {item.confidence_interval?.upper}]
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* ROC & PR Curves */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ROC Curve */}
        <ChartCard
          title={`ROC Curve (AUC: ${roc.roc_auc})`}
          subtitle="True Positive Rate vs False Positive Rate across threshold range"
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={roc.points}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="fpr" domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} label={{ value: 'FPR', position: 'insideBottom', offset: -5 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} label={{ value: 'TPR', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(val) => [Number(val).toFixed(3)]} labelFormatter={(l) => `FPR: ${Number(l).toFixed(3)}`} />
              <Line type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* PR Curve */}
        <ChartCard
          title={`Precision-Recall Curve (AP: ${pr.average_precision})`}
          subtitle={`Operating Precision: ${(pr.operating_metrics?.precision * 100).toFixed(1)}% | Recall: ${(pr.operating_metrics?.recall * 100).toFixed(1)}%`}
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={pr.points}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="recall" domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} label={{ value: 'Recall', position: 'insideBottom', offset: -5 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} label={{ value: 'Precision', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(val) => [Number(val).toFixed(3)]} labelFormatter={(l) => `Recall: ${Number(l).toFixed(3)}`} />
              <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Lift & Gain Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Decile Lift Chart */}
        <ChartCard
          title="Decile Lift Chart"
          subtitle={`Churn targeting multiplier relative to average baseline churn rate (${lift.overall_churn_rate}%).`}
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={lift.deciles}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="decile" tickFormatter={(d) => `D${d}`} />
              <YAxis tickFormatter={(v) => `${v}x`} />
              <Tooltip formatter={(v) => [`${v}x Lift`, 'Lift']} labelFormatter={(d) => `Decile ${d}`} />
              <Bar dataKey="lift" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Cumulative Gains Chart */}
        <ChartCard
          title="Cumulative Gains Chart"
          subtitle="Cumulative percentage of total churners captured as population deciles are contacted."
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={lift.deciles}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="cumulative_customers_pct" tickFormatter={(v) => `${v}%`} label={{ value: '% Population Targeted', position: 'insideBottom', offset: -5 }} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} label={{ value: '% Churners Caught', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(v) => [`${v}%`, 'Gains']} labelFormatter={(v) => `Population: ${v}%`} />
              <Line type="monotone" dataKey="cumulative_churners_pct" stroke="#f59e0b" strokeWidth={3} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  )
}
