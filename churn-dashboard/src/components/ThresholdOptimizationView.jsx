import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Check, DollarSign, Sliders, Target, TrendingUp } from 'lucide-react'
import ChartCard from './ChartCard.jsx'

export default function ThresholdOptimizationView({ onThresholdUpdated }) {
  const [analysis, setAnalysis] = useState(null)
  const [selectedThreshold, setSelectedThreshold] = useState(0.35)
  const [applied, setApplied] = useState(false)
  const [applying, setApplying] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get('/api/threshold-analysis', { params: { step: 0.05 } })
      .then((res) => {
        setAnalysis(res.data)
        if (res.data.current_threshold) {
          setSelectedThreshold(res.data.current_threshold)
        }
      })
      .catch((err) => console.error('Threshold analysis error:', err))
      .finally(() => setLoading(false))
  }, [])

  const handleApplyThreshold = async (t) => {
    setApplying(true)
    try {
      await axios.post('/api/threshold/set', { threshold: parseFloat(t) })
      setSelectedThreshold(parseFloat(t))
      setApplied(true)
      if (onThresholdUpdated) onThresholdUpdated(parseFloat(t))
      setTimeout(() => setApplied(false), 3000)
    } catch (err) {
      console.error('Failed to set operating threshold:', err)
    } finally {
      setApplying(false)
    }
  }

  if (loading) {
    return (
      <div className="py-20 text-center text-sm text-slate-500 animate-pulse">
        Evaluating cost curves and threshold optimization trade-offs…
      </div>
    )
  }

  const optimal = analysis?.optimal_thresholds || {}
  const curve = analysis?.threshold_curve || []

  // Find the curve point closest to selected slider threshold
  const currentPoint = curve.find((p) => Math.abs(p.threshold - selectedThreshold) < 0.026) || curve[0] || {}

  return (
    <div className="space-y-8">
      {/* Header and Interactive Slider */}
      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sliders className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
              <h3 className="text-base font-semibold text-slate-950 dark:text-white">Business Economics & Threshold Optimization</h3>
            </div>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Tune classification decision boundary based on unit costs: $500 missed churner, $50 false positive, $25 intervention.
            </p>
          </div>

          <button
            onClick={() => handleApplyThreshold(selectedThreshold)}
            disabled={applying}
            className="inline-flex items-center gap-2 rounded-2xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:opacity-50"
          >
            {applied ? <Check className="h-4 w-4" /> : null}
            {applied ? 'Threshold Persisted!' : applying ? 'Persisting…' : `Apply ${selectedThreshold.toFixed(2)} to Live API`}
          </button>
        </div>

        {/* Slider bar */}
        <div className="mt-6 rounded-2xl bg-slate-50 p-5 dark:bg-slate-900/60">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-slate-700 dark:text-slate-300">Target Operating Threshold:</span>
            <span className="font-mono text-xl font-bold text-indigo-600 dark:text-indigo-400">
              {Number(selectedThreshold).toFixed(2)}
            </span>
          </div>

          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={selectedThreshold}
            onChange={(e) => setSelectedThreshold(parseFloat(e.target.value))}
            className="mt-3 h-2 w-full cursor-pointer accent-indigo-600"
          />

          <div className="mt-4 grid grid-cols-2 gap-4 text-xs font-mono sm:grid-cols-4">
            <div>
              <span className="text-slate-400">Total Estimated Cost:</span>
              <p className="text-sm font-bold text-slate-900 dark:text-white">
                ${currentPoint.total_estimated_cost?.toLocaleString()}
              </p>
            </div>
            <div>
              <span className="text-slate-400">Flagged Customer Vol:</span>
              <p className="text-sm font-bold text-slate-900 dark:text-white">
                {currentPoint.customers_flagged} ({currentPoint.flagged_percentage}%)
              </p>
            </div>
            <div>
              <span className="text-slate-400">Precision:</span>
              <p className="text-sm font-bold text-emerald-600">
                {(currentPoint.precision * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span className="text-slate-400">Recall:</span>
              <p className="text-sm font-bold text-indigo-600">
                {(currentPoint.recall * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Recommended Thresholds Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* 1. Min Cost */}
        <div className="rounded-3xl border border-indigo-200 bg-indigo-50/50 p-5 dark:border-indigo-900/50 dark:bg-indigo-950/20">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wide text-indigo-700 dark:text-indigo-300">Min Total Cost</span>
            <DollarSign className="h-4 w-4 text-indigo-600" />
          </div>
          <p className="mt-2 font-mono text-3xl font-extrabold text-indigo-950 dark:text-white">
            {optimal.min_cost?.threshold}
          </p>
          <p className="mt-1 text-xs text-slate-500">Minimizes total economic risk (${optimal.min_cost?.total_estimated_cost?.toLocaleString()})</p>
          <button
            onClick={() => handleApplyThreshold(optimal.min_cost?.threshold)}
            className="mt-4 w-full rounded-xl bg-indigo-600 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-700"
          >
            Select Min Cost
          </button>
        </div>

        {/* 2. Max F1 */}
        <div className="rounded-3xl border border-slate-200 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wide text-slate-500">Max F1 Score</span>
            <Target className="h-4 w-4 text-slate-400" />
          </div>
          <p className="mt-2 font-mono text-3xl font-extrabold text-slate-950 dark:text-white">
            {optimal.max_f1?.threshold}
          </p>
          <p className="mt-1 text-xs text-slate-500">Peak balanced F1 score ({optimal.max_f1?.f1_score})</p>
          <button
            onClick={() => handleApplyThreshold(optimal.max_f1?.threshold)}
            className="mt-4 w-full rounded-xl bg-slate-100 py-1.5 text-xs font-semibold text-slate-800 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200"
          >
            Select Max F1
          </button>
        </div>

        {/* 3. Max Recall under budget */}
        <div className="rounded-3xl border border-slate-200 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wide text-slate-500">Budget Constrained</span>
            <TrendingUp className="h-4 w-4 text-slate-400" />
          </div>
          <p className="mt-2 font-mono text-3xl font-extrabold text-slate-950 dark:text-white">
            {optimal.max_recall_under_budget?.threshold}
          </p>
          <p className="mt-1 text-xs text-slate-500">Max recall with &le;30% flagged customers</p>
          <button
            onClick={() => handleApplyThreshold(optimal.max_recall_under_budget?.threshold)}
            className="mt-4 w-full rounded-xl bg-slate-100 py-1.5 text-xs font-semibold text-slate-800 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200"
          >
            Select Budget Point
          </button>
        </div>

        {/* 4. Target Recall Precision */}
        <div className="rounded-3xl border border-slate-200 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wide text-slate-500">Recall Target (&ge;70%)</span>
            <Target className="h-4 w-4 text-slate-400" />
          </div>
          <p className="mt-2 font-mono text-3xl font-extrabold text-slate-950 dark:text-white">
            {optimal.max_precision_at_target_recall?.threshold}
          </p>
          <p className="mt-1 text-xs text-slate-500">Max precision at minimum 70% recall target</p>
          <button
            onClick={() => handleApplyThreshold(optimal.max_precision_at_target_recall?.threshold)}
            className="mt-4 w-full rounded-xl bg-slate-100 py-1.5 text-xs font-semibold text-slate-800 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200"
          >
            Select Recall Target
          </button>
        </div>
      </div>

      {/* Charts: Cost Breakdown & Trade-Offs */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Total Cost Curve */}
        <ChartCard
          title="Total Estimated Cost Curve vs Operating Threshold"
          subtitle="Identifies the global cost minimum balancing missed churners and intervention spend."
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curve}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="threshold" />
              <YAxis tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip formatter={(v) => [`$${Number(v).toLocaleString()}`, 'Cost']} />
              <Line type="monotone" dataKey="total_estimated_cost" stroke="#ef4444" strokeWidth={3} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Precision vs Recall Trade-Off Curve */}
        <ChartCard
          title="Precision vs Recall Across Thresholds"
          subtitle="Track sensitivity to False Positives vs False Negatives."
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curve}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="threshold" />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`]} />
              <Legend />
              <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2.5} name="Precision" dot={false} />
              <Line type="monotone" dataKey="recall" stroke="#6366f1" strokeWidth={2.5} name="Recall" dot={false} />
              <Line type="monotone" dataKey="f1_score" stroke="#8b5cf6" strokeWidth={2} name="F1 Score" strokeDasharray="4 4" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  )
}
