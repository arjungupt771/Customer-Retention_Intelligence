import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertCircle, CheckCircle2, Search, Sparkles } from 'lucide-react'
import ChartCard from './ChartCard.jsx'

export default function ExplainabilityView() {
  const [globalImportance, setGlobalImportance] = useState([])
  const [selectedCustomerId, setSelectedCustomerId] = useState('7590-VHVEG')
  const [searchId, setSearchId] = useState('7590-VHVEG')
  const [customerExplanation, setCustomerExplanation] = useState(null)
  const [loadingCustomer, setLoadingCustomer] = useState(false)
  const [loadingGlobal, setLoadingGlobal] = useState(true)
  const [customerError, setCustomerError] = useState('')

  // Live what-if prediction form state
  const [simForm, setSimForm] = useState({
    gender: 'Male',
    tenure: 12,
    monthly_charges: 75.5,
    contract: 'Month-to-month',
    payment_method: 'Electronic check',
  })
  const [simResult, setSimResult] = useState(null)
  const [simulating, setSimulating] = useState(false)

  // 1. Fetch global feature importance
  useEffect(() => {
    axios.get('/api/explain/global')
      .then((res) => setGlobalImportance(res.data.feature_importance || []))
      .catch((err) => console.error('Global importance error:', err))
      .finally(() => setLoadingGlobal(false))
  }, [])

  // 2. Fetch customer explanation
  useEffect(() => {
    if (!selectedCustomerId) return

    axios.get(`/api/explain/${selectedCustomerId.trim()}`)
      .then((res) => {
        setCustomerExplanation(res.data)
        setCustomerError('')
      })
      .catch((err) => {
        setCustomerError(err.response?.data?.detail || 'Customer not found')
        setCustomerExplanation(null)
      })
      .finally(() => setLoadingCustomer(false))
  }, [selectedCustomerId])

  const handleSearch = (e) => {
    e.preventDefault()
    if (searchId.trim()) {
      setLoadingCustomer(true)
      setSelectedCustomerId(searchId.trim())
    }
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    setSimulating(true)
    try {
      const res = await axios.post('/api/predict', simForm)
      setSimResult(res.data)
    } catch (err) {
      console.error('Prediction simulation error:', err)
    } finally {
      setSimulating(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Global Feature Importance Chart */}
      <ChartCard
        title="SHAP Global Feature Importance"
        subtitle="Average absolute impact of each customer attribute on model churn probability predictions."
      >
        {loadingGlobal ? (
          <div className="py-16 text-center text-sm text-slate-500 animate-pulse">Calculating SHAP global contributions…</div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={globalImportance.slice(0, 10)}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} horizontal={false} />
              <XAxis type="number" tickFormatter={(v) => Number(v).toFixed(2)} />
              <YAxis dataKey="feature" type="category" tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(val, name, item) => [
                  `${val} (${item.payload.relative_importance_pct}%)`,
                  'Mean |SHAP| Value',
                ]}
              />
              <Bar dataKey="mean_abs_shap" fill="#6366f1" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      {/* Customer-Level SHAP Explanation */}
      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-950 dark:text-white">Individual Customer Prediction Explanation</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Inspect fine-grained SHAP contribution waterfall and specific risk vs protective factors.
            </p>
          </div>

          <form onSubmit={handleSearch} className="flex items-center gap-2">
            <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-900">
              <Search className="h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={searchId}
                onChange={(e) => setSearchId(e.target.value)}
                placeholder="Enter Customer ID…"
                className="w-40 bg-transparent text-sm font-mono text-slate-900 outline-none dark:text-white"
              />
            </div>
            <button
              type="submit"
              className="rounded-2xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700"
            >
              Explain
            </button>
          </form>
        </div>

        {loadingCustomer ? (
          <div className="py-16 text-center text-sm text-slate-500 animate-pulse">
            Computing SHAP feature attribution waterfall…
          </div>
        ) : customerError ? (
          <div className="mt-6 rounded-2xl bg-rose-50 p-4 text-sm text-rose-800 dark:bg-rose-950/40 dark:text-rose-200">
            {customerError}
          </div>
        ) : customerExplanation ? (
          <div className="mt-6 space-y-6">
            {/* Header info */}
            <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-slate-50 p-4 dark:bg-slate-900/60">
              <div>
                <span className="text-xs uppercase text-slate-500">Account:</span>
                <p className="font-mono text-lg font-bold text-slate-950 dark:text-white">{customerExplanation.customer_id}</p>
              </div>
              <div>
                <span className="text-xs uppercase text-slate-500">Churn Probability:</span>
                <p className="font-mono text-xl font-bold text-indigo-600 dark:text-indigo-400">
                  {(customerExplanation.churn_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span className="text-xs uppercase text-slate-500">Prediction:</span>
                <p className="mt-0.5">
                  <span className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                    customerExplanation.predicted_churn
                      ? 'bg-rose-100 text-rose-800 dark:bg-rose-950/80 dark:text-rose-300'
                      : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300'
                  }`}>
                    {customerExplanation.predicted_churn ? 'Predicted Churner' : 'Predicted Retained'}
                  </span>
                </p>
              </div>
            </div>

            {/* Side-by-side: Risk vs Protective Factor bullets */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="rounded-2xl border border-rose-200 bg-rose-50/50 p-5 dark:border-rose-900/50 dark:bg-rose-950/20">
                <div className="flex items-center gap-2 text-rose-800 dark:text-rose-300">
                  <AlertCircle className="h-5 w-5" />
                  <h4 className="font-semibold text-sm">Factors Increasing Churn Risk</h4>
                </div>
                <ul className="mt-3 space-y-2 text-sm text-slate-700 dark:text-slate-300">
                  {customerExplanation.top_risk_factors?.length ? (
                    customerExplanation.top_risk_factors.map((rf, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-rose-500 font-bold">•</span>
                        <span>{rf.description} <strong className="font-mono text-rose-600 text-xs">(+{rf.contribution})</strong></span>
                      </li>
                    ))
                  ) : (
                    <li className="text-slate-500 text-xs italic">No major risk factors observed.</li>
                  )}
                </ul>
              </div>

              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/50 p-5 dark:border-emerald-900/50 dark:bg-emerald-950/20">
                <div className="flex items-center gap-2 text-emerald-800 dark:text-emerald-300">
                  <CheckCircle2 className="h-5 w-5" />
                  <h4 className="font-semibold text-sm">Factors Reducing Churn Risk</h4>
                </div>
                <ul className="mt-3 space-y-2 text-sm text-slate-700 dark:text-slate-300">
                  {customerExplanation.top_protective_factors?.length ? (
                    customerExplanation.top_protective_factors.map((pf, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-emerald-500 font-bold">•</span>
                        <span>{pf.description} <strong className="font-mono text-emerald-600 text-xs">({pf.contribution})</strong></span>
                      </li>
                    ))
                  ) : (
                    <li className="text-slate-500 text-xs italic">No protective factors observed.</li>
                  )}
                </ul>
              </div>
            </div>

            {/* Feature contribution horizontal chart */}
            <div className="mt-4">
              <h5 className="text-xs font-semibold uppercase text-slate-500">Feature Contribution Breakdown</h5>
              <div className="mt-3 h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={customerExplanation.feature_contributions || []}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} horizontal={false} />
                    <XAxis type="number" tickFormatter={(v) => Number(v).toFixed(2)} />
                    <YAxis dataKey="transformed_feature" type="category" tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => [v, 'SHAP Value']} />
                    <Bar dataKey="shap_value">
                      {(customerExplanation.feature_contributions || []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#f43f5e' : '#10b981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Interactive Live Inference Simulation (POST /api/predict) */}
      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
          <h3 className="text-base font-semibold text-slate-950 dark:text-white">Live Prediction Inference API Client</h3>
        </div>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          Simulate a real-time churn prediction via POST /api/predict to evaluate counterfactual customer adjustments.
        </p>

        <form onSubmit={handlePredict} className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400">Contract</label>
            <select
              value={simForm.contract}
              onChange={(e) => setSimForm({ ...simForm, contract: e.target.value })}
              className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-sm dark:border-slate-800 dark:bg-slate-900 dark:text-white"
            >
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400">Tenure (months)</label>
            <input
              type="number"
              min="0"
              max="100"
              value={simForm.tenure}
              onChange={(e) => setSimForm({ ...simForm, tenure: parseInt(e.target.value) || 0 })}
              className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-sm dark:border-slate-800 dark:bg-slate-900 dark:text-white"
            />
          </div>

          <div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400">Monthly Charges ($)</label>
            <input
              type="number"
              min="0"
              step="0.5"
              value={simForm.monthly_charges}
              onChange={(e) => setSimForm({ ...simForm, monthly_charges: parseFloat(e.target.value) || 0 })}
              className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-sm dark:border-slate-800 dark:bg-slate-900 dark:text-white"
            />
          </div>

          <div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400">Payment Method</label>
            <select
              value={simForm.payment_method}
              onChange={(e) => setSimForm({ ...simForm, payment_method: e.target.value })}
              className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-sm dark:border-slate-800 dark:bg-slate-900 dark:text-white"
            >
              <option value="Electronic check">Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              disabled={simulating}
              className="w-full rounded-xl bg-indigo-600 p-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:opacity-50"
            >
              {simulating ? 'Inferring…' : 'Run /api/predict'}
            </button>
          </div>
        </form>

        {simResult && (
          <div className="mt-5 flex items-center justify-between rounded-2xl bg-indigo-50/70 p-4 dark:bg-indigo-950/40">
            <div>
              <span className="text-xs uppercase text-indigo-700 dark:text-indigo-300">Churn Probability:</span>
              <p className="font-mono text-2xl font-bold text-indigo-950 dark:text-indigo-200">
                {(simResult.churn_probability * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span className="text-xs uppercase text-indigo-700 dark:text-indigo-300">Risk Category:</span>
              <p className="font-semibold text-sm">
                <span className={`inline-flex rounded-full px-3 py-1 text-xs font-bold ${
                  simResult.risk_category === 'High'
                    ? 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-200'
                    : simResult.risk_category === 'Medium'
                    ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200'
                    : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200'
                }`}>
                  {simResult.risk_category} Risk
                </span>
              </p>
            </div>
            <div>
              <span className="text-xs uppercase text-indigo-700 dark:text-indigo-300">Operating Threshold:</span>
              <p className="font-mono text-sm text-slate-700 dark:text-slate-300">{simResult.threshold}</p>
            </div>
            <div>
              <span className="text-xs uppercase text-indigo-700 dark:text-indigo-300">Verdict:</span>
              <p className="font-bold text-sm text-slate-900 dark:text-white">
                {simResult.predicted_churn ? '🚨 Churn Expected' : '✅ Retained'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
