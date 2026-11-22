import { useEffect, useState } from 'react'
import axios from 'axios'
import { AlertTriangle, Download, Filter, Search, ShieldCheck } from 'lucide-react'
import MetricCard from './MetricCard.jsx'

export default function ActionCenterView() {
  const [summary, setSummary] = useState(null)
  const [customers, setCustomers] = useState([])
  const [totalFiltered, setTotalFiltered] = useState(0)
  const [loading, setLoading] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [filters, setFilters] = useState({
    risk_category: 'High',
    contract: 'All',
    search: '',
  })

  useEffect(() => {
    let isMounted = true

    axios.get('/api/action-center/summary')
      .then((res) => {
        if (isMounted) setSummary(res.data)
      })
      .catch((err) => console.error('Summary error:', err))

    return () => {
      isMounted = false
    }
  }, [])

  const updateFilters = (updater) => {
    setLoading(true)
    setFilters(updater)
  }

  useEffect(() => {
    let isMounted = true

    const params = {
      risk_category: filters.risk_category === 'All' ? undefined : filters.risk_category,
      contract: filters.contract === 'All' ? undefined : filters.contract,
      search: filters.search.trim() || undefined,
      limit: 25,
      offset: 0,
    }

    axios.get('/api/action-center/customers', { params })
      .then((res) => {
        if (isMounted) {
          setCustomers(res.data.customers || [])
          setTotalFiltered(res.data.total || 0)
        }
      })
      .catch((err) => console.error('Customers list error:', err))
      .finally(() => {
        if (isMounted) setLoading(false)
      })

    return () => {
      isMounted = false
    }
  }, [filters])

  const handleExportCsv = async () => {
    setExporting(true)
    try {
      const params = {
        risk_category: filters.risk_category === 'All' ? undefined : filters.risk_category,
        contract: filters.contract === 'All' ? undefined : filters.contract,
        search: filters.search.trim() || undefined,
        limit: 5000,
      }
      const response = await axios.get('/api/action-center/export-csv', {
        params,
        responseType: 'blob',
      })

      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `retention_action_customers_${filters.risk_category.toLowerCase()}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Failed to export CSV:', err)
    } finally {
      setExporting(false)
    }
  }

  const dist = summary?.risk_distribution || {}

  return (
    <div className="space-y-8">
      {/* Executive Action Center Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="High-Risk Customers"
          value={summary?.high_risk_customers?.toLocaleString() ?? '—'}
          caption={`${summary?.high_risk_percentage ?? '—'}% of total population`}
          icon={<AlertTriangle className="h-5 w-5 text-rose-500" />}
        />
        <MetricCard
          label="Monthly Revenue at Risk"
          value={`$${summary?.monthly_revenue_at_risk?.toLocaleString() ?? '—'}`}
          caption="Exposure among high-risk cohort"
          icon={<AlertTriangle className="h-5 w-5 text-amber-500" />}
        />
        <MetricCard
          label="Average Churn Probability"
          value={`${((summary?.average_churn_probability ?? 0) * 100).toFixed(1)}%`}
          caption="Across entire active customer base"
          icon={<ShieldCheck className="h-5 w-5 text-indigo-500" />}
        />
        <MetricCard
          label="Predicted Churn Rate"
          value={`${summary?.predicted_churn_rate ?? '—'}%`}
          caption="At current operating threshold"
          icon={<ShieldCheck className="h-5 w-5 text-violet-500" />}
        />
      </div>

      {/* Risk Distribution Banner */}
      <div className="grid grid-cols-3 gap-4 rounded-3xl border border-slate-200/80 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
        <div className="rounded-2xl bg-rose-50/70 p-4 dark:bg-rose-950/30">
          <p className="text-xs uppercase font-medium text-rose-700 dark:text-rose-400">High Risk (≥ 70%)</p>
          <p className="mt-1 text-2xl font-bold text-rose-950 dark:text-rose-200">{dist.high?.count?.toLocaleString() ?? '—'}</p>
          <span className="text-xs text-rose-600/80 dark:text-rose-400/80">{dist.high?.percentage ?? 0}% of accounts</span>
        </div>
        <div className="rounded-2xl bg-amber-50/70 p-4 dark:bg-amber-950/30">
          <p className="text-xs uppercase font-medium text-amber-700 dark:text-amber-400">Medium Risk (30–69%)</p>
          <p className="mt-1 text-2xl font-bold text-amber-950 dark:text-amber-200">{dist.medium?.count?.toLocaleString() ?? '—'}</p>
          <span className="text-xs text-amber-600/80 dark:text-amber-400/80">{dist.medium?.percentage ?? 0}% of accounts</span>
        </div>
        <div className="rounded-2xl bg-emerald-50/70 p-4 dark:bg-emerald-950/30">
          <p className="text-xs uppercase font-medium text-emerald-700 dark:text-emerald-400">Low Risk (&lt; 30%)</p>
          <p className="mt-1 text-2xl font-bold text-emerald-950 dark:text-emerald-200">{dist.low?.count?.toLocaleString() ?? '—'}</p>
          <span className="text-xs text-emerald-600/80 dark:text-emerald-400/80">{dist.low?.percentage ?? 0}% of accounts</span>
        </div>
      </div>

      {/* Action Table Filter Bar & Export */}
      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h4 className="text-base font-semibold text-slate-950 dark:text-white">Customer Intervention Targets</h4>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Showing top priority customers ranked by model-predicted churn risk ({totalFiltered} accounts found).
            </p>
          </div>

          <button
            onClick={handleExportCsv}
            disabled={exporting}
            className="inline-flex items-center gap-2 rounded-2xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:opacity-50"
          >
            <Download className="h-4 w-4" />
            {exporting ? 'Generating CSV…' : 'Export Action CSV'}
          </button>
        </div>

        {/* Filter Controls */}
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-900">
            <Filter className="h-4 w-4 text-slate-400" />
            <span className="text-xs font-medium text-slate-500">Risk Tier:</span>
            <select
              value={filters.risk_category}
              onChange={(e) => updateFilters((prev) => ({ ...prev, risk_category: e.target.value }))}
              className="bg-transparent font-medium text-slate-900 outline-none dark:text-white"
            >
              <option value="All">All Tiers</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>
          </div>

          <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-900">
            <span className="text-xs font-medium text-slate-500">Contract:</span>
            <select
              value={filters.contract}
              onChange={(e) => updateFilters((prev) => ({ ...prev, contract: e.target.value }))}
              className="bg-transparent font-medium text-slate-900 outline-none dark:text-white"
            >
              <option value="All">All Contracts</option>
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>

          <div className="flex flex-1 items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-900">
            <Search className="h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search customer ID…"
              value={filters.search}
              onChange={(e) => updateFilters((prev) => ({ ...prev, search: e.target.value }))}
              className="w-full bg-transparent text-sm text-slate-900 placeholder:text-slate-400 outline-none dark:text-white"
            />
          </div>
        </div>

        {/* Customer Table */}
        <div className="mt-6 overflow-x-auto">
          {loading ? (
            <div className="py-12 text-center text-sm text-slate-500 animate-pulse">
              Loading prioritized customers…
            </div>
          ) : customers.length === 0 ? (
            <div className="py-12 text-center text-sm text-slate-500">
              No customers match the current filter selection.
            </div>
          ) : (
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase text-slate-400 dark:border-slate-800">
                <tr>
                  <th className="pb-3 font-semibold">Customer ID</th>
                  <th className="pb-3 font-semibold">Churn Risk</th>
                  <th className="pb-3 font-semibold">Risk Tier</th>
                  <th className="pb-3 font-semibold">Contract</th>
                  <th className="pb-3 font-semibold">Tenure</th>
                  <th className="pb-3 font-semibold">Monthly Charge</th>
                  <th className="pb-3 font-semibold">Payment Method</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                {customers.map((c) => {
                  const riskColor =
                    c.risk_category === 'High'
                      ? 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                      : c.risk_category === 'Medium'
                      ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300'
                      : 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'

                  return (
                    <tr key={c.customerID} className="hover:bg-slate-50/60 dark:hover:bg-slate-900/40">
                      <td className="py-3.5 font-bold text-slate-900 dark:text-white">{c.customerID}</td>
                      <td className="py-3.5 font-bold text-indigo-600 dark:text-indigo-400">
                        {(c.churn_probability * 100).toFixed(1)}%
                      </td>
                      <td className="py-3.5">
                        <span className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${riskColor}`}>
                          {c.risk_category}
                        </span>
                      </td>
                      <td className="py-3.5 font-sans text-slate-700 dark:text-slate-300">{c.Contract}</td>
                      <td className="py-3.5 font-sans text-slate-700 dark:text-slate-300">{c.tenure} mo</td>
                      <td className="py-3.5 text-slate-700 dark:text-slate-300">${c.MonthlyCharges}</td>
                      <td className="py-3.5 font-sans text-slate-600 dark:text-slate-400">{c.PaymentMethod}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
