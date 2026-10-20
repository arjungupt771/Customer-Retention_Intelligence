import { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { AnimatePresence, motion } from 'framer-motion'
import { jsPDF } from 'jspdf'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Bell,
  ChevronDown,
  CreditCard,
  Download,
  Layers,
  MessageCircle,
  Moon,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  TrendingUp,
} from 'lucide-react'
import Sidebar from './components/Sidebar.jsx'
import MetricCard from './components/MetricCard.jsx'
import ChartCard from './components/ChartCard.jsx'
import SectionHeader from './components/SectionHeader.jsx'
import InsightPanel from './components/InsightPanel.jsx'
import LoadingSkeleton from './components/LoadingSkeleton.jsx'
import { buildSegmentParams, formatNumber, getApiErrorMessage, normalizeKeyMetrics } from './segment-utils.js'
import './App.css'

const navItems = [
  {
    key: 'overview',
    label: 'Overview',
    description: 'Live performance dashboard',
    icon: <Layers className="h-5 w-5" />,
  },
  {
    key: 'churn',
    label: 'Churn insights',
    description: 'Customer retention signals',
    icon: <ShieldCheck className="h-5 w-5" />,
  },
  {
    key: 'contracts',
    label: 'Contracts',
    description: 'Plan and tenure analysis',
    icon: <CreditCard className="h-5 w-5" />,
  },
  {
    key: 'payments',
    label: 'Payments',
    description: 'Billing and loyalty trends',
    icon: <TrendingUp className="h-5 w-5" />,
  },
  {
    key: 'ml',
    label: 'ML insights',
    description: 'Model performance & predictions',
    icon: <Sparkles className="h-5 w-5" />,
  },
]


const sectionMeta = {
  overview: {
    title: 'Overview',
    subtitle: 'A single-pane view of churn performance, adoption, payments, and machine learning signals.',
    breadcrumbs: ['Home', 'Overview'],
  },
  churn: {
    title: 'Churn insights',
    subtitle: 'Understand what drives cancellations so your retention strategy stays proactive.',
    breadcrumbs: ['Home', 'Churn insights'],
  },
  contracts: {
    title: 'Contracts',
    subtitle: 'Analyze how plan type and contract length affect churn and recurring revenue.',
    breadcrumbs: ['Home', 'Contracts'],
  },
  payments: {
    title: 'Payments',
    subtitle: 'Track billing behavior, payment methods, and loyalty signals across customer cohorts.',
    breadcrumbs: ['Home', 'Payments'],
  },
  ml: {
    title: 'ML insights',
    subtitle: 'Review model health, prediction confidence, and the features that matter most.',
    breadcrumbs: ['Home', 'ML insights'],
  },
  'data-summary': {
    title: 'Data summary',
    subtitle: 'Inspect the dataset footprint, column coverage, and feature distributions.',
    breadcrumbs: ['Home', 'Data summary'],
  },
  'key-metrics': {
    title: 'Key metrics',
    subtitle: 'Surface the business KPIs from our churn dataset and model predictions.',
    breadcrumbs: ['Home', 'Key metrics'],
  },
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState({})
  const [darkMode, setDarkMode] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [filters, setFilters] = useState({
    contract: 'All',
    payment: 'All',
    tenure: 'All',
    segment: 'All',
  })
  const [segmentResult, setSegmentResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [statusMessage, setStatusMessage] = useState('Ready')

  useEffect(() => {
    Promise.all(['/api/data-summary', '/api/key-metrics', '/api/contract-analysis', '/api/payment-analysis', '/api/tenure-analysis'].map((endpoint) => axios.get(endpoint)))
      .then(([summary, metrics, contracts, payments, tenure]) => setData((previous) => ({ ...previous,
        '/api/data-summary': summary.data,
        '/api/key-metrics': normalizeKeyMetrics(metrics.data),
        '/api/contract-analysis': contracts.data,
        '/api/payment-analysis': payments.data,
        '/api/tenure-analysis': tenure.data,
      })))
      .catch((error) => setErrorMessage(getApiErrorMessage(error, 'Unable to load dashboard data')))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  const runAnalysis = useCallback(async (endpoint) => {
    setLoading(true)
    setStatusMessage(`Fetching ${endpoint}…`)

    try {
      const response = await axios.get(endpoint)

      setData((prev) => ({ ...prev, [endpoint]: endpoint === '/api/key-metrics' ? normalizeKeyMetrics(response.data) : response.data }))

      if (endpoint === '/api/full-analysis') {
        setData((prev) => ({
          ...prev,
          '/api/data-summary': response.data.data_summary,
          '/api/key-metrics': normalizeKeyMetrics(response.data.key_metrics),
          '/api/contract-analysis': response.data.contract_analysis,
          '/api/payment-analysis': response.data.payment_analysis,
          '/api/ml-models': response.data.ml_models,
        }))
      }

      setErrorMessage('')
      setStatusMessage('Update complete')
    } catch (error) {
      setErrorMessage(String(getApiErrorMessage(error)))
      setStatusMessage('Analysis unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' })

    if (activeTab === 'data-summary' && !data['/api/data-summary']) {
      void Promise.resolve().then(() => runAnalysis('/api/data-summary'))
    }

    if (activeTab === 'key-metrics' && !data['/api/key-metrics']) {
      void Promise.resolve().then(() => runAnalysis('/api/key-metrics'))
    }

    if (activeTab === 'contracts' && !data['/api/contract-analysis']) {
      void Promise.resolve().then(() => runAnalysis('/api/contract-analysis'))
    }

    if (activeTab === 'payments' && !data['/api/payment-analysis']) {
      void Promise.resolve().then(() => runAnalysis('/api/payment-analysis'))
    }

    if (activeTab === 'ml' && !data['/api/ml-models']) {
      void Promise.resolve().then(() => runAnalysis('/api/ml-models'))
    }
  }, [activeTab, data, runAnalysis])

  const runFullAnalysis = async () => {
    setActiveTab('overview')
    await runAnalysis('/api/full-analysis')
  }

  const runSegmentFilter = async (nextFilters = filters) => {
    const params = buildSegmentParams(nextFilters)
    try {
      const response = await axios.get('/api/segments', { params })
      setSegmentResult(response.data)
      setErrorMessage('')
    } catch (error) {
      setErrorMessage(getApiErrorMessage(error, 'Unable to apply filters'))
    }
  }

  const downloadCsv = () => {
    const metrics = data['/api/key-metrics'] || {}
    const csv = ['Metric,Value', `Customers,${metrics.total_customers ?? ''}`, `Churn rate,${metrics.churn_rate ?? ''}%`, `Active monthly revenue at risk,${metrics.active_monthly_revenue_at_risk ?? ''}`, `Historical churned monthly revenue,${metrics.historical_churned_monthly_revenue ?? ''}`].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'churn_dashboard_summary.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const downloadPdf = () => {
    const doc = new jsPDF({ unit: 'px', format: 'a4' })
    doc.setFontSize(24)
    doc.text('Telecom Churn Summary', 40, 50)
    doc.setFontSize(14)
    const metrics = data['/api/key-metrics'] || {}
    doc.text(`Churn rate: ${metrics.churn_rate ?? 'N/A'}%`, 40, 90)
    doc.text(`Active monthly revenue at risk: $${metrics.active_monthly_revenue_at_risk ?? 'N/A'}`, 40, 120)
    doc.text(`Historical churned monthly revenue: $${metrics.historical_churned_monthly_revenue ?? 'N/A'}`, 40, 150)
    doc.save('churn_dashboard.pdf')
  }


  const dataSummary = data['/api/data-summary'] || { total_rows: 0, total_columns: 0, churn_rate: 0, missing_values: 0, unique_customers: 0, memory_usage: '—' }

  const keyMetrics = normalizeKeyMetrics(data['/api/key-metrics'])
  const analysis = data['/api/full-analysis'] || {}
  const modelResults = data['/api/ml-models'] || analysis.ml_models || {}
  const contractComparison = (analysis.contract_analysis?.data || data['/api/contract-analysis']?.data || []).map((row) => ({ name: row.name, churn: row.churn_rate, customers: row.customers }))
  const paymentMethods = (analysis.payment_analysis?.data || data['/api/payment-analysis']?.data || []).map((row) => ({ name: row.name, value: row.customers, churn: row.churn_rate }))
  const tenureData = analysis.tenure_analysis?.data || data['/api/tenure-analysis']?.data || []
  const selectedModel = modelResults.models?.[modelResults.selected_model] || {}
  const mlSummary = [
    { title: 'Precision (churn)', value: `${((selectedModel.precision || 0) * 100).toFixed(1)}%`, accent: 'Held-out test set' },
    { title: 'Recall (churn)', value: `${((selectedModel.recall || 0) * 100).toFixed(1)}%`, accent: 'Held-out test set' },
    { title: 'F1 (churn)', value: `${((selectedModel.f1_score || 0) * 100).toFixed(1)}%`, accent: 'Held-out test set' },
    { title: 'PR-AUC', value: `${((selectedModel.pr_auc || 0) * 100).toFixed(1)}%`, accent: 'Held-out test set' },
    { title: 'Baseline accuracy', value: `${((modelResults.baseline?.accuracy || 0) * 100).toFixed(1)}%`, accent: 'Most frequent class' },
    { title: 'Baseline PR-AUC', value: `${((modelResults.baseline?.pr_auc || 0) * 100).toFixed(1)}%`, accent: 'Test set churn prevalence' },
  ]
  const featureImportance = (modelResults.feature_importance || []).map((row) => ({ feature: row.feature, score: Math.max(1, Math.round(row.importance * 100)) }))
  const activeMetrics = [
    { title: 'Observed churn rate', value: `${keyMetrics.churn_rate}%`, label: `${keyMetrics.churned_customers} of ${keyMetrics.total_customers} customers in the supplied snapshot`, accent: 'Historical observation' },
    { title: 'Active customers', value: formatNumber(keyMetrics.active_customers), label: 'Churn = No in supplied snapshot', accent: 'Current population' },
    { title: 'Active monthly revenue at risk', value: `$${Number(keyMetrics.active_monthly_revenue_at_risk).toLocaleString()}`, label: 'Active monthly charges; exposure, not expected loss', accent: 'Descriptive only' },
  ]
  const filteredChurnReasons = useMemo(() => contractComparison.map((row) => ({
    title: `${row.name} contract`,
    description: `${row.churn}% observed churn among ${formatNumber(row.customers)} customers in this dataset. This is an association, not a causal effect.`,
    trend: `${formatNumber(row.customers)} customers`,
  })).filter((item) => item.title.toLowerCase().includes(searchTerm.toLowerCase())), [searchTerm, contractComparison])

  const pageActions = {
    overview: [
      {
        label: 'Run full analysis',
        onClick: runFullAnalysis,
        variant: 'primary',
      },
      {
        label: 'Export CSV',
        onClick: downloadCsv,
        variant: 'secondary',
      },
    ],
    'data-summary': [
      { label: 'Refresh summary', onClick: () => runAnalysis('/api/data-summary'), variant: 'primary' },
      { label: 'Export PDF', onClick: downloadPdf, variant: 'secondary' },
    ],
    'key-metrics': [
      { label: 'Refresh metrics', onClick: () => runAnalysis('/api/key-metrics'), variant: 'primary' },
      { label: 'Export CSV', onClick: downloadCsv, variant: 'secondary' },
    ],
    churn: [
      { label: 'Refresh insights', onClick: () => runAnalysis('/api/full-analysis'), variant: 'primary' },
      { label: 'Export PDF', onClick: downloadPdf, variant: 'secondary' },
    ],
    contracts: [
      { label: 'Refresh contracts', onClick: () => runAnalysis('/api/contract-analysis'), variant: 'primary' },
      { label: 'Export CSV', onClick: downloadCsv, variant: 'secondary' },
    ],
    payments: [
      { label: 'Refresh payments', onClick: () => runAnalysis('/api/payment-analysis'), variant: 'primary' },
      { label: 'Export PDF', onClick: downloadPdf, variant: 'secondary' },
    ],
    ml: [
      { label: 'Refresh ML models', onClick: () => runAnalysis('/api/ml-models'), variant: 'primary' },
      { label: 'Export CSV', onClick: downloadCsv, variant: 'secondary' },
    ],
  }

  const renderSection = () => {
    if (loading && !data['/api/full-analysis']) {
      return <LoadingSkeleton rows={3} />
    }

    switch (activeTab) {
      case 'data-summary':
        return (
          <div className="space-y-6">
            <div className="grid gap-5 xl:grid-cols-4">
              <MetricCard title="Rows in dataset" value={formatNumber(dataSummary.total_rows)} label="Total observations in churn dataset" />
              <MetricCard title="Columns" value={`${dataSummary.total_columns}`} label="Feature count across profiles" />
              <MetricCard title="Churn rate" value={`${dataSummary.churn_rate}%`} label="Churned customers ÷ all customers in supplied snapshot" />
              <MetricCard title="Missing values" value={`${dataSummary.missing_values}`} label="Data completeness score" accent="Clean" />
            </div>

            <div className="grid gap-5 lg:grid-cols-[1.35fr_0.65fr]">
              <ChartCard title="Feature distribution" description="Major feature counts and data types" badge="Dataset health">
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={[
                    { name: 'Numeric', value: dataSummary.numeric_columns ?? 0 },
                    { name: 'Categorical', value: dataSummary.categorical_columns ?? 0 },
                  ]} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.3} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                    <Bar dataKey="value" fill="#6366f1" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="space-y-5">
                <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                  <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Summary</p>
                  <div className="mt-5 space-y-4">
                    <div className="flex items-center justify-between rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                      <span className="text-sm text-slate-600 dark:text-slate-300">Unique customers</span>
                      <span className="text-lg font-semibold text-slate-950 dark:text-white">{formatNumber(dataSummary.unique_customers)}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                      <span className="text-sm text-slate-600 dark:text-slate-300">Memory footprint</span>
                      <span className="text-lg font-semibold text-slate-950 dark:text-white">{dataSummary.memory_usage}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                      <span className="text-sm text-slate-600 dark:text-slate-300">Effective sample quality</span>
                      <span className="text-lg font-semibold text-slate-950 dark:text-white">{dataSummary.validation?.duplicate_customer_ids ?? '—'} duplicate IDs</span>
                    </div>
                  </div>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                  <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Filter state</p>
                  <div className="mt-5 space-y-4">
                    <div className="rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                      <p className="text-sm text-slate-600 dark:text-slate-300">Contract</p>
                      <p className="mt-1 font-semibold text-slate-950 dark:text-white">{filters.contract}</p>
                    </div>
                    <div className="rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                      <p className="text-sm text-slate-600 dark:text-slate-300">Payment method</p>
                      <p className="mt-1 font-semibold text-slate-950 dark:text-white">{filters.payment}</p>
                    </div>
                    <div className="rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                      <p className="text-sm text-slate-600 dark:text-slate-300">Churn status</p>
                      <p className="mt-1 font-semibold text-slate-950 dark:text-white">{filters.tenure} / {filters.segment}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'key-metrics':
        return (
          <div className="space-y-6">
            <div className="grid gap-5 md:grid-cols-3">
              <MetricCard title="Churn rate" value={`${keyMetrics.churn_rate}%`} label={`${formatNumber(keyMetrics.churned_customers)} ÷ ${formatNumber(keyMetrics.total_customers)} customers`} />
              <MetricCard title="Active customers" value={formatNumber(keyMetrics.active_customers)} label="Churn = No in supplied snapshot" />
              <MetricCard title="Historical churned monthly charges" value={`$${Number(keyMetrics.historical_churned_monthly_revenue).toLocaleString()}`} label="MonthlyCharges for customers marked Churn = Yes" />
            </div>

            <div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
              <ChartCard title="Monthly revenue by customer state" description="Active versus already-churned MonthlyCharges in the supplied snapshot" badge="Monthly snapshot">
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={[{ name: 'Active', value: keyMetrics.active_monthly_revenue }, { name: 'Historical churn', value: keyMetrics.historical_churned_monthly_revenue }]} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" opacity={0.3} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                    <Bar dataKey="value" fill="#8b5cf6" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="space-y-5">
                <MetricCard title="Active monthly revenue at risk" value={`$${Number(keyMetrics.active_monthly_revenue_at_risk).toLocaleString()}`} label="Active customer charges; descriptive exposure, not expected loss" />
                <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                  <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Revenue at risk</p>
                  <p className="mt-4 text-3xl font-semibold text-slate-950 dark:text-white">{keyMetrics.revenue_loss_percentage}%</p>
                  <p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-400">Historical churned MonthlyCharges ÷ all customer MonthlyCharges. No recoverability claim.</p>
                </div>
              </div>
            </div>
          </div>
        )

      case 'churn':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
              <h2 className="text-lg font-semibold">Filter customer population</h2>
              <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                {[
                  ['contract', 'Contract', ['All', 'Month-to-month', 'One year', 'Two year']],
                  ['tenure', 'Tenure', ['All', '0–12 months', '13–24 months', '25–48 months', '49+ months']],
                  ['payment', 'Payment method', ['All', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']],
                  ['segment', 'Heuristic segment', ['All', 'Higher heuristic score', 'Lower heuristic score']],
                ].map(([key, label, options]) => <label key={key} className="text-sm font-medium text-slate-700 dark:text-slate-200">{label}<select value={filters[key]} onChange={(event) => { const next = { ...filters, [key]: event.target.value }; setFilters(next); runSegmentFilter(next) }} className="mt-1 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-900" aria-label={label}>{options.map((option) => <option key={option}>{option}</option>)}</select></label>)}
                <div className="rounded-xl bg-indigo-50 p-3 text-sm dark:bg-indigo-950/50" aria-live="polite"><strong>{segmentResult?.count ?? keyMetrics.total_customers}</strong> customers<br />{segmentResult ? `${segmentResult.metrics.churn_rate}% observed churn` : 'Apply filters to inspect segment churn'}</div>
              </div>
            </div>
            <div className="grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
              <ChartCard title="Customer churn by tenure" description="Segment exposure by months with provider" badge="Retention planning">
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={tenureData.map((row) => ({ name: row.name, churn: row.churn_rate, customers: row.customers }))} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.3} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                    <Bar dataKey="churn" fill="#f97316" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="space-y-5">
                <MetricCard title="Filtered customers" value={formatNumber(segmentResult?.count ?? keyMetrics.total_customers)} label="After selected contract, tenure, payment and heuristic filters" />
                <MetricCard title="Observed churn" value={`${segmentResult?.metrics.churn_rate ?? keyMetrics.churn_rate}%`} label="Historical churn share for this filtered population" />
              </div>
            </div>

            <InsightPanel insights={filteredChurnReasons} />

            <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
              <ChartCard title="Heuristic score interpretation" description="Manually weighted score for triage; not a predicted probability" badge="Transparent assumptions">
                <div className="space-y-3 text-sm leading-6 text-slate-600 dark:text-slate-300"><p>Weights: month-to-month +30; fiber +15; electronic check +12; no tech support +10; tenure under 12 months +10; senior customer +8; monthly charges above $75 +8.</p><p>Segment overlap is expected. The score ranks rule matches and does not measure treatment impact or prove these attributes cause churn.</p><p>Filtered churn metrics use observed labels in the selected population. For active-customer planning, filter churn to No through the API.</p></div>
              </ChartCard>

              <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Retention health</p>
                <div className="mt-6 space-y-4">
                  <div className="rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                    <p className="text-sm text-slate-600 dark:text-slate-300">Customers in tenure cohorts</p>
                    <p className="mt-2 text-xl font-semibold text-slate-950 dark:text-white">{formatNumber((analysis.tenure_analysis?.data || []).reduce((n, r) => n + (Number(r.customers) || 0), 0))}</p>
                  </div>
                  <div className="rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                    <p className="text-sm text-slate-600 dark:text-slate-300">Observed churn in filtered population</p>
                    <p className="mt-2 text-xl font-semibold text-slate-950 dark:text-white">{segmentResult?.metrics.churn_rate ?? keyMetrics.churn_rate}%</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'contracts':
        return (
          <div className="space-y-6">
            <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
              <ChartCard title="Contract churn" description="Monthly churn by plan length" badge="Plan retention">
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={contractComparison} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.3} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                    <Bar dataKey="churn" fill="#f97316" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="space-y-5">
                <MetricCard title="Month-to-month churn" value={`${contractComparison.find((row) => row.name === 'Month-to-month')?.churn ?? '—'}%`} label={`${contractComparison.find((row) => row.name === 'Month-to-month')?.customers?.toLocaleString() ?? '—'} customers; observed association`} />
                <MetricCard title="Two-year churn" value={`${contractComparison.find((row) => row.name === 'Two year')?.churn ?? '—'}%`} label={`${contractComparison.find((row) => row.name === 'Two year')?.customers?.toLocaleString() ?? '—'} customers; observed association`} />
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
              <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Contract distribution</p>
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                {contractComparison.map((item) => (
                  <div key={item.name} className="rounded-3xl bg-slate-100 p-4 dark:bg-slate-900">
                    <p className="text-sm text-slate-600 dark:text-slate-300">{item.name}</p>
                    <p className="mt-2 text-2xl font-semibold text-slate-950 dark:text-white">{item.churn}% churn</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )

      case 'payments':
        return (
          <div className="space-y-6">
            <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
              <ChartCard title="Billing channel mix" description="Payment methods and churn exposure" badge="Payment analytics">
                <ResponsiveContainer width="100%" height={340}>
                  <PieChart>
                    <Pie data={paymentMethods} dataKey="value" nameKey="name" innerRadius={60} outerRadius={90} paddingAngle={4} stroke="transparent">
                      {paymentMethods.map((entry) => (
                        <Cell key={entry.name} fill={entry.name === 'Electronic check' ? '#f43f5e' : entry.name === 'Mailed check' ? '#fbbf24' : entry.name === 'Bank transfer' ? '#22c55e' : '#38bdf8'} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="space-y-5">
                <MetricCard title="Electronic check churn" value={`${paymentMethods.find((row) => row.name === 'Electronic check')?.churn ?? '—'}%`} label={`${paymentMethods.find((row) => row.name === 'Electronic check')?.value?.toLocaleString() ?? '—'} customers; observed association`} />
                <MetricCard title="Payment methods" value={paymentMethods.length.toLocaleString()} label="Categories in supplied snapshot" />
              </div>
            </div>

            <div className="grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
              <ChartCard title="Observed churn by payment method" description="Historical churn rate by payment category; not a predicted likelihood" badge="Observed association">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={paymentMethods.map((row) => ({ name: row.name.replace(' (automatic)', ''), value: row.churn }))} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" opacity={0.3} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                    <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={4} dot={{ r: 4, fill: '#3b82f6' }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
              <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Billing insights</p>
                <p className="mt-4 text-sm leading-6 text-slate-600 dark:text-slate-400">Payment method is associated with different observed churn rates in this snapshot. This comparison does not establish that changing payment method will change churn.</p>
              </div>
            </div>
          </div>
        )

      case 'ml':
        return (
          <div className="space-y-6">
            <div className="grid gap-5 md:grid-cols-2">
              {mlSummary.map((metric) => (
                <MetricCard key={metric.title} title={metric.title} value={metric.value} label={metric.accent} />
              ))}
            </div>

            <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
              <ChartCard title="Features associated with model risk" description="Relative feature importance; association does not establish causation" badge="Explainability">
                <div className="space-y-4">
                  {featureImportance.map((feature) => (
                    <div key={feature.feature} className="space-y-2">
                      <div className="flex items-center justify-between text-sm text-slate-500 dark:text-slate-400">
                        <span>{feature.feature}</span>
                        <span>{feature.score}%</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-cyan-500 to-emerald-400" style={{ width: `${feature.score}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </ChartCard>

              <div className="space-y-5">
                <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                  <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Selected model</p>
                  <p className="mt-4 text-2xl font-semibold text-slate-950 dark:text-white">{modelResults.selected_model || '—'}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-400">Decision threshold {selectedModel.threshold ?? '—'}; selected from training validation only.</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                  <p className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">Auto-generated actions</p>
                  <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600 dark:text-slate-400">
                    <li>Use model scores to prioritize review, then evaluate campaign effects with a controlled experiment.</li>
                    <li>Feature importance and segment differences are descriptive associations, not causal effects.</li>
                    <li>Use actual contact costs and missed-churn economics before selecting a production threshold.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return (
          <div className="space-y-6">
            <div className="grid gap-5 md:grid-cols-3">
              {activeMetrics.map((item) => (
                <MetricCard key={item.title} title={item.title} value={item.value} label={item.label} accent={item.accent} />
              ))}
            </div>

            <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
              <ChartCard title="Observed churn by tenure cohort" description="The supplied source is a point-in-time snapshot, not a monthly time series" badge="Cohort comparison">
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart data={tenureData.map((row) => ({ name: row.name, churn: row.churn_rate }))} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" opacity={0.3} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                    <Tooltip contentStyle={{ borderRadius: 16 }} />
                    <Line type="monotone" dataKey="churn" stroke="#6366f1" strokeWidth={4} dot={{ r: 4, fill: '#6366f1' }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="space-y-5">
                <ChartCard title="Contract mix" description="Churn by plan type" badge="Subscriber mix">
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={contractComparison} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.3} />
                      <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                      <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b' }} />
                      <Tooltip contentStyle={{ borderRadius: 16 }} />
                      <Bar dataKey="churn" fill="#f97316" radius={[10, 10, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
                <ChartCard title="Payment method mix" description="Billing channel distribution" badge="Revenue ops">
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={paymentMethods} dataKey="value" nameKey="name" innerRadius={40} outerRadius={70} paddingAngle={4} stroke="transparent">
                        {paymentMethods.map((entry) => (
                          <Cell key={entry.name} fill={entry.name === 'Electronic check' ? '#f43f5e' : entry.name === 'Mailed check' ? '#fbbf24' : entry.name === 'Bank transfer' ? '#22c55e' : '#38bdf8'} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ borderRadius: 16 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </ChartCard>
              </div>
            </div>

            <div className="space-y-5">
              <SectionHeader title="Insights" subtitle="Most important signals from churn profiling and retention modeling." breadcrumbs={['Home', 'Overview']} />
              <InsightPanel insights={filteredChurnReasons} />
            </div>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen overflow-hidden bg-slate-100 text-slate-900 transition-colors duration-300 dark:bg-slate-950 dark:text-slate-100">
      <div className="relative flex h-full overflow-hidden">
        <Sidebar
          items={navItems}
          activeTab={activeTab}
          onTabChange={(key) => {
            setActiveTab(key)
            setSidebarOpen(false)
          }}
          open={sidebarOpen}
          onToggle={() => setSidebarOpen(false)}
        />

        <div className="flex-1 overflow-y-auto">
          <div className="sticky top-0 z-10 border-b border-slate-200/80 bg-slate-100/95 px-4 py-3 backdrop-blur dark:border-slate-800/80 dark:bg-slate-950/95 lg:px-5">
            <div className="flex items-center justify-between gap-3">
              <button className="inline-flex h-11 w-11 items-center justify-center rounded-3xl bg-white text-slate-900 shadow-sm transition hover:bg-slate-50 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800 lg:hidden" onClick={() => setSidebarOpen((value) => !value)}>
                ☰
              </button>
              <div className="flex items-center gap-3">
                <div className="relative hidden md:block">
                  <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(event) => setSearchTerm(event.target.value)}
                    placeholder="Search insights..."
                    className="h-11 w-72 rounded-3xl border border-slate-200 bg-white pl-11 pr-4 text-sm text-slate-900 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-200 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:border-indigo-500 dark:focus:ring-indigo-500/20"
                  />
                </div>
                <button
                  onClick={() => setDarkMode((prev) => !prev)}
                  className="inline-flex h-11 w-11 items-center justify-center rounded-3xl bg-white text-slate-900 shadow-sm transition hover:bg-slate-50 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800"
                  aria-label="Toggle theme"
                >
                  {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                </button>
                <button className="inline-flex h-11 w-11 items-center justify-center rounded-3xl bg-white text-slate-900 shadow-sm transition hover:bg-slate-50 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800" aria-label="Notifications">
                  <Bell className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>

          <main className="max-w-7xl mx-auto px-4 py-6 lg:px-6">
            <header className="mb-6 rounded-[2rem] border border-slate-200 bg-gradient-to-br from-slate-50 via-white to-slate-100 p-6 shadow-xl shadow-slate-900/5 dark:border-slate-800 dark:from-slate-900 dark:via-slate-950 dark:to-slate-950/80 lg:p-8">
              <div className="grid gap-6 lg:grid-cols-[1.4fr_0.9fr] items-start">
                <div className="space-y-4 max-w-2xl">
                  <p className="text-sm uppercase tracking-[0.32em] text-slate-500 dark:text-slate-400">Customer Retention Intelligence</p>
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white lg:text-4xl">Understand churn. Identify risk. Improve retention.</h1>
                  <p className="max-w-xl text-base leading-6 text-slate-600 dark:text-slate-400">Analyze telecom customer behavior, uncover high-churn segments, evaluate machine-learning models, and explore actionable retention insights through interactive analytics powered by Python, SQL, FastAPI, and React.</p>
                  <div className="flex flex-wrap gap-3">
                    <button onClick={runFullAnalysis} className="inline-flex items-center gap-2 rounded-3xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-500">
                      <Layers className="h-4 w-4" /> Run full analysis
                    </button>
                    <button onClick={() => { setActiveTab('data-summary'); runAnalysis('/api/data-summary') }} className="inline-flex items-center gap-2 rounded-3xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:border-indigo-200 hover:bg-indigo-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800">
                      <ChevronDown className="h-4 w-4" /> Data summary
                    </button>
                    <button onClick={() => { setActiveTab('key-metrics'); runAnalysis('/api/key-metrics') }} className="inline-flex items-center gap-2 rounded-3xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:border-indigo-200 hover:bg-indigo-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800">
                      <Download className="h-4 w-4" /> Key metrics
                    </button>
                  </div>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-3xl border border-slate-200 bg-white/95 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                    <p className="text-xs uppercase tracking-[0.32em] text-slate-500 dark:text-slate-400">Analytics Dashboard</p>
                    <p className="mt-3 text-base font-semibold leading-6 text-slate-950 dark:text-white">Explore customer KPIs, churn patterns, and model evaluation results.</p>
                    <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">{loading ? 'Analysis running' : errorMessage ? 'Action needed' : statusMessage}</p>
                  </div>
                  <div className="rounded-3xl border border-slate-200 bg-white/95 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-950/80">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-xs uppercase tracking-[0.32em] text-slate-500 dark:text-slate-400">Active segment</p>
                        <p className="mt-3 text-xl font-semibold text-slate-950 dark:text-white">{activeTab === 'overview' ? 'Customer Insights' : sectionMeta[activeTab]?.title || 'Customer Insights'}</p>
                        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">{activeTab === 'overview' ? 'Segment-level churn analysis' : sectionMeta[activeTab]?.subtitle || 'Segment-level churn analysis'}</p>
                      </div>
                      <span className="rounded-full bg-emerald-100 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200">Live</span>
                    </div>
                  </div>
                </div>
              </div>
            </header>

            <SectionHeader
              title={sectionMeta[activeTab]?.title || 'Overview'}
              subtitle={sectionMeta[activeTab]?.subtitle || sectionMeta.overview.subtitle}
              breadcrumbs={sectionMeta[activeTab]?.breadcrumbs || sectionMeta.overview.breadcrumbs}
              actions={pageActions[activeTab]?.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  onClick={item.onClick}
                  className={`inline-flex items-center rounded-3xl px-4 py-2.5 text-sm font-semibold transition ${item.variant === 'primary' ? 'bg-indigo-600 text-white hover:bg-indigo-500' : 'border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800'}`}
                >
                  {item.label}
                </button>
              ))}
            />

            {errorMessage && <div role="alert" className="mb-5 rounded-2xl border border-rose-300 bg-rose-50 p-4 text-rose-900 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-100">{errorMessage}</div>}

            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -18 }}
                transition={{ duration: 0.35 }}
                className="space-y-6"
              >
                {renderSection()}
              </motion.div>
            </AnimatePresence>

            <footer className="mt-6 rounded-3xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm dark:border-slate-800 dark:bg-slate-950/80 dark:text-slate-400">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <p>Built with React, Tailwind CSS, Recharts, Framer Motion, Axios, jsPDF.</p>
                <p>Interactive churn dashboard with section-level navigation.</p>
              </div>
            </footer>
          </main>
        </div>

        <button
          className="fixed bottom-6 right-6 z-20 inline-flex items-center gap-3 rounded-full bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-2xl shadow-indigo-500/30 transition hover:bg-indigo-500"
          aria-label="Open AI assistant"
        >
          <MessageCircle className="h-5 w-5" /> AI assistant
        </button>
      </div>
    </div>
  )
}

export default App
