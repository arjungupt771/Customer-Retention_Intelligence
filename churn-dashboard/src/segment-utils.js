export function buildSegmentParams(filters) {
  return {
    contract: filters.contract === 'All' ? undefined : filters.contract,
    payment_method: filters.payment === 'All' ? undefined : filters.payment,
    tenure_group: filters.tenure === 'All' ? undefined : filters.tenure,
    segment: filters.segment === 'All' ? undefined : filters.segment,
  }
}

export function getApiErrorMessage(error, fallback = 'Unable to load analysis') {
  return error?.response?.data?.detail || error?.message || fallback
}

export function normalizeKeyMetrics(payload) {
  const metrics = payload?.key_metrics ?? payload?.data ?? payload ?? {}
  const numberOrZero = (value) => {
    const number = Number(value)
    return Number.isFinite(number) ? number : 0
  }

  return {
    total_customers: numberOrZero(metrics.total_customers ?? metrics.total),
    active_customers: numberOrZero(metrics.active_customers ?? metrics.retained_customers ?? metrics.active),
    churned_customers: numberOrZero(metrics.churned_customers ?? metrics.churned),
    churn_rate: numberOrZero(metrics.churn_rate ?? metrics.churn_rate_pct),
    active_monthly_revenue: numberOrZero(metrics.active_monthly_revenue),
    active_monthly_revenue_at_risk: numberOrZero(metrics.active_monthly_revenue_at_risk ?? metrics.active_monthly_revenue),
    historical_churned_monthly_revenue: numberOrZero(metrics.historical_churned_monthly_revenue ?? metrics.monthly_revenue_lost),
    revenue_loss_percentage: numberOrZero(metrics.revenue_loss_percentage),
  }
}

export function formatNumber(value) {
  const number = Number(value)
  return (Number.isFinite(number) ? number : 0).toLocaleString()
}
