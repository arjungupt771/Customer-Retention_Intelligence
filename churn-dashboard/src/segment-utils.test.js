import assert from 'node:assert/strict'
import test from 'node:test'
import { buildSegmentParams, formatNumber, getApiErrorMessage, normalizeKeyMetrics } from './segment-utils.js'

test('segment filters omit All and map the remaining UI selections to API query names', () => {
  assert.deepEqual(buildSegmentParams({
    contract: 'Month-to-month',
    payment: 'All',
    tenure: '0–12 months',
    segment: 'Higher heuristic score',
  }), {
    contract: 'Month-to-month',
    payment_method: undefined,
    tenure_group: '0–12 months',
    segment: 'Higher heuristic score',
  })
})

test('API error messages prefer readable server detail', () => {
  assert.equal(getApiErrorMessage({ response: { data: { detail: 'Invalid contract' } } }), 'Invalid contract')
  assert.equal(getApiErrorMessage({ message: 'Network down' }), 'Network down')
  assert.equal(getApiErrorMessage({}), 'Unable to load analysis')
})

test('key metrics tolerate wrapped and older API payloads with missing fields', () => {
  assert.deepEqual(normalizeKeyMetrics({ key_metrics: { total: '12', retained_customers: 9, churned: 3, churn_rate_pct: '25', monthly_revenue_lost: 45 } }), {
    total_customers: 12,
    active_customers: 9,
    churned_customers: 3,
    churn_rate: 25,
    active_monthly_revenue: 0,
    active_monthly_revenue_at_risk: 0,
    historical_churned_monthly_revenue: 45,
    revenue_loss_percentage: 0,
  })
  assert.equal(formatNumber(undefined), '0')
})
