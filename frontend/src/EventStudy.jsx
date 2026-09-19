import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { NotRun, Panel, Table } from './Panel.jsx'
import { fmt } from './results.js'

// Quintiles are ordered with a meaningful middle, so the ramp diverges: red arm for the
// lowest scores, grey at Q3, blue arm for the highest. Validated against this page's panel
// surface (#0d1117) with the dataviz validator: worst all-pairs CVD dE 8.7, normal-vision
// dE 15.4, all five clear 3:1. The validator's "lightness band" and "chroma floor" checks
// are categorical-only and do not apply to a diverging ramp, whose midpoint must read grey.
// Red/green was rejected: it is the pair colour-blind readers lose.
const QUINTILE = {
  1: '#e34948',
  2: '#f0a3a3',
  3: '#898781',
  4: '#9ec5f4',
  5: '#3987e5',
}

const bps = (r) => r * 1e4

/** [{q, offset, cum_r}] -> one row per offset, one key per quintile, in basis points. */
function pivot(rows) {
  const byOffset = new Map()
  for (const { q, offset, cum_r } of rows) {
    if (!byOffset.has(offset)) byOffset.set(offset, { offset })
    byOffset.get(offset)[`q${q}`] = bps(cum_r)
  }
  return [...byOffset.values()].sort((a, b) => a.offset - b.offset)
}

function TooltipBody({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="tooltip">
      <div className="tooltip-head">
        {label === 0 ? 'day 0 — first tradable return' : `day ${label}`}
      </div>
      {payload
        .slice()
        .reverse()
        .map((entry) => (
          <div key={entry.dataKey} className="tooltip-row">
            <span className="swatch" style={{ background: entry.color }} aria-hidden="true" />
            <span className="tooltip-name">{entry.name}</span>
            <span className="tooltip-value">{fmt(entry.value, 1)} bps</span>
          </div>
        ))}
    </div>
  )
}

export function EventStudy({ summary }) {
  const rows = summary?.event_study ?? []
  if (!rows.length) {
    return (
      <Panel title="Event study" sub="cumulative excess return by score quintile">
        <NotRun command="uv run ss summarize" />
      </Panel>
    )
  }

  const data = pivot(rows)
  const last = data.length - 1
  const quintiles = [1, 2, 3, 4, 5]

  return (
    <Panel
      title="Event study"
      sub="cumulative excess return by score quintile"
      hint="Day 0 is the first return the signal could trade. A spread that keeps widening after day 0 is drift; one that closes was a one-day effect."
    >
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 8, right: 44, left: 4, bottom: 4 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="offset"
            tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            tickLine={false}
            axisLine={{ stroke: 'var(--axis)' }}
            label={{
              value: 'trading days from the signal',
              position: 'insideBottom',
              offset: -2,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            tickLine={false}
            axisLine={false}
            width={52}
            label={{
              value: 'bps',
              angle: -90,
              position: 'insideLeft',
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
          />
          <ReferenceLine y={0} stroke="var(--axis)" />
          <ReferenceLine x={0} stroke="var(--axis)" strokeDasharray="3 3" />
          <Tooltip content={<TooltipBody />} cursor={{ stroke: 'var(--axis)' }} />
          <Legend
            iconType="plainline"
            wrapperStyle={{ fontSize: 12, color: 'var(--text-muted)' }}
          />
          {quintiles.map((q) => (
            <Line
              key={q}
              type="linear"
              dataKey={`q${q}`}
              name={q === 1 ? 'Q1 (lowest score)' : q === 5 ? 'Q5 (highest score)' : `Q${q}`}
              stroke={QUINTILE[q]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
              isAnimationActive={false}
              label={(props) =>
                props.index === last && props.value != null ? (
                  <text
                    x={props.x + 8}
                    y={props.y}
                    dy={4}
                    fontSize={11}
                    fill={QUINTILE[q]}
                    key={`l${q}`}
                  >
                    Q{q}
                  </text>
                ) : null
              }
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <details className="table-view">
        <summary>Table view</summary>
        <Table
          head={['Day', ...quintiles.map((q) => `Q${q}`)]}
          rows={data.map((row) => [
            row.offset,
            ...quintiles.map((q) => fmt(row[`q${q}`], 1)),
          ])}
          align={['l', 'r', 'r', 'r', 'r', 'r']}
        />
        <p className="panel-hint">Cumulative mean excess return, basis points.</p>
      </details>
    </Panel>
  )
}
