import { NotRun, Panel, Table, Verdict } from './Panel.jsx'
import { fmt } from './results.js'

// The four pre-registered conditions, keyed as evaluate.publish_bar writes them.
const LABELS = {
  '1_ic_t_ge_2': ['Daily rank IC t-stat (Newey-West)', '>= 2'],
  '2_net_sharpe_ge_0_5': ['Quintile long-short net Sharpe, after 10 bps', '>= 0.5'],
  '3_lag_ic_le_half': ['Mean IC one day later', '<= half the mean IC'],
  '4_holdout_ic_gt_0': ['Holdout mean IC, different news source', '> 0'],
}

export function PublishBar({ summary }) {
  if (!summary?.bar?.conditions) {
    return (
      <Panel title="Publish bar" sub="four conditions, fixed before any result was seen">
        <NotRun command="uv run ss summarize" />
      </Panel>
    )
  }

  const { conditions, publish } = summary.bar
  const rows = Object.entries(conditions).map(([key, { value, pass }]) => {
    const [label, threshold] = LABELS[key] ?? [key, '']
    return [label, threshold, fmt(value), <Verdict key="v" pass={pass} />]
  })

  return (
    <Panel title="Publish bar" sub="four conditions, fixed before any result was seen">
      <p className={publish ? 'headline headline--pass' : 'headline headline--fail'}>
        {publish ? 'All four conditions clear.' : 'The bar does not clear.'}
      </p>
      <Table
        head={['Condition', 'Threshold', 'Value', '']}
        rows={rows}
        align={['l', 'l', 'r', 'l']}
      />
    </Panel>
  )
}
