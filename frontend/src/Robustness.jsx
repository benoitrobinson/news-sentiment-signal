import { NotRun, Panel, Table, Verdict } from './Panel.jsx'
import { fmt, fmtInt } from './results.js'

export function Robustness({ summary, holdout }) {
  if (!summary) {
    return (
      <Panel title="Robustness">
        <NotRun command="uv run ss summarize" />
      </Panel>
    )
  }

  return (
    <Panel title="Robustness">
      <dl className="stats">
        <div>
          <dt>Mean lag IC</dt>
          <dd>{fmt(summary.mean_lag_ic)}</dd>
          <dd className="stat-note">the same scores against the day after the traded one</dd>
        </div>
        <div>
          <dt>Deflated Sharpe ratio</dt>
          <dd>{fmt(summary.deflated_sharpe)}</dd>
          <dd className="stat-note">probability the Sharpe beats the best of the grid by luck</dd>
        </div>
        <div>
          <dt>Holdout mean IC</dt>
          <dd>{fmt(holdout?.mean_ic)}</dd>
          <dd className="stat-note">
            {holdout ? `${fmtInt(holdout.days)} days, Alpha Vantage` : 'ss holdout'}
          </dd>
        </div>
      </dl>

      {summary.terciles?.length ? (
        <Table
          head={['Liquidity tercile', 'Days', 'Mean IC', 'IC t (NW)', 'Net Sharpe']}
          rows={summary.terciles.map((t) => [
            t.tercile,
            fmtInt(t.days),
            fmt(t.mean_ic),
            fmt(t.ic_t, 2),
            fmt(t.net_sharpe, 2),
          ])}
          align={['l', 'r', 'r', 'r', 'r']}
        />
      ) : null}
    </Panel>
  )
}

export function Gates({ gates }) {
  const entries = Object.entries(gates ?? {})
  const expected = ['G1', 'G2ab', 'G2c', 'G4']
  const rows = expected.map((name) => {
    const gate = gates?.[name]
    return [
      <span key="n" className="mono">
        {name}
      </span>,
      <Verdict key="v" pass={gate?.pass} yes="pass" no="fail" />,
      gate?.recorded_at?.replace('T', ' ').replace('+00:00', ' UTC') ?? 'not recorded',
    ]
  })

  return (
    <Panel
      title="Gates"
      sub="recorded before any model result"
      hint="Nothing is modelled until all four are recorded and G1, G2ab and G2c pass. G4 decides the primary model."
    >
      {entries.length ? (
        <Table head={['Gate', '', 'Recorded']} rows={rows} align={['l', 'l', 'l']} />
      ) : (
        <NotRun command="uv run ss gate-g1 --licence ... --licence-ok" />
      )}
    </Panel>
  )
}

export function Deviations({ deviations }) {
  const list = deviations?.deviations ?? []
  return (
    <Panel title="Deviations" sub="changes made to the protocol after it was fixed">
      {list.length ? (
        <ul className="list">
          {list.map((d, i) => (
            <li key={i}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>
          ))}
        </ul>
      ) : (
        <p className="panel-hint">None.</p>
      )}
    </Panel>
  )
}
