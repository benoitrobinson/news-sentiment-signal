import { NotRun, Panel, Table } from './Panel.jsx'
import { fmt, fmtInt } from './results.js'

const NOTE = {
  null: 'retrained on labels permuted across all training rows; |t| must stay below 2',
  old_bug_vader: "repeats the original repo's same-day join",
}

export function ModelTable({ files, primary }) {
  // dict.fromkeys order in report.py: primary, its null, tfidf, vader, old-bug vader.
  const names = [...new Set([primary, `${primary}_null`, 'tfidf', 'vader', 'old_bug_vader'])]
  const rows = names
    .map((name) => [name, files[name]?.overall])
    .filter(([, overall]) => overall)
    .map(([name, o]) => [
      <span key="n" className="mono">
        {name}
      </span>,
      fmtInt(o.days),
      fmt(o.mean_ic),
      fmt(o.ic_t, 2),
      fmt(o.net_sharpe, 2),
      fmt(o.hit_rate, 3),
    ])

  return (
    <Panel
      title="Models"
      sub="walk-forward 2013-2020, out of sample"
      hint={`${primary}_null ${NOTE.null}. old_bug_vader ${NOTE.old_bug_vader}.`}
    >
      {rows.length ? (
        <Table
          head={['Run', 'Days', 'Mean IC', 'IC t (NW)', 'Net Sharpe', 'Hit rate']}
          rows={rows}
          align={['l', 'r', 'r', 'r', 'r', 'r']}
        />
      ) : (
        <NotRun command={`uv run ss run --model ${primary}`} />
      )}
    </Panel>
  )
}

export function ByYear({ run, primary }) {
  const rows = (run?.by_year ?? []).map((y) => [
    y.test_year,
    fmtInt(y.days),
    fmt(y.mean_ic),
    fmt(y.ic_t, 2),
    fmt(y.net_sharpe, 2),
  ])

  return (
    <Panel title="By test year" sub={primary}>
      {rows.length ? (
        <Table
          head={['Test year', 'Days', 'Mean IC', 'IC t (NW)', 'Net Sharpe']}
          rows={rows}
          align={['l', 'r', 'r', 'r', 'r']}
        />
      ) : (
        <NotRun command={`uv run ss run --model ${primary}`} />
      )}
    </Panel>
  )
}
