export function Panel({ title, sub, hint, children }) {
  return (
    <section className="panel">
      <h2 className="panel-title">
        {title}
        {sub ? <span className="panel-sub"> · {sub}</span> : null}
      </h2>
      {hint ? <p className="panel-hint">{hint}</p> : null}
      {children}
    </section>
  )
}

/** A stage that has not run yet, naming the command that produces it. */
export function NotRun({ command }) {
  return (
    <p className="not-run">
      Not run yet: <code>{command}</code>
    </p>
  )
}

export function Table({ head, rows, align = [] }) {
  return (
    <div className="table-scroll">
      <table className="table">
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={h} className={align[i] === 'r' ? 'num' : undefined}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c} className={align[c] === 'r' ? 'num' : undefined}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Status never rides on colour alone: the word is the label, the dot is the reinforcement. */
export function Verdict({ pass, yes = 'pass', no = 'fail' }) {
  if (typeof pass !== 'boolean') return <span className="verdict verdict--none">n/a</span>
  return (
    <span className={pass ? 'verdict verdict--pass' : 'verdict verdict--fail'}>
      <span aria-hidden="true" className="verdict-dot" />
      {pass ? yes : no}
    </span>
  )
}
