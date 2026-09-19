import { EventStudy } from './EventStudy.jsx'
import { ByYear, ModelTable } from './ModelTable.jsx'
import { PublishBar } from './PublishBar.jsx'
import { Deviations, Gates, Robustness } from './Robustness.jsx'
import { useResults } from './results.js'

const QUESTION =
  "Does the text of a news headline predict a stock's next-day return relative to the market, " +
  'once every source of lookahead is removed?'

export default function App() {
  const { loading, files } = useResults()

  if (loading) return <main className="page" />

  // G4 picks the primary; before it is recorded the page still has to name something.
  const primary = files.summary?.primary ?? (files.gates?.G4?.pass === false ? 'tfidf' : 'chrono')

  return (
    <main className="page">
      {import.meta.env.MODE === 'demo' ? (
        <p className="demo-banner">
          Demo mode: every number below is synthetic, from <span className="mono">frontend/fixtures/</span>.
          Run <span className="mono">npm run dev</span> to read the real{' '}
          <span className="mono">results/</span>.
        </p>
      ) : null}
      <header className="masthead">
        <h1>News sentiment signal</h1>
        <p className="question">{QUESTION}</p>
        <p className="meta">
          Primary model <span className="mono">{primary}</span> · headlines matched to the first
          trading day whose 15:50 New York cut-off they precede · 1,000 most liquid names above $5
        </p>
      </header>

      <PublishBar summary={files.summary} />
      <EventStudy summary={files.summary} />
      <ModelTable files={files} primary={primary} />
      <ByYear run={files[primary]} primary={primary} />
      <Robustness summary={files.summary} holdout={files.holdout} />
      <Gates gates={files.gates} />
      <Deviations deviations={files.deviations} />

      <footer className="footer">
        <p>
          Every number on this page is read from <span className="mono">results/*.json</span>. The
          written report, including the limitations this page does not repeat, is{' '}
          <a href="https://github.com/benoitrobinson/news-sentiment-signal/blob/main/REPORT.md">
            REPORT.md
          </a>
          .
        </p>
      </footer>
    </main>
  )
}
