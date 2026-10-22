import Link from "next/link";
import { ArrowRight, CheckCircle, FileText, MagnifyingGlass, Scales } from "@phosphor-icons/react/dist/ssr";

export default function Home() {
  return (
    <main id="main" className="home">
      <header className="home-nav">
        <Link href="/" className="wordmark">LawAI</Link>
        <nav aria-label="Public navigation">
          <Link href="#how-it-works">How it works</Link>
          <Link href="#coverage">Coverage</Link>
          <Link href="/login" className="btn btn-small">Sign in</Link>
        </nav>
      </header>

      <section className="home-hero">
        <div className="home-hero-copy">
          <p className="eyebrow"><span className="eyebrow-rule" /> Indian law, with the source beside it</p>
          <h1>Research the law without losing the trail.</h1>
          <p className="home-lede">
            LawAI connects every answer to the BNS, BNSS, BSA, the laws they replaced, and the Gazette notifications that put them into force.
          </p>
          <div className="home-actions">
            <Link href="/login" className="btn btn-primary">Start researching <ArrowRight size={17} aria-hidden="true" /></Link>
            <Link href="#how-it-works" className="btn">See how it works</Link>
          </div>
          <p className="home-note"><CheckCircle size={16} aria-hidden="true" /> Built for careful first-pass research, not legal advice.</p>
        </div>
        <div className="home-specimen" aria-label="Example of a sourced LawAI answer">
          <div className="specimen-topline"><span>LAWAI / EVIDENCE VIEW</span><span>01</span></div>
          <p className="specimen-kicker">A provision is more than a number.</p>
          <h2>Section 103, BNS</h2>
          <p className="specimen-body">Punishment for murder — with the former IPC provision, commencement date, and source instrument kept in view.</p>
          <div className="specimen-source">
            <span className="tag tag-ok">Matched to a source</span>
            <span>Notification S.O. 850(E)</span>
            <span>In force · 1 July 2024</span>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="home-section">
        <div className="home-section-head">
          <p className="eyebrow">A research workflow with guardrails</p>
          <h2>Find the provision. Understand the change. Verify the authority.</h2>
        </div>
        <div className="home-grid">
          <article className="home-card">
            <MagnifyingGlass size={24} aria-hidden="true" />
            <p className="card-index">01 / Search</p>
            <h3>Ask in plain language</h3>
            <p>Start with a question about an offence, procedure, evidence, or a provision transition.</p>
          </article>
          <article className="home-card">
            <Scales size={24} aria-hidden="true" />
            <p className="card-index">02 / Compare</p>
            <h3>See what changed</h3>
            <p>Trace BNS, BNSS, and BSA provisions back to the laws they replaced and their effective dates.</p>
          </article>
          <article className="home-card">
            <FileText size={24} aria-hidden="true" />
            <p className="card-index">03 / Cite</p>
            <h3>Keep the source close</h3>
            <p>Every answer shows its authorities, retrieval strength, and warnings when evidence is incomplete.</p>
          </article>
        </div>
      </section>

      <section id="coverage" className="home-proof">
        <div>
          <p className="eyebrow">Designed for the transition</p>
          <h2>One workspace for the new criminal codes and the record behind them.</h2>
        </div>
        <div className="home-proof-list">
          <span>BNS</span><span>BNSS</span><span>BSA</span><span>IPC</span><span>CrPC</span><span>IEA</span><span>Gazette notifications</span>
        </div>
      </section>

      <footer className="home-footer">
        <span className="wordmark">LawAI</span>
        <span>Research support, not legal advice.</span>
        <Link href="/login">Open the workspace <ArrowRight size={15} aria-hidden="true" /></Link>
      </footer>
    </main>
  );
}
