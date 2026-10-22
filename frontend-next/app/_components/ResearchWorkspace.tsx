"use client";

import { ArrowSquareOut, CheckCircle, CircleNotch, Printer, Warning } from "@phosphor-icons/react";
import Link from "next/link";
import { FormEvent, useEffect, useRef, useState } from "react";
import {
  askQuestion,
  ChatResponse,
  createMatter,
  FeedbackKind,
  listMatters,
  Matter,
  saveToMatter,
  sendFeedback,
  Source,
} from "../api";
import { Prose } from "./Prose";
import { ErrorBanner, errorText, evidenceState, formatDate, StatusTag } from "./ui";

const EXAMPLES = [
  "Which BNS section replaced IPC section 302, and what does it say?",
  "When did the BNSS come into force?",
  "How does the BSA treat electronic records as evidence?",
];

const PROBLEMS: { kind: Exclude<FeedbackKind, "helpful">; label: string }[] = [
  { kind: "wrong_source", label: "It cites the wrong source" },
  { kind: "stale_status", label: "The law's status is out of date" },
  { kind: "missing_authority", label: "An authority is missing" },
  { kind: "poor_explanation", label: "The explanation is unclear" },
];

const RECENT_KEY = "lawai.recent";

function sourceTitle(source: Source) {
  return source.act_name ?? source.case_name ?? source.source_file.split(/[\\/]/).pop() ?? "Source document";
}

function savedAnswer(result: ChatResponse) {
  const { answer, confidence_tier, citations, sources, citation_warnings, legal_explanation, disclaimer } = result;
  return { answer, confidence_tier, citations, sources, citation_warnings, legal_explanation, disclaimer };
}

function SaveToMatter({ question, asOf, result }: { question: string; asOf: string; result: ChatResponse }) {
  const [open, setOpen] = useState(false);
  const [matters, setMatters] = useState<Matter[] | null>(null);
  const [choice, setChoice] = useState("");
  const [title, setTitle] = useState("");
  const [saved, setSaved] = useState<Matter | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");

  async function openPanel() {
    setOpen(true);
    setError("");
    try {
      const list = await listMatters();
      setMatters(list);
      setChoice(list[0]?.id ?? "new");
    } catch (requestError) {
      setError(errorText(requestError, "Matters could not be loaded."));
    }
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      let matter = matters?.find((item) => item.id === choice) ?? null;
      if (choice === "new") {
        if (!title.trim()) {
          setError("Give the new matter a title.");
          setPending(false);
          return;
        }
        matter = await createMatter({ title: title.trim() });
      }
      if (!matter) throw new Error("Choose a matter.");
      await saveToMatter(matter.id, { question, as_of: asOf || null, answer: savedAnswer(result) });
      setSaved(matter);
      setOpen(false);
    } catch (requestError) {
      setError(errorText(requestError, "The answer was not saved."));
    } finally {
      setPending(false);
    }
  }

  if (saved) {
    return (
      <p className="save-done" role="status">
        Saved to <Link href={`/matters/${saved.id}`}>{saved.title}</Link>.
      </p>
    );
  }
  if (!open) {
    return <button type="button" className="btn" onClick={openPanel}>Save to matter</button>;
  }
  return (
    <form className="save-panel" onSubmit={save}>
      <div className="field">
        <label htmlFor="matter-choice">Matter</label>
        <select id="matter-choice" value={choice} onChange={(event) => setChoice(event.target.value)} disabled={!matters}>
          {matters?.map((matter) => <option key={matter.id} value={matter.id}>{matter.title}</option>)}
          <option value="new">New matter…</option>
        </select>
      </div>
      {choice === "new" && (
        <div className="field">
          <label htmlFor="matter-title">New matter title</label>
          <input id="matter-title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="State v. Sharma, bail" />
        </div>
      )}
      {error && <p className="field-error" role="alert">{error}</p>}
      <button type="submit" className="btn btn-primary" disabled={pending || !matters}>
        {pending && <CircleNotch className="spin" size={16} aria-hidden="true" />}
        Save answer
      </button>
      <button type="button" className="btn-quiet" onClick={() => setOpen(false)}>Cancel</button>
    </form>
  );
}

function Feedback({ question, result }: { question: string; result: ChatResponse }) {
  const [kind, setKind] = useState<Exclude<FeedbackKind, "helpful">>("wrong_source");
  const [share, setShare] = useState(true);
  const [comment, setComment] = useState("");
  const [sent, setSent] = useState<"helpful" | "problem" | null>(null);
  const [error, setError] = useState("");

  async function submit(feedbackKind: FeedbackKind) {
    setError("");
    try {
      await sendFeedback({
        query: question,
        kind: feedbackKind,
        comment: comment || undefined,
        share_question: feedbackKind !== "helpful" && share,
        answer: result.answer,
        citations: result.citations.map((citation) => citation.text),
        sources: result.sources.map((source) => ({ title: sourceTitle(source), section: source.section, status: source.legal_status })),
        confidence_tier: result.confidence_tier,
      });
      setSent(feedbackKind === "helpful" ? "helpful" : "problem");
    } catch (requestError) {
      setError(errorText(requestError, "Feedback was not recorded."));
    }
  }

  if (sent) {
    return (
      <p className="save-done" role="status">
        <CheckCircle size={16} weight="fill" aria-hidden="true" />{" "}
        {sent === "helpful" ? "Thanks, recorded." : "Reported. A reviewer will check it."}
      </p>
    );
  }
  return (
    <>
      <button type="button" className="btn-quiet" onClick={() => submit("helpful")}>This answer is correct</button>
      <details className="report">
        <summary>Report a problem</summary>
        <fieldset style={{ border: 0, margin: 0, padding: 0 }}>
          <legend className="sr-only">What is wrong?</legend>
          {PROBLEMS.map((problem) => (
            <label key={problem.kind} className="check">
              <input type="radio" name="problem" checked={kind === problem.kind} onChange={() => setKind(problem.kind)} />
              {problem.label}
            </label>
          ))}
        </fieldset>
        <div className="field" style={{ marginTop: 10 }}>
          <label htmlFor="problem-comment">What should it say? (optional)</label>
          <textarea id="problem-comment" rows={3} maxLength={500} value={comment} onChange={(event) => setComment(event.target.value)} />
        </div>
        <label className="check">
          <input type="checkbox" checked={share} onChange={(event) => setShare(event.target.checked)} />
          Share my question and this answer with reviewers
        </label>
        <p className="hint">Without it, reviewers see only which authorities were cited.</p>
        {error && <p className="field-error" role="alert">{error}</p>}
        <button type="button" className="btn btn-small" onClick={() => submit(kind)}>Send report</button>
      </details>
    </>
  );
}

function ResultView({ question, asOf, result, headingRef }: {
  question: string;
  asOf: string;
  result: ChatResponse;
  headingRef: React.RefObject<HTMLHeadingElement | null>;
}) {
  const state = evidenceState(result);
  return (
    <article className="result" aria-labelledby="result-heading">
      <div className="result-head">
        <div className={`evidence state-${state.key}`}>
          <h2 id="result-heading" ref={headingRef} tabIndex={-1}>{state.label}</h2>
          <p>{state.detail}</p>
        </div>
        <dl className="facts">
          <div><dt>Retrieval strength</dt><dd>{result.confidence_tier.charAt(0) + result.confidence_tier.slice(1).toLowerCase()}</dd></div>
          <div><dt>Passages</dt><dd>{result.context_chunks_used}</dd></div>
          <div><dt>Time</dt><dd>{(result.latency_ms / 1000).toFixed(1)} s</dd></div>
        </dl>
      </div>
      <p className="asked">{question}{asOf && ` (law as on ${formatDate(asOf)})`}</p>

      <div className="answer-layout">
        <aside className="authorities" aria-label="Authorities cited">
          <h3>Authorities cited</h3>
          {result.citations.length === 0 && <p className="hint">None cited.</p>}
          {result.citations.map((citation) => (
            <div className="authority" key={`${citation.text}-${citation.source_chunk_ids.join()}`}>
              <span className={`authority-text${citation.verified ? " mark" : ""}`}>{citation.text}</span>
              <div>
                <span className={`tag ${citation.verified ? "tag-ok" : "tag-warn"}`}>
                  {citation.verified ? "Matched to a source" : "Not found in sources"}
                </span>
              </div>
              {citation.source_urls.slice(0, 1).map((url) => (
                <a key={url} href={url} target="_blank" rel="noreferrer">
                  Open the authority <ArrowSquareOut size={12} aria-label="(opens in a new tab)" />
                </a>
              ))}
            </div>
          ))}
        </aside>

        <div>
          <div className="law-text"><Prose text={result.answer} /></div>
          {result.citation_warnings.length > 0 && (
            <div className="banner banner-warn" role="note">
              <Warning size={18} aria-hidden="true" />
              <div>
                <strong>Check before relying on this</strong>
                <ul>{result.citation_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
              </div>
            </div>
          )}
          {result.legal_explanation && result.legal_explanation !== result.answer && (
            <details className="explanation">
              <summary>Plain-language explanation</summary>
              <div className="law-text"><Prose text={result.legal_explanation} /></div>
            </details>
          )}
        </div>

        <aside className="answer-actions" aria-label="Actions">
          <h3>Use this answer</h3>
          <SaveToMatter question={question} asOf={asOf} result={result} />
          <button type="button" className="btn" onClick={() => window.print()}>
            <Printer size={16} aria-hidden="true" /> Print or save as PDF
          </button>
          <Feedback question={question} result={result} />
        </aside>
      </div>

      {result.sources.length > 0 && (
        <section className="section" aria-labelledby="sources-heading">
          <div className="section-head">
            <h2 id="sources-heading">Sources consulted <span className="count">{result.sources.length}</span></h2>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Authority</th><th>Section</th><th>Status</th><th>Effective</th><th>Publisher</th><th><span className="sr-only">Link</span></th></tr>
              </thead>
              <tbody>
                {result.sources.map((source) => (
                  <tr key={source.chunk_id}>
                    <td>{sourceTitle(source)}{source.page_anchor && <span className="cell-sub">{source.page_anchor}</span>}</td>
                    <td className="num">{source.section ?? ""}</td>
                    <td><StatusTag status={source.legal_status} /></td>
                    <td className="num">{formatDate(source.effective_date)}</td>
                    <td>{source.publisher ?? ""}</td>
                    <td className="actions">
                      {source.source_url && (
                        <a href={source.source_url} target="_blank" rel="noreferrer">
                          Official text <ArrowSquareOut size={12} aria-label="(opens in a new tab)" />
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <p className="hint" style={{ marginTop: 24 }}>{result.disclaimer}</p>
    </article>
  );
}

export function ResearchWorkspace() {
  const [query, setQuery] = useState("");
  const [asOf, setAsOf] = useState("");
  const [asked, setAsked] = useState({ question: "", asOf: "" });
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [recent, setRecent] = useState<string[]>([]);
  const sessionId = useRef("");
  const headingRef = useRef<HTMLHeadingElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    sessionId.current = crypto.randomUUID();
    try {
      setRecent(JSON.parse(window.localStorage.getItem(RECENT_KEY) ?? "[]"));
    } catch {
      setRecent([]);
    }
    const prefill = new URLSearchParams(window.location.search).get("q");
    if (prefill) {
      setQuery(prefill);
      inputRef.current?.focus();
    }
  }, []);

  function remember(question: string) {
    const next = [question, ...recent.filter((item) => item !== question)].slice(0, 5);
    setRecent(next);
    try {
      window.localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    } catch {
      // Recent questions are a convenience only.
    }
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = query.trim();
    if (question.length < 3) {
      setError("Write a question of at least three characters.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await askQuestion({ query: question, as_of: asOf || undefined, session_id: sessionId.current });
      setAsked({ question, asOf });
      setResult(response);
      remember(question);
      requestAnimationFrame(() => headingRef.current?.focus());
    } catch (requestError) {
      setError(errorText(requestError, "The question could not be checked."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="display">Ask the law</h1>
          <p className="lede">Answers come only from the BNS, BNSS, BSA, the laws they replaced, and Gazette notifications in the corpus.</p>
        </div>
      </div>

      <form className="composer" onSubmit={submit}>
        <div>
          <label htmlFor="question" className="sr-only">Your question</label>
          <textarea
            id="question"
            ref={inputRef}
            value={query}
            rows={3}
            placeholder="What is the punishment for organised crime under the BNS?"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) submit();
            }}
            aria-describedby="composer-hint"
          />
        </div>
        <div className="composer-side">
          <div>
            <label htmlFor="as-of">Law as on</label>
            <input id="as-of" type="date" value={asOf} max={new Date().toISOString().slice(0, 10)} onChange={(event) => setAsOf(event.target.value)} />
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading && <CircleNotch className="spin" size={18} aria-hidden="true" />}
            {loading ? "Checking sources" : "Check sources"}
          </button>
          <p id="composer-hint" className="hint">Ctrl + Enter also sends. Leave the date empty for today.</p>
        </div>
      </form>
      <ErrorBanner message={error} />

      {loading && (
        <div className="result" aria-live="polite" aria-busy="true">
          <div className="loadbar" />
          <p className="hint">Searching statutes and notifications, then matching each citation to its source.</p>
        </div>
      )}

      {!result && !loading && (
        <section className="examples" aria-labelledby="examples-heading">
          <h2 id="examples-heading">Questions lawyers often start with</h2>
          <ul className="example-list">
            {EXAMPLES.map((example) => (
              <li key={example}><button type="button" onClick={() => setQuery(example)}>{example}</button></li>
            ))}
          </ul>
          {recent.length > 0 && (
            <>
              <h2 style={{ marginTop: 32 }}>Your recent questions</h2>
              <ul className="example-list examples-quiet">
                {recent.map((item) => <li key={item}><button type="button" onClick={() => setQuery(item)}>{item}</button></li>)}
              </ul>
            </>
          )}
        </section>
      )}

      {result && !loading && <ResultView question={asked.question} asOf={asked.asOf} result={result} headingRef={headingRef} />}
    </>
  );
}
