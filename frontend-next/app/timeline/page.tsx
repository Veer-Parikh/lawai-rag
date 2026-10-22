"use client";

import { ArrowSquareOut, CircleNotch } from "@phosphor-icons/react";
import Link from "next/link";
import { FormEvent, useCallback, useEffect, useState } from "react";
import { ActSummary, fetchTimeline, listActs, Timeline } from "../api";
import { Shell } from "../_components/Shell";
import { EmptyState, ErrorBanner, errorText, formatDate, RELATION_LABEL, StatusTag } from "../_components/ui";

const QUICK = [
  ["BNS", "103"],
  ["BNS", "106"],
  ["IPC", "302"],
  ["BNSS", "173"],
  ["IEA", "65B"],
] as const;

const NODE: Record<string, string> = {
  in_force: "node-good",
  replaced: "node-good",
  repealed: "node-bad",
  omitted: "node-bad",
  exception: "node-warn",
  predecessor: "node-past",
  predecessor_repealed: "node-past",
};

function TimelineView() {
  const [acts, setActs] = useState<ActSummary[]>([]);
  const [code, setCode] = useState("BNS");
  const [section, setSection] = useState("");
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const show = useCallback(async (nextCode: string, nextSection: string) => {
    const trimmed = nextSection.trim();
    if (!trimmed) {
      setError("Enter a section number, such as 103 or 65B.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      setTimeline(await fetchTimeline(nextCode, trimmed));
      window.history.replaceState(null, "", `/timeline?code=${nextCode}&section=${encodeURIComponent(trimmed)}`);
    } catch (requestError) {
      setError(errorText(requestError, "The timeline could not be loaded."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    listActs().then(setActs).catch((requestError) => setError(errorText(requestError, "Acts could not be loaded.")));
    const params = new URLSearchParams(window.location.search);
    const initialCode = params.get("code");
    const initialSection = params.get("section");
    if (initialCode && initialSection) {
      setCode(initialCode);
      setSection(initialSection);
      show(initialCode, initialSection);
    }
  }, [show]);

  function submit(event: FormEvent) {
    event.preventDefault();
    show(code, section);
  }

  const act = acts.find((item) => item.code === timeline?.code);

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="display">Timeline</h1>
          <p className="lede">The dated history of one provision: enactment, commencement, exceptions, repeal and what replaced it.</p>
        </div>
      </div>

      <form className="tl-form" onSubmit={submit}>
        <div className="field">
          <label htmlFor="tl-act">Act</label>
          <select id="tl-act" value={code} onChange={(event) => setCode(event.target.value)}>
            {(acts.length ? acts : [{ code: "BNS", title: "Bharatiya Nyaya Sanhita, 2023" } as ActSummary]).map((item) => (
              <option key={item.code} value={item.code}>{item.code}, {item.title}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="tl-section">Section</label>
          <input id="tl-section" value={section} onChange={(event) => setSection(event.target.value)} placeholder="103" inputMode="text" />
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading && <CircleNotch className="spin" size={16} aria-hidden="true" />}
          Show timeline
        </button>
      </form>
      <div className="quick-picks">
        <span className="muted">Try</span>
        {QUICK.map(([quickCode, quickSection]) => (
          <button
            key={`${quickCode}-${quickSection}`}
            type="button"
            onClick={() => {
              setCode(quickCode);
              setSection(quickSection);
              show(quickCode, quickSection);
            }}
          >
            {quickCode} {quickSection}
          </button>
        ))}
      </div>
      <ErrorBanner message={error} />

      {!timeline && !loading && !error && (
        <EmptyState title="Pick a provision">
          Choose an Act and a section number. Old sections show what replaced them; new sections show what they replaced.
        </EmptyState>
      )}

      {timeline && (
        <section className="section" aria-labelledby="tl-heading">
          <div className="tl-title">
            <h2 id="tl-heading">{timeline.code} section {timeline.section}</h2>
            <StatusTag status={timeline.status} label={timeline.status_label} />
          </div>
          <p className="lede">{[timeline.heading, timeline.act_title].filter(Boolean).join(", ")}</p>

          <div className="tl-layout">
            <ol className="timeline" aria-label="Events">
              {timeline.events.map((event, index) => (
                <li className={`tl-event ${NODE[event.kind] ?? ""}`} key={`${event.date}-${event.kind}-${index}`}>
                  <time className="tl-date" dateTime={event.date}>{formatDate(event.date)}</time>
                  <span className="tl-axis" aria-hidden="true"><span className="tl-node" /></span>
                  <div className="tl-body">
                    <strong>{event.title}</strong>
                    {event.detail && <p>{event.detail}</p>}
                    {event.instrument && (
                      <span className="tag tag-ink">
                        {event.url ? <a href={event.url} target="_blank" rel="noreferrer">{event.instrument}</a> : event.instrument}
                      </span>
                    )}
                    {event.code && event.section && (event.code !== timeline.code || event.section !== timeline.section) && event.kind !== "exception" && (
                      <p>
                        <button
                          type="button"
                          className="btn-quiet"
                          onClick={() => {
                            const [nextCode, nextSection] = [event.code as string, event.section as string];
                            setCode(nextCode);
                            setSection(nextSection);
                            show(nextCode, nextSection);
                          }}
                        >
                          Open {event.code} section {event.section}
                        </button>
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ol>

            <div>
              <div className="provision" style={{ marginTop: 0 }}>
                <h3>
                  <span>Text</span>
                  {timeline.page_anchor && <span className="muted">{timeline.page_anchor}</span>}
                </h3>
                {timeline.text ? (
                  <div className="law-text"><p>{timeline.text}</p></div>
                ) : (
                  <p className="hint">
                    {act && act.sections_in_catalog === 0
                      ? `The ${timeline.code} text is not in the catalog. ${["IPC", "CrPC"].includes(timeline.code) ? "Its PDF in the corpus is empty. " : ""}An administrator can add it from Corpus.`
                      : "This section was not found in the catalog text."}
                  </p>
                )}
                <p className="hint">
                  <a href={timeline.source_url} target="_blank" rel="noreferrer">
                    Official text on India Code <ArrowSquareOut size={12} aria-label="(opens in a new tab)" />
                  </a>
                </p>
              </div>

              {timeline.related.length > 0 && (
                <div className="provision">
                  <h3><span>Correspondences</span></h3>
                  <ul className="plain-list">
                    {timeline.related.map((row) => (
                      <li key={row.id}>
                        {row.from_code ? `${row.from_code} ${row.from_section}` : "No earlier provision"} to{" "}
                        {row.to_code ? `${row.to_code} ${row.to_section}` : "no new provision"}:{" "}
                        {RELATION_LABEL[row.relation].label.toLowerCase()}
                        {!row.verified && " (not yet checked)"}
                      </li>
                    ))}
                  </ul>
                  <p className="hint"><Link href={`/crosswalk?q=${encodeURIComponent(`${timeline.code} ${timeline.section}`)}`}>Compare the texts in the crosswalk</Link></p>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </>
  );
}

export default function TimelinePage() {
  return (
    <Shell>
      <TimelineView />
    </Shell>
  );
}
