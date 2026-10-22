"use client";

import { ArrowSquareOut, ArrowsLeftRight, CircleNotch, WarningOctagon } from "@phosphor-icons/react";
import { FormEvent, useEffect, useState } from "react";
import { compareDocuments, DocumentChangeResponse, DocumentSummary, listDocuments } from "../api";
import { Shell } from "../_components/Shell";

function DocumentFacts({ document }: { document?: DocumentSummary }) {
  if (!document) return null;
  return (
    <div className="doc-facts">
      <strong>{document.title}</strong>
      <span>{document.publisher}{document.effective_date ? `, effective ${document.effective_date}` : ""}</span>
      <a href={document.source_url} target="_blank" rel="noreferrer">
        Official text <ArrowSquareOut size={12} aria-label="(opens in a new tab)" />
      </a>
    </div>
  );
}

function SectionColumn({ title, sections, tone }: { title: string; sections: string[]; tone: string }) {
  return (
    <section className={`diff-column diff-${tone}`} aria-label={title}>
      <h3>{title} <span className="count">{sections.length}</span></h3>
      {sections.length === 0 ? (
        <p className="hint">None</p>
      ) : (
        <ul>{sections.map((section) => <li key={section}>Section {section}</li>)}</ul>
      )}
    </section>
  );
}

function CompareView() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [fromId, setFromId] = useState("");
  const [toId, setToId] = useState("");
  const [diff, setDiff] = useState<DocumentChangeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    listDocuments()
      .then((items) => {
        setDocuments(items);
        if (items.length >= 2) {
          setFromId(items[0].document_id);
          setToId(items[1].document_id);
        }
      })
      .catch((requestError) => setError(requestError.message));
  }, []);

  async function compare(event: FormEvent) {
    event.preventDefault();
    if (!fromId || !toId || fromId === toId) {
      setError("Choose two different document versions.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      setDiff(await compareDocuments(fromId, toId));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Comparison failed.");
    } finally {
      setLoading(false);
    }
  }

  const byId = (id: string) => documents.find((document) => document.document_id === id);

  return (
    <div className="research">
      <header className="page-head">
        <h1>Compare two versions</h1>
        <p>See which sections changed, were added, or were dropped between two versions held in the legal catalog.</p>
      </header>

      {documents.length < 2 && !error ? (
        <p className="empty-note">
          The catalog needs at least two document versions to compare. Versions are added when statutes are
          registered in the legal catalog.
        </p>
      ) : (
        <form className="compare-form" onSubmit={compare}>
          <div className="field">
            <label htmlFor="from-doc">Earlier version</label>
            <select id="from-doc" value={fromId} onChange={(event) => setFromId(event.target.value)}>
              {documents.map((document) => <option key={document.document_id} value={document.document_id}>{document.title} ({document.document_id})</option>)}
            </select>
            <DocumentFacts document={byId(fromId)} />
          </div>
          <ArrowsLeftRight className="compare-glyph" size={22} aria-hidden="true" />
          <div className="field">
            <label htmlFor="to-doc">Later version</label>
            <select id="to-doc" value={toId} onChange={(event) => setToId(event.target.value)}>
              {documents.map((document) => <option key={document.document_id} value={document.document_id}>{document.title} ({document.document_id})</option>)}
            </select>
            <DocumentFacts document={byId(toId)} />
          </div>
          <button type="submit" className="button button-primary" disabled={loading}>
            {loading && <CircleNotch className="spin" size={18} aria-hidden="true" />}
            Compare versions
          </button>
        </form>
      )}

      {error && <p className="banner banner-bad" role="alert"><WarningOctagon size={18} aria-hidden="true" /> {error}</p>}

      {diff && (
        <div className="diff-grid" aria-live="polite">
          <SectionColumn title="Changed" sections={diff.changed_sections} tone="warn" />
          <SectionColumn title="Added" sections={diff.added_sections} tone="ok" />
          <SectionColumn title="Removed" sections={diff.removed_sections} tone="bad" />
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Shell>
      <CompareView />
    </Shell>
  );
}
