"use client";

import { ArrowSquareOut, DownloadSimple, Printer } from "@phosphor-icons/react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import {
  deleteEntry,
  deleteMatter,
  downloadMemo,
  fetchMatter,
  Matter,
  MatterEntry,
  updateEntryNote,
  updateMatter,
} from "../../api";
import { useAuth } from "../../auth";
import { Prose } from "../../_components/Prose";
import { Shell } from "../../_components/Shell";
import { downloadText, EmptyState, ErrorBanner, errorText, evidenceState, formatDate, formatDateTime, StatusTag } from "../../_components/ui";

function useAutosave(save: (value: string) => Promise<unknown>) {
  const [state, setState] = useState<"" | "saving" | "saved" | "failed">("");
  return {
    state,
    label: { "": "", saving: "Saving…", saved: "Saved", failed: "Not saved. Try again." }[state],
    async commit(value: string) {
      setState("saving");
      try {
        await save(value);
        setState("saved");
      } catch {
        setState("failed");
      }
    },
  };
}

function ConfirmButton({ label, question, onConfirm }: { label: string; question: string; onConfirm: () => Promise<void> }) {
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  if (!asking) return <button type="button" className="btn btn-small btn-danger" onClick={() => setAsking(true)}>{label}</button>;
  return (
    <span className="row-actions" role="group" aria-label={question}>
      <span>{question}</span>
      <button
        type="button"
        className="btn btn-small btn-danger"
        onClick={async () => {
          try {
            await onConfirm();
          } catch (requestError) {
            setError(errorText(requestError, "That did not work."));
          }
        }}
      >
        {label}
      </button>
      <button type="button" className="btn-quiet" onClick={() => setAsking(false)}>Keep it</button>
      {error && <span className="field-error" role="alert">{error}</span>}
    </span>
  );
}

function EntryBlock({ matterId, entry, onRemoved }: { matterId: string; entry: MatterEntry; onRemoved: () => void }) {
  const [note, setNote] = useState(entry.note);
  const autosave = useAutosave((value) => updateEntryNote(matterId, entry.id, value));
  const answer = entry.answer;
  const state = evidenceState(answer);

  return (
    <article className="entry" aria-labelledby={`entry-${entry.id}`}>
      <div className="entry-head">
        <h3 id={`entry-${entry.id}`}>{entry.question}</h3>
        <div className="no-print">
          <ConfirmButton label="Remove" question="Remove this answer from the matter?" onConfirm={async () => {
            await deleteEntry(matterId, entry.id);
            onRemoved();
          }} />
        </div>
      </div>
      <div className="matter-meta">
        <span className={`tag tag-${state.key === "supported" ? "ok" : state.key === "partial" ? "warn" : "muted"}`}>{state.label}</span>
        <span>Saved {formatDateTime(entry.created_at)}</span>
        {entry.as_of && <span>Law as on {formatDate(entry.as_of)}</span>}
      </div>
      <div className="entry-grid">
        <aside className="authorities" aria-label="Authorities cited">
          <h3>Authorities cited</h3>
          {answer.citations.length === 0 && <p className="hint">None cited.</p>}
          {answer.citations.map((citation) => (
            <div className="authority" key={citation.text}>
              <span className={`authority-text${citation.verified ? " mark" : ""}`}>{citation.text}</span>
              <div><span className={`tag ${citation.verified ? "tag-ok" : "tag-warn"}`}>{citation.verified ? "Matched to a source" : "Not found in sources"}</span></div>
            </div>
          ))}
        </aside>
        <div>
          <div className="law-text"><Prose text={answer.answer} /></div>
          {answer.sources.length > 0 && (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Authority</th><th>Section</th><th>Status</th><th>Effective</th></tr></thead>
                <tbody>
                  {answer.sources.map((source) => (
                    <tr key={source.chunk_id}>
                      <td>
                        {source.act_name ?? source.case_name ?? source.source_file}
                        {source.source_url && (
                          <> <a href={source.source_url} target="_blank" rel="noreferrer" aria-label="Official text (opens in a new tab)"><ArrowSquareOut size={12} aria-hidden="true" /></a></>
                        )}
                      </td>
                      <td className="num">{source.section ?? ""}</td>
                      <td><StatusTag status={source.legal_status} /></td>
                      <td className="num">{formatDate(source.effective_date)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="field notes" style={{ marginTop: 16 }}>
            <label htmlFor={`note-${entry.id}`}>Your note</label>
            <textarea
              id={`note-${entry.id}`}
              value={note}
              rows={3}
              onChange={(event) => setNote(event.target.value)}
              onBlur={() => note !== entry.note && autosave.commit(note)}
              placeholder="How this answer applies to the matter"
            />
            <p className="save-state" role="status">{autosave.label}</p>
          </div>
        </div>
      </div>
    </article>
  );
}

function MatterView() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { session } = useAuth();
  const [matter, setMatter] = useState<Matter | null>(null);
  const [notes, setNotes] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState("");
  const [reference, setReference] = useState("");
  const [error, setError] = useState("");
  const autosave = useAutosave((value) => updateMatter(id, { notes: value }));

  useEffect(() => {
    fetchMatter(id)
      .then((loaded) => {
        setMatter(loaded);
        setNotes(loaded.notes);
        setTitle(loaded.title);
        setReference(loaded.reference);
      })
      .catch((requestError) => setError(errorText(requestError, "The matter could not be loaded.")));
  }, [id]);

  async function rename(event: FormEvent) {
    event.preventDefault();
    try {
      const updated = await updateMatter(id, { title, reference });
      setMatter((current) => current && { ...current, title: updated.title, reference: updated.reference });
      setRenaming(false);
    } catch (requestError) {
      setError(errorText(requestError, "The matter was not renamed."));
    }
  }

  async function exportMemo() {
    setError("");
    try {
      const memo = await downloadMemo(id);
      downloadText(memo.filename, memo.text);
    } catch (requestError) {
      setError(errorText(requestError, "The memo could not be downloaded."));
    }
  }

  if (error && !matter) {
    return (
      <>
        <ErrorBanner message={error} />
        <Link href="/matters">Back to matters</Link>
      </>
    );
  }
  if (!matter) return <div className="loadbar" />;

  return (
    <>
      <div className="memo-only">
        <p className="hint">Research memo prepared by {session?.name} on {formatDate(new Date().toISOString())} with LawAI. Research support only, not legal advice.</p>
      </div>
      <div className="page-head">
        <div>
          <p className="no-print"><Link href="/matters">Matters</Link></p>
          {renaming ? (
            <form className="matter-new" onSubmit={rename} style={{ marginTop: 8 }}>
              <div className="field">
                <label htmlFor="rename-title">Title</label>
                <input id="rename-title" value={title} onChange={(event) => setTitle(event.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="rename-ref">Reference</label>
                <input id="rename-ref" value={reference} onChange={(event) => setReference(event.target.value)} />
              </div>
              <div className="head-actions">
                <button type="submit" className="btn btn-primary">Save</button>
                <button type="button" className="btn-quiet" onClick={() => setRenaming(false)}>Cancel</button>
              </div>
            </form>
          ) : (
            <h1 className="display">{matter.title}</h1>
          )}
          <div className="matter-meta">
            {matter.reference && <span>Reference {matter.reference}</span>}
            <span>{matter.entries.length} saved {matter.entries.length === 1 ? "answer" : "answers"}</span>
            <span>Updated {formatDateTime(matter.updated_at)}</span>
          </div>
        </div>
        <div className="head-actions no-print">
          <button type="button" className="btn btn-primary" onClick={exportMemo}><DownloadSimple size={16} aria-hidden="true" /> Download memo</button>
          <button type="button" className="btn" onClick={() => window.print()}><Printer size={16} aria-hidden="true" /> Print or save as PDF</button>
          {!renaming && <button type="button" className="btn-quiet" onClick={() => setRenaming(true)}>Rename</button>}
        </div>
      </div>
      <ErrorBanner message={error} />

      <section className="notes" aria-labelledby="notes-heading">
        <label id="notes-heading" htmlFor="matter-notes">Matter notes</label>
        <textarea
          id="matter-notes"
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          onBlur={() => notes !== matter.notes && autosave.commit(notes).then(() => setMatter({ ...matter, notes }))}
          placeholder="Facts, charges, the question you need answered"
        />
        <p className="save-state" role="status">{autosave.label}</p>
      </section>

      {matter.entries.length === 0 ? (
        <EmptyState title="Nothing saved yet" action={<Link className="btn" href="/research">Ask a question</Link>}>
          Ask a question in Research, then choose Save to matter on the answer.
        </EmptyState>
      ) : (
        matter.entries.map((entry) => (
          <EntryBlock
            key={entry.id}
            matterId={matter.id}
            entry={entry}
            onRemoved={() => setMatter({ ...matter, entries: matter.entries.filter((item) => item.id !== entry.id) })}
          />
        ))
      )}

      <section className="section no-print" aria-labelledby="danger-heading">
        <h2 id="danger-heading">Delete this matter</h2>
        <p className="hint" style={{ marginBottom: 12 }}>This removes the matter and every answer saved in it. It cannot be undone.</p>
        <ConfirmButton
          label="Delete matter"
          question={`Delete "${matter.title}" and its ${matter.entries.length} saved answers?`}
          onConfirm={async () => {
            await deleteMatter(matter.id);
            router.replace("/matters");
          }}
        />
      </section>
    </>
  );
}

export default function MatterPage() {
  return (
    <Shell>
      <MatterView />
    </Shell>
  );
}
