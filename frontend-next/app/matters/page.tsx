"use client";

import { CircleNotch } from "@phosphor-icons/react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { createMatter, listMatters, Matter } from "../api";
import { Shell } from "../_components/Shell";
import { EmptyState, ErrorBanner, errorText, formatDateTime } from "../_components/ui";

function MattersView() {
  const router = useRouter();
  const [matters, setMatters] = useState<Matter[] | null>(null);
  const [title, setTitle] = useState("");
  const [reference, setReference] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [formError, setFormError] = useState("");

  useEffect(() => {
    listMatters().then(setMatters).catch((requestError) => setError(errorText(requestError, "Matters could not be loaded.")));
  }, []);

  async function create(event: FormEvent) {
    event.preventDefault();
    if (!title.trim()) {
      setFormError("Give the matter a title.");
      return;
    }
    setPending(true);
    setFormError("");
    try {
      const matter = await createMatter({ title: title.trim(), reference: reference.trim() });
      router.push(`/matters/${matter.id}`);
    } catch (requestError) {
      setFormError(errorText(requestError, "The matter was not created."));
      setPending(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="display">Matters</h1>
          <p className="lede">Keep the answers you rely on together, add your notes, and export a research memo.</p>
        </div>
      </div>

      <form className="matter-new" onSubmit={create} noValidate>
        <div className="field">
          <label htmlFor="matter-title">New matter</label>
          <input id="matter-title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="State v. Sharma, bail application" aria-invalid={!!formError} />
        </div>
        <div className="field">
          <label htmlFor="matter-ref">Reference (optional)</label>
          <input id="matter-ref" value={reference} onChange={(event) => setReference(event.target.value)} placeholder="CR-17/2026" />
        </div>
        <button type="submit" className="btn btn-primary" disabled={pending}>
          {pending && <CircleNotch className="spin" size={16} aria-hidden="true" />}
          Create matter
        </button>
      </form>
      {formError && <p className="field-error" role="alert">{formError}</p>}
      <ErrorBanner message={error} />

      <section className="section" aria-labelledby="matters-heading">
        <div className="section-head">
          <h2 id="matters-heading">Your matters {matters && <span className="count">{matters.length}</span>}</h2>
        </div>
        {!matters && !error && <div className="loadbar" />}
        {matters && matters.length === 0 && (
          <EmptyState title="No matters yet" action={<Link className="btn" href="/research">Ask a question</Link>}>
            Create one above, or save an answer from Research into a new matter.
          </EmptyState>
        )}
        {matters && matters.length > 0 && (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Matter</th><th>Reference</th><th className="num">Saved answers</th><th>Last updated</th></tr></thead>
              <tbody>
                {matters.map((matter) => (
                  <tr key={matter.id}>
                    <td><Link href={`/matters/${matter.id}`}><strong>{matter.title}</strong></Link></td>
                    <td>{matter.reference}</td>
                    <td className="num">{matter.entry_count}</td>
                    <td className="num">{formatDateTime(matter.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

export default function MattersPage() {
  return (
    <Shell>
      <MattersView />
    </Shell>
  );
}
