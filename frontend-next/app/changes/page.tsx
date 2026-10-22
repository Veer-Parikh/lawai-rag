"use client";

import { ArrowSquareOut } from "@phosphor-icons/react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ChangeEvent, fetchChanges } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../_components/Shell";
import { EmptyState, ErrorBanner, errorText, formatDate } from "../_components/ui";

const OUTCOME: Record<string, { label: string; tone: string }> = {
  added: { label: "Added to corpus", tone: "ok" },
  changed: { label: "Changed at source", tone: "warn" },
  new: { label: "First recorded", tone: "ink" },
};

const AUTHORITY: Record<string, string> = {
  gazette: "Gazette of India",
  india_code: "India Code",
  mha: "Ministry of Home Affairs",
  supreme_court: "Supreme Court",
};

function ChangesView() {
  const { session } = useAuth();
  const [events, setEvents] = useState<ChangeEvent[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchChanges(100)
      .then((body) => setEvents(body.events))
      .catch((requestError) => setError(errorText(requestError, "The change feed could not be loaded.")));
  }, []);

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="display">Changes</h1>
          <p className="lede">Official documents that were added to the corpus or changed at their source, newest first.</p>
        </div>
      </div>
      <ErrorBanner message={error} />
      {events && events.length === 0 && (
        <EmptyState
          title="No changes recorded yet"
          action={session?.role === "admin" ? <Link className="btn" href="/admin/sources">Check official sources</Link> : undefined}
        >
          Changes appear here when an administrator checks the Gazette, India Code and MHA, or adds a new notification to the corpus.
        </EmptyState>
      )}
      {events && events.length > 0 && (
        <ol className="feed">
          {events.map((event) => {
            const outcome = OUTCOME[event.outcome] ?? { label: event.outcome, tone: "muted" };
            return (
              <li key={event.id}>
                <time dateTime={event.checked_at}>{formatDate(event.checked_at)}</time>
                <span className={`tag tag-${outcome.tone}`}>{outcome.label}</span>
                <div>
                  <strong>{event.title}</strong>
                  <span className="cell-sub">
                    {[event.authority && (AUTHORITY[event.authority] ?? event.authority), event.detail].filter(Boolean).join(". ")}
                  </span>
                  {event.url && (
                    <a href={event.url} target="_blank" rel="noreferrer">
                      Open the source <ArrowSquareOut size={12} aria-label="(opens in a new tab)" />
                    </a>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </>
  );
}

export default function ChangesPage() {
  return (
    <Shell>
      <ChangesView />
    </Shell>
  );
}
