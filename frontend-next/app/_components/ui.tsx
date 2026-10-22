"use client";

import { CheckCircle, CircleNotch, WarningOctagon, XCircle } from "@phosphor-icons/react";
import { ReactNode } from "react";
import type { BackgroundJob, SavedAnswer } from "../api";

// ── Dates ────────────────────────────────────────────────────────────────────

export function formatDate(value?: string | null) {
  if (!value) return "";
  const date = new Date(value.length === 10 ? `${value}T00:00:00` : value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export function formatDateTime(value?: string | null) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

// ── Status ───────────────────────────────────────────────────────────────────

const STATUS: Record<string, { label: string; tone: string }> = {
  in_force: { label: "In force", tone: "ok" },
  partly_in_force: { label: "Partly in force", tone: "warn" },
  not_in_force: { label: "Not in force", tone: "warn" },
  not_yet_in_force: { label: "Not yet in force", tone: "warn" },
  notified: { label: "Notified", tone: "ink" },
  enacted: { label: "Enacted", tone: "ink" },
  superseded: { label: "Superseded", tone: "warn" },
  repealed: { label: "Repealed", tone: "bad" },
};

export function StatusTag({ status, label }: { status?: string | null; label?: string }) {
  const known = status ? STATUS[status] : undefined;
  return <span className={`tag tag-${known?.tone ?? "muted"}`}>{label ?? known?.label ?? "Status unknown"}</span>;
}

export const RELATION_LABEL: Record<string, { label: string; tone: string }> = {
  equivalent: { label: "Equivalent", tone: "ink" },
  modified: { label: "Modified", tone: "warn" },
  omitted: { label: "Not carried over", tone: "bad" },
  new: { label: "New offence or power", tone: "ok" },
};

export type EvidenceKey = "supported" | "partial" | "abstained";

/** The evidence state shown above every answer (same rule as the memo export). */
export function evidenceState(result: Pick<SavedAnswer, "confidence_tier" | "sources" | "citations" | "citation_warnings">): {
  key: EvidenceKey;
  label: string;
  detail: string;
} {
  if (result.confidence_tier === "LOW" || result.sources.length === 0) {
    return { key: "abstained", label: "Not answered", detail: "The corpus did not hold strong enough evidence, so no answer was generated." };
  }
  const unmatched = result.citations.filter((citation) => !citation.verified).length;
  if (result.confidence_tier === "MEDIUM" || unmatched > 0 || result.citation_warnings.length > 0) {
    return {
      key: "partial",
      label: "Partly supported",
      detail: unmatched ? `${unmatched} of ${result.citations.length} citations could not be matched to a retrieved source.` : "Retrieval was only moderately strong. Check each authority.",
    };
  }
  return { key: "supported", label: "Supported by sources", detail: "Every citation matched a retrieved source." };
}

// ── Messages ─────────────────────────────────────────────────────────────────

export function ErrorBanner({ message }: { message: string }) {
  if (!message) return null;
  return (
    <p className="banner banner-bad" role="alert">
      <WarningOctagon size={18} aria-hidden="true" />
      <span>{message}</span>
    </p>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function JobProgress<R>({ job, describe }: { job: BackgroundJob<R>; describe: (job: BackgroundJob<R>) => string }) {
  const running = job.status === "running";
  const percent = job.progress_total ? Math.round((job.progress_done / job.progress_total) * 100) : 0;
  const Icon = running ? CircleNotch : job.status === "completed" ? CheckCircle : XCircle;
  return (
    <section className={`job job-${job.status}`} aria-live="polite">
      <div className="job-head">
        <Icon className={running ? "spin" : undefined} size={18} weight={running ? "regular" : "fill"} aria-hidden="true" />
        <strong>{describe(job)}</strong>
        <span className="hint">Started by {job.requested_by}, {formatDateTime(job.started_at)}</span>
      </div>
      {running && (
        <>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-label="Progress">
            <span style={{ width: `${percent}%` }} />
          </div>
          {job.current && <p className="hint">Working on {job.current}</p>}
        </>
      )}
      {job.error && <p className="job-errors">{job.error}</p>}
    </section>
  );
}

// ── Files ────────────────────────────────────────────────────────────────────

export function downloadText(filename: string, text: string, type = "text/markdown;charset=utf-8") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function errorText(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}
