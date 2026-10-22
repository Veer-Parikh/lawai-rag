"use client";

import {
  ArrowClockwise,
  ArrowSquareOut,
  CheckCircle,
  CircleNotch,
  CloudArrowUp,
  FilePdf,
  Globe,
  WarningOctagon,
  XCircle,
} from "@phosphor-icons/react";
import { DragEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Corpus,
  CorpusFile,
  fetchCorpus,
  fetchIndexStats,
  fetchJobs,
  IndexStats,
  IngestJob,
  startIngest,
  uploadDocument,
} from "../../api";
import { BarList } from "../../_components/charts";
import { Shell } from "../../_components/Shell";

type Filter = "all" | "waiting" | "searchable" | "failed" | "web";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All PDFs" },
  { key: "waiting", label: "Waiting" },
  { key: "searchable", label: "Searchable" },
  { key: "failed", label: "Needs attention" },
  { key: "web", label: "Web captures" },
];

const AUTHORITY: Record<string, string> = {
  india_code: "India Code",
  gazette: "Gazette of India",
  mha: "Ministry of Home Affairs",
  supreme_court: "Supreme Court",
  doj: "Department of Justice",
  pib: "PIB",
};

function bytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(0)} KB`;
  return `${(size / 1024 ** 2).toFixed(1)} MB`;
}

function when(iso?: string | null) {
  if (!iso) return "";
  return new Date(iso).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function stateOf(file: CorpusFile): { key: Filter; label: string; tone: string } {
  if (file.empty) return { key: "failed", label: "Empty file", tone: "bad" };
  if (!file.ingestible) return { key: "web", label: "Reference only", tone: "neutral" };
  switch (file.ingestion?.status) {
    case "ingested": return { key: "searchable", label: "Searchable", tone: "ok" };
    case "failed": return { key: "failed", label: "Failed", tone: "bad" };
    case "dry_run": return { key: "waiting", label: "Test run only", tone: "info" };
    default: return { key: "waiting", label: "Waiting", tone: "warn" };
  }
}

function JobProgress({ job }: { job: IngestJob }) {
  const running = job.status === "started" || job.status === "running";
  const percent = job.progress_total ? Math.round((job.progress_done / job.progress_total) * 100) : 0;
  return (
    <section className={`job job-${job.status}`} aria-live="polite">
      <div className="job-head">
        {running ? <CircleNotch className="spin" size={18} aria-hidden="true" /> : job.status === "completed" ? <CheckCircle size={18} weight="fill" aria-hidden="true" /> : <XCircle size={18} weight="fill" aria-hidden="true" />}
        <strong>
          {running
            ? `${job.dry_run ? "Test run" : "Ingesting"}: ${job.progress_done} of ${job.progress_total || "?"} files`
            : job.status === "completed"
              ? `${job.dry_run ? "Test run" : "Ingestion"} finished: ${job.files_processed} of ${job.files_found} files, ${job.total_chunks_created} passages${job.dry_run ? " parsed" : ` indexed`}`
              : "Ingestion stopped with an error"}
        </strong>
        <span className="hint">{job.target_path === "." ? "Whole corpus" : job.target_path}</span>
      </div>
      {running && (
        <>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-label="Ingestion progress">
            <span style={{ width: `${percent}%` }} />
          </div>
          {job.current_file && <p className="hint">Reading {job.current_file}</p>}
        </>
      )}
      {job.errors.length > 0 && (
        <ul className="job-errors">{job.errors.slice(0, 5).map((error) => <li key={error}>{error}</li>)}</ul>
      )}
    </section>
  );
}

function UploadForm({ destinations, onUploaded }: { destinations: Corpus["destinations"]; onUploaded: (path: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [destination, setDestination] = useState(destinations[0]?.folder ?? "");
  const [title, setTitle] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [publisher, setPublisher] = useState("");
  const [dragging, setDragging] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);

  function choose(next: File | undefined) {
    setError("");
    if (!next) return;
    if (!next.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files can be added to the corpus.");
      return;
    }
    setFile(next);
    if (!title) setTitle(next.name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " "));
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Choose a PDF to upload.");
      return;
    }
    setPending(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("destination", destination);
    if (title) form.append("title", title);
    if (sourceUrl) form.append("source_url", sourceUrl);
    if (publisher) form.append("publisher", publisher);
    try {
      const stored = await uploadDocument(form);
      setFile(null);
      setTitle("");
      setSourceUrl("");
      setPublisher("");
      onUploaded(stored.path);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Upload failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <form className="upload" onSubmit={submit} noValidate>
      <h2>Add a document</h2>
      <div
        className={`dropzone${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        {file ? <FilePdf size={28} aria-hidden="true" /> : <CloudArrowUp size={28} aria-hidden="true" />}
        <p>{file ? <><strong>{file.name}</strong> ({bytes(file.size)})</> : "Drop a PDF here, or"}</p>
        <button type="button" className="text-button" onClick={() => input.current?.click()}>
          {file ? "Choose a different file" : "browse your files"}
        </button>
        <input ref={input} type="file" accept="application/pdf,.pdf" className="sr-only" tabIndex={-1} onChange={(event) => choose(event.target.files?.[0])} aria-label="PDF file" />
      </div>
      <div className="field">
        <label htmlFor="destination">Where it belongs</label>
        <select id="destination" value={destination} onChange={(event) => setDestination(event.target.value)}>
          {destinations.map((item) => <option key={item.folder} value={item.folder}>{item.label}</option>)}
        </select>
        <p className="hint">The folder decides how the file is read: statute, judgment or crosswalk table.</p>
      </div>
      <div className="field">
        <label htmlFor="doc-title">Title</label>
        <input id="doc-title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Bharatiya Nyaya Sanhita (Amendment) Act, 2025" />
      </div>
      <div className="field-row">
        <div className="field">
          <label htmlFor="doc-url">Official source URL</label>
          <input id="doc-url" type="url" inputMode="url" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder="https://egazette.gov.in/…" />
        </div>
        <div className="field">
          <label htmlFor="doc-publisher">Publisher</label>
          <input id="doc-publisher" value={publisher} onChange={(event) => setPublisher(event.target.value)} placeholder="Gazette of India" />
        </div>
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
      <button type="submit" className="button button-primary" disabled={pending}>
        {pending && <CircleNotch className="spin" size={18} aria-hidden="true" />}
        Upload document
      </button>
    </form>
  );
}

function CorpusView() {
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [index, setIndex] = useState<IndexStats | null>(null);
  const [job, setJob] = useState<IngestJob | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [nextCorpus, jobs] = await Promise.all([fetchCorpus(), fetchJobs()]);
      setCorpus(nextCorpus);
      setJob((current) => jobs[0] ?? current);
      setError("");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The corpus could not be loaded.");
    } finally {
      setLoading(false);
    }
    fetchIndexStats().then(setIndex).catch(() => setIndex(null));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const running = job?.status === "started" || job?.status === "running";

  // Poll while a job runs, then refresh the inventory once it finishes.
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(async () => {
      try {
        const [latest] = await fetchJobs();
        if (!latest) return;
        setJob(latest);
        if (latest.status === "completed" || latest.status === "failed") load();
      } catch {
        // Keep polling; a transient failure should not hide the job.
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [running, load]);

  async function ingest(target: string, options: { dry_run?: boolean; skip_ingested?: boolean } = {}) {
    setError("");
    setNotice("");
    try {
      setJob(await startIngest(target, options));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Ingestion could not start.");
    }
  }

  const files = useMemo(() => {
    if (!corpus) return [];
    const term = search.trim().toLowerCase();
    return corpus.files.filter((file) => {
      const state = stateOf(file).key;
      const matchesFilter = filter === "all" ? file.kind === "pdf" : filter === state;
      const matchesSearch = !term || [file.name, file.title, file.folder].some((value) => value?.toLowerCase().includes(term));
      return matchesFilter && matchesSearch;
    });
  }, [corpus, filter, search]);

  const counts = useMemo(() => {
    const result: Record<Filter, number> = { all: 0, waiting: 0, searchable: 0, failed: 0, web: 0 };
    corpus?.files.forEach((file) => {
      if (file.kind === "pdf") result.all += 1;
      result[stateOf(file).key] += 1;
    });
    return result;
  }, [corpus]);

  const summary = corpus?.summary;

  return (
    <div className="admin">
      <header className="page-head page-head-row">
        <div>
          <h1>Corpus and ingestion</h1>
          <p>Every file in the data folder, where it came from, and whether researchers can search it yet.</p>
        </div>
        <div className="head-actions">
          <button type="button" className="button button-secondary" disabled={running || !summary?.pending} onClick={() => ingest(".", { skip_ingested: true })}>
            Ingest all waiting files{summary?.pending ? ` (${summary.pending})` : ""}
          </button>
          <button type="button" className="icon-button" onClick={load} aria-label="Refresh" title="Refresh">
            {loading ? <CircleNotch className="spin" size={18} aria-hidden="true" /> : <ArrowClockwise size={18} aria-hidden="true" />}
          </button>
        </div>
      </header>

      {error && <p className="banner banner-bad" role="alert"><WarningOctagon size={18} aria-hidden="true" /> {error}</p>}
      {notice && <p className="banner banner-ok" role="status"><CheckCircle size={18} aria-hidden="true" /> {notice}</p>}
      {job && <JobProgress job={job} />}

      {summary && corpus && (
        <>
          <dl className="figures">
            <div><dt>PDF documents</dt><dd>{summary.pdfs}</dd></div>
            <div><dt>Searchable</dt><dd>{summary.ingested}</dd></div>
            <div><dt>Waiting</dt><dd>{summary.pending}</dd></div>
            <div><dt>Failed or empty</dt><dd className={summary.failed + summary.empty ? "text-bad" : undefined}>{summary.failed + summary.empty}</dd></div>
            <div><dt>Passages indexed</dt><dd>{summary.chunks_uploaded.toLocaleString("en-IN")}</dd></div>
            <div><dt>Corpus size</dt><dd>{bytes(summary.bytes)}</dd></div>
          </dl>

          <div className="panel-grid corpus-top">
            <section className="panel">
              <UploadForm
                destinations={corpus.destinations}
                onUploaded={(path) => {
                  setNotice(`Uploaded ${path}. It is waiting to be ingested.`);
                  load();
                }}
              />
            </section>
            <section className="panel" aria-labelledby="index-heading">
              <h2 id="index-heading">Vector index</h2>
              {!index ? (
                <p className="hint">Loading index statistics…</p>
              ) : index.available ? (
                <>
                  <p className="index-total"><strong>{index.total_vectors.toLocaleString("en-IN")}</strong> passages in <code>{index.index}</code></p>
                  <BarList items={Object.entries(index.namespaces).map(([label, value]) => ({ label, value }))} empty="The index exists but holds no passages yet." />
                </>
              ) : (
                <p className="banner banner-warn"><WarningOctagon size={18} aria-hidden="true" /> The index <code>{index.index}</code> could not be reached: {index.error}</p>
              )}
              {corpus.unavailable.length > 0 && (
                <details className="unavailable">
                  <summary>{corpus.unavailable.length} official sources could not be downloaded</summary>
                  <ul>
                    {corpus.unavailable.map((source) => (
                      <li key={source.source_id}>
                        <a href={source.url} target="_blank" rel="noreferrer">{source.title} <ArrowSquareOut size={12} aria-label="(opens in a new tab)" /></a>
                        <span className="hint">{source.error}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="hint">These portals block automated downloads. Download the PDF by hand and add it with the form.</p>
                </details>
              )}
            </section>
          </div>

          <section className="panel" aria-labelledby="files-heading">
            <div className="table-head">
              <h2 id="files-heading">Documents</h2>
              <input type="search" className="table-search" placeholder="Filter by name or folder" value={search} onChange={(event) => setSearch(event.target.value)} aria-label="Filter documents" />
            </div>
            <div className="tabs" role="tablist" aria-label="Document status">
              {FILTERS.map((item) => (
                <button key={item.key} role="tab" type="button" aria-selected={filter === item.key} onClick={() => setFilter(item.key)}>
                  {item.label} <span className="count">{counts[item.key]}</span>
                </button>
              ))}
            </div>
            {files.length === 0 ? (
              <p className="empty-note">
                {filter === "waiting" ? "Nothing is waiting. Every PDF has been ingested." : "No documents match this view."}
              </p>
            ) : (
              <div className="table-scroll">
                <table className="data-table corpus-table">
                  <thead>
                    <tr>
                      <th>Document</th><th>Source</th><th className="num">Size</th><th>Status</th>
                      <th className="num">Passages</th><th>Last run</th><th><span className="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {files.map((file) => {
                      const state = stateOf(file);
                      return (
                        <tr key={file.path}>
                          <td>
                            <span className="doc-cell">
                              {file.ingestible ? <FilePdf size={18} aria-hidden="true" /> : <Globe size={18} aria-hidden="true" />}
                              <span>
                                <strong>{file.title ?? file.name}</strong>
                                <code>{file.path}</code>
                              </span>
                            </span>
                          </td>
                          <td>
                            {file.authority ? AUTHORITY[file.authority] ?? file.authority : "Not recorded"}
                            {file.source_url && (
                              <a className="cell-link" href={file.source_url} target="_blank" rel="noreferrer" aria-label={`Official source for ${file.name} (opens in a new tab)`}>
                                <ArrowSquareOut size={14} aria-hidden="true" />
                              </a>
                            )}
                          </td>
                          <td className="num">{bytes(file.size_bytes)}</td>
                          <td>
                            <span className={`badge badge-${state.tone}`} title={file.ingestion?.error ?? undefined}>{state.label}</span>
                            {file.empty && <span className="cell-error">The download produced no content. Fetch it manually and upload it.</span>}
                            {file.ingestion?.error && <span className="cell-error">{file.ingestion.error}</span>}
                          </td>
                          <td className="num">{file.ingestion ? file.ingestion.chunks_created : ""}</td>
                          <td>{when(file.ingestion?.ingested_at)}</td>
                          <td>
                            {file.ingestible && (
                              <div className="row-actions">
                                <button type="button" className="button button-small" disabled={running} onClick={() => ingest(file.path)}>
                                  {state.key === "searchable" ? "Re-ingest" : "Ingest"}
                                </button>
                                <button type="button" className="text-button" disabled={running} onClick={() => ingest(file.path, { dry_run: true })}>
                                  Test run
                                </button>
                              </div>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {corpus.history.length > 0 && (
            <section className="panel" aria-labelledby="history-heading">
              <h2 id="history-heading">Ingestion history</h2>
              <div className="table-scroll">
                <table className="data-table">
                  <thead><tr><th>When</th><th>File</th><th>Result</th><th className="num">Passages</th><th className="num">Uploaded</th></tr></thead>
                  <tbody>
                    {corpus.history.map((entry, position) => (
                      <tr key={`${entry.ingested_at}-${position}`}>
                        <td>{when(entry.ingested_at)}</td>
                        <td><code>{entry.path.split("/").slice(-2).join("/")}</code></td>
                        <td>
                          <span className={`badge badge-${entry.status === "ingested" ? "ok" : entry.status === "failed" ? "bad" : "info"}`}>
                            {entry.status === "dry_run" ? "Test run" : entry.status === "ingested" ? "Ingested" : "Failed"}
                          </span>
                          {entry.error && <span className="cell-error">{entry.error}</span>}
                        </td>
                        <td className="num">{entry.chunks_created}</td>
                        <td className="num">{entry.chunks_uploaded}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

export default function CorpusPage() {
  return (
    <Shell role="admin">
      <CorpusView />
    </Shell>
  );
}
