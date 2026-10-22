"use client";

import { ArrowRight, CircleNotch } from "@phosphor-icons/react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  CrosswalkResult,
  CrosswalkRow,
  fetchProvision,
  Provision,
  searchCrosswalk,
  verifyCrosswalkRow,
} from "../api";
import { useAuth } from "../auth";
import { Shell } from "../_components/Shell";
import { EmptyState, ErrorBanner, errorText, formatDate, RELATION_LABEL, StatusTag } from "../_components/ui";
import { WordDiff } from "../_components/WordDiff";

const FILTERS = [
  { key: "all", label: "All", params: {} },
  { key: "IPC", label: "IPC to BNS", params: { code: "IPC" } },
  { key: "CrPC", label: "CrPC to BNSS", params: { code: "CrPC" } },
  { key: "IEA", label: "Evidence Act to BSA", params: { code: "IEA" } },
  { key: "new", label: "New", params: { relation: "new" } },
  { key: "omitted", label: "Not carried over", params: { relation: "omitted" } },
] as const;

type Side = { provision: Provision | null; missing: string };

const ref = (code: string | null, section: string | null) => (code && section ? `${code} ${section}` : "None");

async function loadSide(code: string | null, section: string | null): Promise<Side> {
  if (!code || !section) return { provision: null, missing: "" };
  try {
    return { provision: await fetchProvision(code, section), missing: "" };
  } catch (error) {
    const empty = code === "IPC" || code === "CrPC";
    const detail = error instanceof ApiError && error.status === 404 ? error.message : errorText(error, "Text unavailable.");
    return { provision: null, missing: empty ? `${detail} The ${code} PDF in the corpus is empty, so its text cannot be shown yet.` : detail };
  }
}

function ProvisionText({ label, side }: { label: string; side: Side }) {
  if (!side.provision && !side.missing) return null;
  return (
    <div className="provision">
      <h3>
        <span>{label}</span>
        {side.provision && <span className="muted">{side.provision.extracted_from}, {side.provision.page_anchor}</span>}
      </h3>
      {side.provision ? (
        <div className="law-text"><p>{side.provision.text}</p></div>
      ) : (
        <p className="hint">{side.missing}</p>
      )}
    </div>
  );
}

function Detail({ row, onChange }: { row: CrosswalkRow; onChange: (row: CrosswalkRow) => void }) {
  const { session } = useAuth();
  const [before, setBefore] = useState<Side>({ provision: null, missing: "" });
  const [after, setAfter] = useState<Side>({ provision: null, missing: "" });
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<"side" | "diff">("side");
  const [error, setError] = useState("");
  const relation = RELATION_LABEL[row.relation];

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setView("side");
    Promise.all([loadSide(row.from_code, row.from_section), loadSide(row.to_code, row.to_section)]).then(([from, to]) => {
      if (cancelled) return;
      setBefore(from);
      setAfter(to);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [row.id, row.from_code, row.from_section, row.to_code, row.to_section]);

  async function toggleVerified() {
    setError("");
    try {
      onChange(await verifyCrosswalkRow(row.id, !row.verified));
    } catch (requestError) {
      setError(errorText(requestError, "The row was not updated."));
    }
  }

  const askAbout = row.to_code
    ? `What does ${row.to_code} section ${row.to_section} provide${row.from_code ? `, and how does it differ from ${row.from_code} section ${row.from_section}` : ""}?`
    : `What happened to ${row.from_code} section ${row.from_section} under the new criminal laws?`;

  return (
    <aside className="xw-detail" aria-labelledby="xw-detail-heading" id="xw-detail">
      <h2 id="xw-detail-heading" className="xw-pair">
        <span>{ref(row.from_code, row.from_section)}</span>
        <ArrowRight size={20} aria-hidden="true" />
        <span className="sr-only">replaced by</span>
        <span className="to">{ref(row.to_code, row.to_section)}</span>
      </h2>
      <p className="xw-note">
        <span className={`tag tag-${relation.tone}`}>{relation.label}</span>
        {row.note && <> {row.note}</>}
      </p>
      <div className="xw-verify">
        {row.verified ? (
          <span className="tag tag-ok">Checked by {row.verified_by} on {formatDate(row.verified_at)}</span>
        ) : (
          <span className="tag tag-warn">Not yet checked against the official tables</span>
        )}
        {session?.role === "admin" && (
          <button type="button" className="btn btn-small" onClick={toggleVerified}>
            {row.verified ? "Mark as unchecked" : "Mark as checked"}
          </button>
        )}
      </div>
      <ErrorBanner message={error} />

      {loading ? (
        <p className="hint" role="status"><CircleNotch className="spin" size={14} aria-hidden="true" /> Loading section text</p>
      ) : (
        <>
          {before.provision && after.provision && (
            <div className="tabs" role="tablist" aria-label="Text view" style={{ marginTop: 20 }}>
              <button role="tab" type="button" aria-selected={view === "side"} onClick={() => setView("side")}>Both texts</button>
              <button role="tab" type="button" aria-selected={view === "diff"} onClick={() => setView("diff")}>What changed</button>
            </div>
          )}
          {view === "diff" && before.provision && after.provision ? (
            <>
              <p className="hint">Words only in the new text are highlighted; removed words are struck through.</p>
              <WordDiff before={before.provision.text} after={after.provision.text} />
            </>
          ) : (
            <>
              <ProvisionText label={`${ref(row.from_code, row.from_section)}, old law`} side={before} />
              <ProvisionText label={`${ref(row.to_code, row.to_section)}, new law`} side={after} />
            </>
          )}
          {(before.provision?.extracted_from === "Gazette of India PDF" || after.provision?.extracted_from === "Gazette of India PDF") && (
            <p className="hint">Gazette PDFs interleave marginal notes with the text, so a few words may appear out of place.</p>
          )}
        </>
      )}

      <div className="links-row">
        {row.to_code && <Link href={`/timeline?code=${row.to_code}&section=${encodeURIComponent(row.to_section ?? "")}`}>Timeline of {ref(row.to_code, row.to_section)}</Link>}
        {row.from_code && <Link href={`/timeline?code=${row.from_code}&section=${encodeURIComponent(row.from_section ?? "")}`}>Timeline of {ref(row.from_code, row.from_section)}</Link>}
        <Link href={`/?q=${encodeURIComponent(askAbout)}`}>Ask about this</Link>
      </div>
    </aside>
  );
}

function CrosswalkView() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["key"]>("all");
  const [data, setData] = useState<CrosswalkResult | null>(null);
  const [selected, setSelected] = useState<CrosswalkRow | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const detailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const initial = new URLSearchParams(window.location.search).get("q");
    if (initial) setQuery(initial);
  }, []);

  useEffect(() => {
    const params = FILTERS.find((item) => item.key === filter)?.params ?? {};
    const timer = window.setTimeout(async () => {
      setLoading(true);
      try {
        const result = await searchCrosswalk({ q: query, ...params });
        setData(result);
        setError("");
        setSelected((current) => result.rows.find((row) => row.id === current?.id) ?? result.rows[0] ?? null);
      } catch (requestError) {
        setError(errorText(requestError, "The crosswalk could not be loaded."));
      } finally {
        setLoading(false);
      }
    }, 220);
    return () => window.clearTimeout(timer);
  }, [query, filter]);

  function choose(row: CrosswalkRow) {
    setSelected(row);
    if (window.matchMedia("(max-width: 1100px)").matches) {
      requestAnimationFrame(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }

  function replaceRow(updated: CrosswalkRow) {
    setSelected(updated);
    setData((current) => current && {
      ...current,
      verified: current.verified + (updated.verified ? 1 : -1),
      rows: current.rows.map((row) => (row.id === updated.id ? updated : row)),
    });
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="display">Crosswalk</h1>
          <p className="lede">Find where an IPC, CrPC or Evidence Act section went in the new laws, and read both texts.</p>
        </div>
        {data && <p className="muted num">{data.verified} of {data.total} correspondences checked</p>}
      </div>

      <div className="xw-controls">
        <div className="field">
          <label htmlFor="xw-search">Section or subject</label>
          <input id="xw-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="IPC 302, 65B, bail, dowry death" />
        </div>
        <div className="tabs" role="tablist" aria-label="Filter correspondences">
          {FILTERS.map((item) => (
            <button key={item.key} role="tab" type="button" aria-selected={filter === item.key} onClick={() => setFilter(item.key)}>
              {item.label}
            </button>
          ))}
        </div>
      </div>
      {data && <p className="hint" style={{ marginBottom: 20 }}>{data.note}</p>}
      <ErrorBanner message={error} />
      {loading && !data && <div className="loadbar" />}

      {data && (
        <div className="xw-layout">
          <div>
            {data.rows.length === 0 ? (
              <EmptyState title="No correspondences match">
                Try a bare section number such as 302, or a word from the heading such as bail.
              </EmptyState>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Old law</th><th>New law</th><th>Change</th><th>Checked</th></tr></thead>
                  <tbody>
                    {data.rows.map((row) => (
                      <tr key={row.id} className={selected?.id === row.id ? "is-selected" : undefined}>
                        <td>
                          <button type="button" className="row-button" onClick={() => choose(row)} aria-pressed={selected?.id === row.id}>
                            <strong>{ref(row.from_code, row.from_section)}</strong>
                            {row.from_heading && <span className="cell-sub">{row.from_heading}</span>}
                          </button>
                        </td>
                        <td>
                          <strong>{ref(row.to_code, row.to_section)}</strong>
                          {row.to_heading && <span className="cell-sub">{row.to_heading}</span>}
                        </td>
                        <td><span className={`tag tag-${RELATION_LABEL[row.relation].tone}`}>{RELATION_LABEL[row.relation].label}</span></td>
                        <td>{row.verified ? <StatusTag status="in_force" label="Checked" /> : <span className="tag tag-muted">Not yet</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          <div ref={detailRef}>{selected && <Detail key={selected.id} row={selected} onChange={replaceRow} />}</div>
        </div>
      )}
    </>
  );
}

export default function CrosswalkPage() {
  return (
    <Shell>
      <CrosswalkView />
    </Shell>
  );
}
