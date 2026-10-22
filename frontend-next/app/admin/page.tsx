"use client";

import { ArrowClockwise, CircleNotch, Warning } from "@phosphor-icons/react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Analytics, EvalDataset, fetchAnalytics, fetchEvalDatasets, fetchReview, ReviewStatus } from "../api";
import { BarList, Legend, LineChart, Series, shortDate, StackedColumns } from "../_components/charts";
import { Shell } from "../_components/Shell";
import { ErrorBanner, errorText, formatDateTime } from "../_components/ui";

const RANGES = [7, 14, 30];

const STRENGTH: Series[] = [
  { key: "high", label: "High", color: "var(--viz-ord-3)" },
  { key: "medium", label: "Medium", color: "var(--viz-ord-2)" },
  { key: "low", label: "Low (not answered)", color: "var(--viz-ord-1)" },
];

const LATENCY: Series[] = [
  { key: "p50_latency_ms", label: "Median", color: "var(--viz-two-2)" },
  { key: "p95_latency_ms", label: "Slowest 5%", color: "var(--viz-two-1)" },
];

const FEEDBACK_LABELS: Record<string, string> = {
  helpful: "Correct",
  wrong_source: "Wrong source",
  stale_status: "Outdated status",
  missing_authority: "Missing authority",
  poor_explanation: "Unclear explanation",
};

const ROUTING: Record<string, string> = {
  override: "Chosen by the workspace",
  keyword: "Keyword match",
  llm: "Model classifier",
  general: "Searched everything",
};

const seconds = (ms: number) => (ms >= 10_000 ? `${Math.round(ms / 1000)} s` : `${(ms / 1000).toFixed(1)} s`);
const axisSeconds = (ms: number) => `${Number((ms / 1000).toFixed(1))} s`;
const number = (value: number) => value.toLocaleString("en-IN");
const entries = (record: Record<string, number>, labels: Record<string, string> = {}) =>
  Object.entries(record).map(([label, value]) => ({ label: labels[label] ?? label, value }));

function Overview() {
  const [days, setDays] = useState(14);
  const [data, setData] = useState<Analytics | null>(null);
  const [review, setReview] = useState<Record<ReviewStatus, number> | null>(null);
  const [datasets, setDatasets] = useState<EvalDataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showAll, setShowAll] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [analytics, queue, evals] = await Promise.all([fetchAnalytics(days), fetchReview("open"), fetchEvalDatasets()]);
      setData(analytics);
      setReview(queue.counts);
      setDatasets(evals.datasets);
    } catch (requestError) {
      setError(errorText(requestError, "Analytics could not be loaded."));
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  const routes = useMemo(() => {
    if (!data) return [];
    const byRoute = new Map<string, { requests: number; errors: number; mean: number }>();
    for (const counter of data.http.counters.filter((c) => c.name === "http_requests_total")) {
      const row = byRoute.get(counter.labels.route) ?? { requests: 0, errors: 0, mean: 0 };
      row.requests += counter.value;
      if (Number(counter.labels.status) >= 400) row.errors += counter.value;
      byRoute.set(counter.labels.route, row);
    }
    for (const observation of data.http.observations.filter((o) => o.name === "http_latency_ms")) {
      const row = byRoute.get(observation.labels.route);
      if (row) row.mean = observation.mean;
    }
    return [...byRoute.entries()].sort((a, b) => b[1].requests - a[1].requests).slice(0, 12);
  }, [data]);

  const totals = data?.totals;
  const evalCases = datasets.reduce((sum, dataset) => sum + dataset.cases, 0);

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="display">Overview</h1>
          <p className="lede">Questions, answer quality and service health. Questions are logged as hashes; their text is never stored.</p>
        </div>
        <div className="head-actions">
          <div className="segmented" role="group" aria-label="Time range">
            {RANGES.map((range) => (
              <button key={range} type="button" aria-pressed={days === range} onClick={() => setDays(range)}>{range} days</button>
            ))}
          </div>
          <button type="button" className="btn" onClick={load} aria-label="Refresh">
            {loading ? <CircleNotch className="spin" size={16} aria-hidden="true" /> : <ArrowClockwise size={16} aria-hidden="true" />}
          </button>
        </div>
      </div>
      <ErrorBanner message={error} />

      {totals && data && (
        <>
          {totals.requests > 0 && totals.abstention_rate >= 0.8 && (
            <div className="banner banner-warn" role="note">
              <Warning size={18} aria-hidden="true" />
              <div>
                <strong>{Math.round(totals.abstention_rate * 100)}% of questions went unanswered for lack of evidence.</strong>
                <p>That usually means the vector index is empty or missing key statutes. <Link href="/admin/corpus">See what has been ingested</Link>.</p>
              </div>
            </div>
          )}

          <dl className="figures">
            <div><dt>Questions</dt><dd>{number(totals.requests)}</dd></div>
            <div><dt>Answered</dt><dd>{number(totals.answered)}</dd></div>
            <div className={totals.abstention_rate >= 0.5 ? "is-alert" : undefined}><dt>Not answered</dt><dd>{Math.round(totals.abstention_rate * 100)}%</dd></div>
            <div><dt>Median response</dt><dd>{seconds(totals.p50_latency_ms)}</dd></div>
            <div className={totals.p95_latency_ms > 8000 ? "is-alert" : undefined}><dt>Slowest 5%</dt><dd>{seconds(totals.p95_latency_ms)}</dd></div>
            <div><dt>Model tokens</dt><dd>{number(totals.tokens_used)}</dd></div>
            <div><dt>Feedback</dt><dd>{number(totals.feedback)}</dd></div>
          </dl>
          <p className="hint">Slowest 5% is flagged above 8 seconds, the pilot target.</p>

          <div className="cols-2">
            <section className="section" aria-labelledby="volume-heading">
              <h2 id="volume-heading">Questions per day, by retrieval strength</h2>
              <Legend series={STRENGTH} />
              <StackedColumns data={data.daily} series={STRENGTH} unit="questions" />
              <details className="table-toggle">
                <summary>Show as table</summary>
                <table className="table">
                  <thead><tr><th>Day</th><th className="num">High</th><th className="num">Medium</th><th className="num">Low</th><th className="num">Total</th></tr></thead>
                  <tbody>
                    {data.daily.map((row) => (
                      <tr key={row.date}><td>{shortDate(row.date)}</td><td className="num">{row.high}</td><td className="num">{row.medium}</td><td className="num">{row.low}</td><td className="num">{row.total}</td></tr>
                    ))}
                  </tbody>
                </table>
              </details>
            </section>
            <section className="section" aria-labelledby="latency-heading">
              <h2 id="latency-heading">Response time per day</h2>
              <Legend series={LATENCY} />
              <LineChart data={data.daily} series={LATENCY} format={seconds} axisFormat={axisSeconds} />
              <details className="table-toggle">
                <summary>Show as table</summary>
                <table className="table">
                  <thead><tr><th>Day</th><th className="num">Median</th><th className="num">Slowest 5%</th></tr></thead>
                  <tbody>
                    {data.daily.filter((row) => row.total > 0).map((row) => (
                      <tr key={row.date}><td>{shortDate(row.date)}</td><td className="num">{seconds(row.p50_latency_ms)}</td><td className="num">{seconds(row.p95_latency_ms)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </details>
            </section>
          </div>

          <div className="cols-3">
            <section className="section" aria-labelledby="quality-heading">
              <h2 id="quality-heading">Answer quality</h2>
              <dl className="figures">
                <div><dt>Open reports</dt><dd>{review?.open ?? 0}</dd></div>
                <div><dt>Eval cases</dt><dd>{evalCases}</dd></div>
              </dl>
              <p className="hint"><Link href="/admin/review">Review reported answers</Link> and turn confirmed problems into eval cases.</p>
            </section>
            <section className="section" aria-labelledby="feedback-heading">
              <h2 id="feedback-heading">Reader feedback</h2>
              <BarList items={entries(data.feedback.by_kind, FEEDBACK_LABELS)} empty="No feedback yet. Researchers can rate any answer." />
            </section>
            <section className="section" aria-labelledby="routing-heading">
              <h2 id="routing-heading">How questions were routed</h2>
              <BarList items={entries(data.router_stages, ROUTING)} empty="No questions in this period." />
            </section>
          </div>

          <section className="section" aria-labelledby="recent-heading">
            <div className="section-head"><h2 id="recent-heading">Latest questions</h2></div>
            {data.recent.length === 0 ? (
              <p className="hint">No questions in this period. Ask one from Research to see it here.</p>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th>When</th><th>Reference</th><th>Area</th><th>Result</th><th className="num">Top match</th><th className="num">Passages</th><th className="num">Citation warnings</th><th className="num">Time</th></tr>
                  </thead>
                  <tbody>
                    {data.recent.slice(0, showAll ? undefined : 10).map((row) => (
                      <tr key={`${row.timestamp}-${row.query_ref}`}>
                        <td className="num">{formatDateTime(row.timestamp)}</td>
                        <td><code>{row.query_ref}</code></td>
                        <td>{row.domain}</td>
                        <td>
                          <span className={`tag ${row.answered ? (row.confidence_tier === "HIGH" ? "tag-ok" : "tag-warn") : "tag-muted"}`}>
                            {row.answered ? `${row.confidence_tier.charAt(0)}${row.confidence_tier.slice(1).toLowerCase()} retrieval` : "Not answered"}
                          </span>
                        </td>
                        <td className="num">{row.top_chunk_score.toFixed(2)}</td>
                        <td className="num">{row.chunks_used}</td>
                        <td className="num">{row.citation_warnings}</td>
                        <td className="num">{seconds(row.latency_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {data.recent.length > 10 && (
                  <button type="button" className="btn-quiet" style={{ marginTop: 12 }} onClick={() => setShowAll(!showAll)}>
                    {showAll ? "Show fewer" : `Show all ${data.recent.length}`}
                  </button>
                )}
              </div>
            )}
          </section>

          <div className="cols-2">
            <section className="section" aria-labelledby="domains-heading">
              <h2 id="domains-heading">Areas of law asked about</h2>
              <BarList items={entries(data.domains)} empty="No questions in this period." />
            </section>
            <section className="section" aria-labelledby="traffic-heading">
              <h2 id="traffic-heading">API traffic since the last restart</h2>
              {routes.length === 0 ? (
                <p className="hint">No traffic recorded yet.</p>
              ) : (
                <div className="table-wrap">
                  <table className="table">
                    <thead><tr><th>Endpoint</th><th className="num">Requests</th><th className="num">Errors</th><th className="num">Mean time</th></tr></thead>
                    <tbody>
                      {routes.map(([route, row]) => (
                        <tr key={route}>
                          <td><code>{route}</code></td>
                          <td className="num">{number(row.requests)}</td>
                          <td className="num">{row.errors > 0 ? <span className="tag tag-bad">{number(row.errors)}</span> : 0}</td>
                          <td className="num">{row.mean < 1000 ? `${Math.round(row.mean)} ms` : seconds(row.mean)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </div>
        </>
      )}
    </>
  );
}

export default function AdminPage() {
  return (
    <Shell role="admin">
      <Overview />
    </Shell>
  );
}
