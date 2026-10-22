"use client";

import { ReactNode, useLayoutEffect, useRef, useState } from "react";

// Shared chart primitives. Colors come from CSS custom properties
// (--viz-*), validated light and dark with the dataviz palette validator.

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    // Measure before first paint; the observer then tracks later resizes.
    setWidth(Math.floor(element.getBoundingClientRect().width));
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

function niceMax(value: number) {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((candidate) => candidate * magnitude >= value) ?? 10;
  return step * magnitude;
}

export function shortDate(iso: string) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

export type Series = { key: string; label: string; color: string };

export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="legend" aria-label="Legend">
      {series.map((item) => (
        <li key={item.key}>
          <span className="legend-swatch" style={{ background: item.color }} aria-hidden="true" />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

function Tooltip({ x, chartWidth, children }: { x: number; chartWidth: number; children: ReactNode }) {
  const flip = x > chartWidth * 0.62;
  return (
    <div className="viz-tooltip" style={flip ? { right: chartWidth - x + 12 } : { left: x + 12 }} role="status">
      {children}
    </div>
  );
}

const PAD = { top: 12, right: 12, bottom: 28, left: 40 };

type Row = { date: string } & Record<string, number | string>;

/** Stacked columns per day. `series` order is bottom to top. */
export function StackedColumns({ data, series, height = 220, unit }: { data: Row[]; series: Series[]; height?: number; unit: string }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const max = niceMax(Math.max(0, ...data.map((row) => series.reduce((sum, s) => sum + Number(row[s.key] ?? 0), 0))));
  const innerW = Math.max(0, width - PAD.left - PAD.right);
  const innerH = height - PAD.top - PAD.bottom;
  const slot = data.length ? innerW / data.length : 0;
  const barW = Math.max(4, Math.min(28, slot * 0.6));
  const y = (value: number) => PAD.top + innerH - (value / max) * innerH;
  const ticks = [0, max / 2, max];
  const labelEvery = Math.ceil(data.length / Math.max(1, Math.floor(innerW / 64)));

  return (
    <div className="viz" ref={ref} onMouseLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={`Daily ${unit} by confidence`}>
          {ticks.map((tick) => (
            <g key={tick}>
              <line className="viz-grid" x1={PAD.left} x2={width - PAD.right} y1={y(tick)} y2={y(tick)} />
              <text className="viz-axis" x={PAD.left - 8} y={y(tick) + 4} textAnchor="end">{tick}</text>
            </g>
          ))}
          {data.map((row, index) => {
            const cx = PAD.left + slot * index + slot / 2;
            let base = 0;
            const segments = series.map((s) => ({ ...s, value: Number(row[s.key] ?? 0) })).filter((s) => s.value > 0);
            return (
              <g key={row.date} opacity={hover === null || hover === index ? 1 : 0.45}>
                {segments.map((segment, segmentIndex) => {
                  const top = y(base + segment.value);
                  const bottom = y(base);
                  base += segment.value;
                  const isTop = segmentIndex === segments.length - 1;
                  // 2px surface gap between stacked segments; 4px rounded end on the top segment only.
                  const h = Math.max(1, bottom - top - (segmentIndex > 0 ? 2 : 0));
                  const r = isTop ? Math.min(4, h / 2, barW / 2) : 0;
                  const x0 = cx - barW / 2;
                  const path = `M${x0},${top + h} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x0 + barW - r} Q${x0 + barW},${top} ${x0 + barW},${top + r} V${top + h} Z`;
                  return <path key={segment.key} d={path} fill={segment.color} />;
                })}
                {index % labelEvery === 0 && (
                  <text className="viz-axis" x={cx} y={height - 8} textAnchor="middle">{shortDate(row.date)}</text>
                )}
                <rect
                  x={cx - slot / 2}
                  y={PAD.top}
                  width={slot}
                  height={innerH}
                  fill="transparent"
                  onMouseEnter={() => setHover(index)}
                />
              </g>
            );
          })}
          <line className="viz-baseline" x1={PAD.left} x2={width - PAD.right} y1={y(0)} y2={y(0)} />
        </svg>
      )}
      {hover !== null && data[hover] && (
        <Tooltip x={PAD.left + slot * hover + slot / 2} chartWidth={width}>
          <strong>{shortDate(data[hover].date)}</strong>
          {[...series].reverse().map((s) => (
            <span key={s.key} className="viz-tooltip-row">
              <span className="legend-swatch" style={{ background: s.color }} aria-hidden="true" />
              {s.label}<b>{Number(data[hover][s.key] ?? 0)}</b>
            </span>
          ))}
        </Tooltip>
      )}
    </div>
  );
}

/** Multi-series line chart with a crosshair tooltip. Values share one unit and one axis. */
export function LineChart({ data, series, height = 220, format, axisFormat = format }: { data: Row[]; series: Series[]; height?: number; format: (value: number) => string; axisFormat?: (value: number) => string }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const max = niceMax(Math.max(0, ...data.flatMap((row) => series.map((s) => Number(row[s.key] ?? 0)))));
  const innerW = Math.max(0, width - PAD.left - PAD.right - 84);
  const innerH = height - PAD.top - PAD.bottom;
  const x = (index: number) => PAD.left + (data.length > 1 ? (innerW * index) / (data.length - 1) : innerW / 2);
  const y = (value: number) => PAD.top + innerH - (value / max) * innerH;
  const ticks = [0, max / 2, max];
  const labelEvery = Math.ceil(data.length / Math.max(1, Math.floor(innerW / 64)));
  const active = data.map((row) => Number(row.total ?? 1) > 0);

  return (
    <div
      className="viz"
      ref={ref}
      onMouseLeave={() => setHover(null)}
      onMouseMove={(event) => {
        const box = event.currentTarget.getBoundingClientRect();
        const relative = event.clientX - box.left - PAD.left;
        const index = Math.round((relative / Math.max(1, innerW)) * (data.length - 1));
        setHover(Math.min(data.length - 1, Math.max(0, index)));
      }}
    >
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Daily latency">
          {ticks.map((tick) => (
            <g key={tick}>
              <line className="viz-grid" x1={PAD.left} x2={PAD.left + innerW} y1={y(tick)} y2={y(tick)} />
              <text className="viz-axis" x={PAD.left - 8} y={y(tick) + 4} textAnchor="end">{axisFormat(tick)}</text>
            </g>
          ))}
          {data.map((row, index) => index % labelEvery === 0 && (
            <text key={row.date} className="viz-axis" x={x(index)} y={height - 8} textAnchor="middle">{shortDate(row.date)}</text>
          ))}
          {series.map((s) => {
            // Break the line on days without traffic instead of drawing a false zero.
            const segments: string[] = [];
            let current = "";
            data.forEach((row, index) => {
              if (!active[index]) {
                if (current) segments.push(current);
                current = "";
                return;
              }
              current += `${current ? "L" : "M"}${x(index)},${y(Number(row[s.key] ?? 0))}`;
            });
            if (current) segments.push(current);
            return (
              <g key={s.key}>
                {segments.map((d) => <path key={d} d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />)}
                {data.map((row, index) => active[index] && (
                  <circle key={row.date} cx={x(index)} cy={y(Number(row[s.key] ?? 0))} r={hover === index ? 4.5 : 3} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                ))}
              </g>
            );
          })}
          {directLabels(data, series, active, x, y)}
          {hover !== null && <line className="viz-crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + innerH} />}
          <line className="viz-baseline" x1={PAD.left} x2={PAD.left + innerW} y1={y(0)} y2={y(0)} />
        </svg>
      )}
      {hover !== null && data[hover] && (
        <Tooltip x={x(hover)} chartWidth={width}>
          <strong>{shortDate(data[hover].date)}</strong>
          {active[hover] ? series.map((s) => (
            <span key={s.key} className="viz-tooltip-row">
              <span className="legend-swatch" style={{ background: s.color }} aria-hidden="true" />
              {s.label}<b>{format(Number(data[hover][s.key] ?? 0))}</b>
            </span>
          )) : <span className="viz-tooltip-row">No questions this day</span>}
        </Tooltip>
      )}
    </div>
  );
}

// End-of-line labels, nudged apart so close values never overlap.
function directLabels(
  data: Row[],
  series: Series[],
  active: boolean[],
  x: (index: number) => number,
  y: (value: number) => number,
) {
  const lastIndex = active.lastIndexOf(true);
  if (lastIndex < 0) return null;
  const placed = series
    .map((s) => ({ s, at: y(Number(data[lastIndex][s.key] ?? 0)) + 4 }))
    .sort((a, b) => a.at - b.at);
  for (let i = 1; i < placed.length; i++) {
    placed[i].at = Math.max(placed[i].at, placed[i - 1].at + 14);
  }
  return placed.map(({ s, at }) => (
    <text key={s.key} className="viz-direct-label" x={x(lastIndex) + 10} y={at}>{s.label}</text>
  ));
}

/** Horizontal magnitude bars for a handful of nominal categories (one color). */
export function BarList({ items, empty }: { items: { label: string; value: number }[]; empty: string }) {
  const max = Math.max(1, ...items.map((item) => item.value));
  if (items.length === 0) return <p className="hint">{empty}</p>;
  return (
    <ul className="bar-list">
      {items.map((item) => (
        <li key={item.label} title={`${item.label}: ${item.value}`}>
          <span className="bar-label">{item.label}</span>
          <span className="bar-track" aria-hidden="true">
            <span className="bar-fill" style={{ width: `${(item.value / max) * 100}%` }} />
          </span>
          <span className="bar-value">{item.value}</span>
        </li>
      ))}
    </ul>
  );
}
