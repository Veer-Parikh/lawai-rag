# LawAI design system: "Marked-up Bare Act"

Source of truth for every LawAI screen. Page files in `pages/` override this
file where they disagree (see `.github/skills/ui-ux-pro-max/SKILL.md`, step 2b).

## Where this came from

- **ui-ux-pro-max product match:** "Legal Services" and "Research Lab". The
  primary style recommendation for both is **Swiss Modernism 2.0 + Minimalism
  & Swiss Style**, with Accessible & Ethical as a secondary style.
- **Rejected:** the generator's default legal profile (navy + gold + EB
  Garamond + Lato). It is the stock "law firm" look and read as templated in
  review on 2026-10-21.
- **Subject vernacular:** lawyers read statutes as *bare acts* and mark them up
  with a highlighter. That's the one bold element here: highlighter yellow is
  the only accent, used where a lawyer would use a highlighter.

## Principles

1. **Grid carries hierarchy, not boxes.** A 12-column grid, black rules and
   whitespace separate content. No cards, no drop shadows, no gradients.
2. **One accent with one job.** Highlighter yellow marks what matters: cited
   authorities, the active nav item, the current selection. It is never used
   for text and never used for status.
3. **Serif means the law.** Source Serif 4 is reserved for statute text and
   answers drawn from it. The interface around it is always sans.
4. **Status is never colour alone.** Every status has a label. Green, amber and
   red are reserved for in force / superseded / repealed-or-error.
5. **Say less.** Headings have no explanatory subtitle unless it changes what
   the reader does. Labels are sentence case, never all caps.

## Color tokens

| Token | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `#FFFFFF` | `#141414` | Page background |
| `--ground` | `#F4F4F4` | `#1C1C1C` | Input wells, table hover, code |
| `--ink` | `#000000` | `#F2F2F2` | Text, primary buttons, strong rules |
| `--ink-2` | `#3D3D3D` | `#C8C8C8` | Secondary text (11:1 / 10:1) |
| `--muted` | `#666666` | `#9A9A9A` | Tertiary text, axis labels (5.7:1 / 6.3:1) |
| `--rule` | `#E0E0E0` | `#2E2E2E` | Hairlines between rows |
| `--control` | `#8A8A8A` | `#6E6E6E` | Input borders (3.4:1 non-text contrast) |
| `--mark` | `#FFE14D` | `#FFE14D` | Highlighter. Text on it is always `#000` (16:1) |
| `--mark-soft` | `#FFF2A8` | `#4D4300` | Hover state of marked elements |
| `--ok` | `#0B7A3B` | `#4CC27A` | In force, success (5.4:1) |
| `--warn` | `#9A5B00` | `#F0B44C` | Superseded, needs review (5.3:1) |
| `--bad` | `#C0261A` | `#FF7A6B` | Repealed, failed, error (5.6:1) |

### Chart ramps (validated with the dataviz palette validator)

| Ramp | Light | Dark |
|---|---|---|
| Ordinal, 3 steps (low → high) | `#A6A6A6` `#5E5E5E` `#111111` | `#4A4A4A` `#8C8C8C` `#E6E6E6` |
| Ordinal, 2 steps (p95 → median) | `#8F8F8F` `#111111` | `#5A5A5A` `#E6E6E6` |

Charts are monochrome. The highlighter may mark one annotated point, never a series.

## Typography

| Role | Family | Settings |
|---|---|---|
| Interface and headings | **Schibsted Grotesk** 400/500/700/800 | Headings 800, tracking −0.03em |
| Law and answers | **Source Serif 4** 400/600, optical sizing | 18px/1.65, max 68ch |
| Identifiers (hashes, paths, IDs) | **JetBrains Mono** 400/500 | 12.5px |

Scale (px): 12 · 13 · 15 (UI base) · 18 (serif body) · 22 · 28 · 40 · clamp(40–64) display.
Numbers in tables and figures use `font-variant-numeric: tabular-nums`.

## Layout

- **Shell:** a top bar (56px, 2px black bottom rule) instead of a sidebar.
  Primary nav: Research, Crosswalk, Timeline, Changes, Matters. An admin
  sub-nav row appears inside `/admin`.
- **Grid:** 12 columns, max width 1320px, 24px gutters, side margins
  `clamp(16px, 4vw, 48px)`. Page titles sit in columns 1–8, left aligned.
- **Spacing:** 8px base (4, 8, 12, 16, 24, 32, 48, 64).
- **Radius:** 2px on controls, 0 on sections. Shadows only on floating layers
  (menus, tooltips).

## Components

- **Buttons:** primary = black fill, white text; secondary = 1px black
  outline; tertiary = underlined text. 40px tall (44px under 720px).
- **Inputs:** 1px `--control` border, 2px black on focus, label above.
- **Marks:** `.mark` puts a highlighter stroke behind inline text. Use for
  cited sections, active nav and the selected row.
- **Status tag:** 8px square swatch in the status colour + label in ink.
- **Tables:** bold 13px header with a 1px black rule below; hairline rows;
  numeric columns right-aligned; wide tables scroll horizontally.
- **Focus:** 2px black outline, 2px offset; on black controls a white inner
  ring plus black outer ring.

## Motion

Colour and opacity transitions at 120ms. No entrance animations. Loading is a
thin indeterminate bar, not a skeleton shimmer. `prefers-reduced-motion`
disables everything.

## Avoid

- Cards with shadows, gradient washes, glassmorphism.
- Serif headlines, navy + gold, cream backgrounds.
- All-caps eyebrow labels, `A · B · C` meta strings, arrows appended to buttons.
- Pastel pill badges as the only status signal.
- Emoji as icons. Icons are Phosphor, regular weight, 18px, `aria-hidden` when
  next to text.

## Pre-delivery checklist (ui-ux-pro-max)

- [ ] Text contrast ≥ 4.5:1 in light and dark; control borders ≥ 3:1
- [ ] Visible keyboard focus everywhere; skip link present
- [ ] Works at 375px, 768px, 1024px, 1440px with no horizontal page scroll
- [ ] `prefers-reduced-motion` respected
- [ ] Every empty state offers the next action
- [ ] Destructive actions ask for confirmation; successful actions confirm
