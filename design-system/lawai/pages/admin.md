# Admin pages: override of MASTER.md

Applies to `/admin`, `/admin/corpus`, `/admin/sources` and `/admin/review`.

ui-ux-pro-max match: **Data-Dense Dashboard** (BI/Analytics), applied on top
of the Swiss master rules.

## Differences from master

- **Density:** 8px grid gaps, 12px cell padding, 13px table text, 36px rows.
- **Figures row:** one ruled strip of figures (label above, number below,
  28px Schibsted 700, tabular). No tiles or boxes.
- **Panels:** sections are separated by a 2px black top rule and a 15px bold
  heading. Still no cards.
- **Tables:** sticky header inside scroll containers; row hover uses `--ground`.
- **Charts:** monochrome ordinal ramps from MASTER.md; legend always present
  for two or more series; "Show as table" fallback under every chart.

## Keeps from master

Top bar, highlighter for the active sub-nav item and selected rows, status
tags, focus treatment, motion rules.
