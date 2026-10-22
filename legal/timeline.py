"""Dated history of a single provision: enactment, commencement, exceptions, repeal, replacement."""
from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from legal.acts import Act, get_act, provision_status
from legal.crosswalk import CrosswalkStore, base_section
from legal.documents import LegalDocument


class CatalogReader(Protocol):
    def get(self, document_id: str) -> LegalDocument: ...

STATUS_LABELS = {
    "in_force": "In force",
    "partly_in_force": "Partly in force",
    "not_in_force": "Not in force",
    "not_yet_in_force": "Not yet in force",
    "repealed": "Repealed",
}


def _event(day: date, kind: str, title: str, *, detail: str = "", act: Act | None = None,
           section: str | None = None, instrument: str | None = None, url: str | None = None) -> dict[str, Any]:
    return {
        "date": day.isoformat(),
        "kind": kind,
        "title": title,
        "detail": detail,
        "code": act.code if act else None,
        "section": section,
        "instrument": instrument,
        "url": url,
    }


def _act_events(act: Act, section: str) -> list[dict[str, Any]]:
    label = f"{act.code} section {section}"
    events = [
        _event(act.assented_on, "enacted", f"{act.title} enacted",
               detail=f"{act.act_number} received assent.", act=act, section=section),
    ]
    if act.commencement:
        events.append(_event(
            act.commencement.issued_on, "notified", "Commencement notified",
            detail=f"Brings the {act.code} into force from {act.in_force_from:%d %B %Y}.",
            act=act, section=section, instrument=act.commencement.label, url=act.commencement.url,
        ))
        excluded = [item for item in act.not_in_force if base_section(item) == base_section(section)]
        for item in excluded:
            events.append(_event(
                act.commencement.issued_on, "exception", f"Section {item} excluded from commencement",
                detail="This part has not been brought into force.",
                act=act, section=item, instrument=act.commencement.label, url=act.commencement.url,
            ))
    events.append(_event(act.in_force_from, "in_force", f"{label} in force", act=act, section=section,
                         instrument=act.commencement.label if act.commencement else None))
    if act.repealed_on:
        events.append(_event(
            act.repealed_on, "repealed", f"{label} repealed",
            detail=f"Repealed by {act.repealed_by}, with savings for things done before repeal.",
            act=act, section=section, instrument=act.repealed_by,
        ))
    return events


def build_timeline(
    code: str,
    section: str,
    *,
    crosswalk: CrosswalkStore,
    catalog: CatalogReader | None = None,
    on: date | None = None,
) -> dict[str, Any]:
    act = get_act(code)
    section = section.strip().upper().replace(" ", "")
    status = provision_status(act.code, section, on=on)
    related = crosswalk.for_section(act.code, section)

    events = _act_events(act, section)
    for row in related:
        if row.relation == "omitted" and row.from_code == act.code:
            events.append(_event(act.repealed_on or date.today(), "omitted",
                                 "No corresponding provision in the new law", detail=row.note, act=act, section=section))
            continue
        other_code, other_section = (
            (row.to_code, row.to_section) if row.from_code == act.code else (row.from_code, row.from_section)
        )
        if not other_code or not other_section:
            continue
        other = get_act(other_code)
        if row.from_code == act.code:
            events.append(_event(other.in_force_from, "replaced",
                                 f"Replaced by {other.code} section {other_section}", detail=row.note,
                                 act=other, section=other_section, instrument=other.commencement.label if other.commencement else None))
        else:
            events.append(_event(other.in_force_from, "predecessor",
                                 f"Predecessor {other.code} section {other_section} in force", act=other, section=other_section))
            if other.repealed_on:
                events.append(_event(other.repealed_on, "predecessor_repealed",
                                     f"{other.code} section {other_section} repealed", detail=f"Repealed by {other.repealed_by}.",
                                     act=other, section=other_section, instrument=other.repealed_by))

    order = {"predecessor": 0, "enacted": 1, "notified": 2, "exception": 3, "predecessor_repealed": 4,
             "repealed": 5, "omitted": 6, "in_force": 7, "replaced": 8}
    events.sort(key=lambda event: (event["date"], order.get(event["kind"], 9)))

    provision = None
    if catalog is not None:
        try:
            document = catalog.get(act.document_id)
            provision = next(
                (p for p in document.provisions if p.section_number.upper() == base_section(section)), None
            )
        except KeyError:
            provision = None

    heading = next(
        (getattr(row, f"{side}_heading") for row in related for side in ("from", "to")
         if (getattr(row, f"{side}_code") or "").lower() == act.code.lower()
         and base_section(getattr(row, f"{side}_section")) == base_section(section)
         and getattr(row, f"{side}_heading")),
        provision.heading if provision and provision.heading else None,
    )
    return {
        "code": act.code,
        "section": section,
        "act_title": act.title,
        "heading": heading,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "source_url": act.source_url,
        "text": provision.text if provision else None,
        "page_anchor": provision.page_anchor if provision else None,
        "events": events,
        "related": [row.to_dict() for row in related],
    }
