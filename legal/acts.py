"""Registry of the Acts in scope, with their enactment, commencement and repeal facts.

These facts drive provision timelines and status. Each carries the instrument
that proves it so the UI can cite it. Keep this file in sync with the Gazette:
it is reference data, not something the model infers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Instrument:
    label: str           # "Notification S.O. 850(E)"
    issued_on: date
    publisher: str
    url: str | None = None


@dataclass(frozen=True)
class Act:
    code: str                 # "BNS"
    title: str                # "Bharatiya Nyaya Sanhita, 2023"
    act_number: str           # "Act No. 45 of 2023"
    document_id: str          # Catalog ID, e.g. "bns-2023"
    pdf_path: str             # Relative to DATA_DIR
    assented_on: date
    in_force_from: date
    commencement: Instrument | None
    source_url: str
    publisher: str
    successor: str | None = None      # Code of the Act that replaced it
    predecessor: str | None = None
    repealed_on: date | None = None
    repealed_by: str | None = None    # "BNS, section 358"
    not_in_force: tuple[str, ...] = field(default_factory=tuple)  # Provisions excluded at commencement

    @property
    def status(self) -> str:
        return "repealed" if self.repealed_on else "in_force"


_GAZETTE = "Gazette of India"
_LEGISLATIVE = "Legislative Department, Ministry of Law and Justice"
_NEW_LAWS_COMMENCED = date(2024, 7, 1)

ACTS: dict[str, Act] = {
    act.code: act
    for act in (
        Act(
            code="BNS",
            title="Bharatiya Nyaya Sanhita, 2023",
            act_number="Act No. 45 of 2023",
            document_id="bns-2023",
            pdf_path="primary_statutes/current/bns/bns_2023.pdf",
            assented_on=date(2023, 12, 25),
            in_force_from=_NEW_LAWS_COMMENCED,
            commencement=Instrument("Notification S.O. 850(E)", date(2024, 2, 23), _GAZETTE, "https://egazette.gov.in/"),
            source_url="https://www.indiacode.nic.in/",
            publisher=_LEGISLATIVE,
            predecessor="IPC",
            not_in_force=("106(2)",),
        ),
        Act(
            code="BNSS",
            title="Bharatiya Nagarik Suraksha Sanhita, 2023",
            act_number="Act No. 46 of 2023",
            document_id="bnss-2023",
            pdf_path="primary_statutes/current/bnss/bnss_2023.pdf",
            assented_on=date(2023, 12, 25),
            in_force_from=_NEW_LAWS_COMMENCED,
            commencement=Instrument("Notification S.O. 848(E)", date(2024, 2, 23), _GAZETTE, "https://egazette.gov.in/"),
            source_url="https://www.indiacode.nic.in/",
            publisher=_LEGISLATIVE,
            predecessor="CrPC",
        ),
        Act(
            code="BSA",
            title="Bharatiya Sakshya Adhiniyam, 2023",
            act_number="Act No. 47 of 2023",
            document_id="bsa-2023",
            pdf_path="primary_statutes/current/bsa/bsa_2023.pdf",
            assented_on=date(2023, 12, 25),
            in_force_from=_NEW_LAWS_COMMENCED,
            commencement=Instrument("Notification S.O. 849(E)", date(2024, 2, 23), _GAZETTE, "https://egazette.gov.in/"),
            source_url="https://www.indiacode.nic.in/",
            publisher=_LEGISLATIVE,
            predecessor="IEA",
        ),
        Act(
            code="IPC",
            title="Indian Penal Code, 1860",
            act_number="Act No. 45 of 1860",
            document_id="ipc-1860",
            pdf_path="primary_statutes/historical/ipc/ipc_1860.pdf",
            assented_on=date(1860, 10, 6),
            in_force_from=date(1862, 1, 1),
            commencement=None,
            source_url="https://www.indiacode.nic.in/handle/123456789/2263",
            publisher=_LEGISLATIVE,
            successor="BNS",
            repealed_on=_NEW_LAWS_COMMENCED,
            repealed_by="BNS, section 358",
        ),
        Act(
            code="CrPC",
            title="Code of Criminal Procedure, 1973",
            act_number="Act No. 2 of 1974",
            document_id="crpc-1973",
            pdf_path="primary_statutes/historical/crpc/crpc_1973.pdf",
            assented_on=date(1974, 1, 25),
            in_force_from=date(1974, 4, 1),
            commencement=None,
            source_url="https://www.indiacode.nic.in/handle/123456789/15098",
            publisher=_LEGISLATIVE,
            successor="BNSS",
            repealed_on=_NEW_LAWS_COMMENCED,
            repealed_by="BNSS, section 531",
        ),
        Act(
            code="IEA",
            title="Indian Evidence Act, 1872",
            act_number="Act No. 1 of 1872",
            document_id="iea-1872",
            pdf_path="primary_statutes/historical/iea/indian_evidence_act_1872.pdf",
            assented_on=date(1872, 3, 15),
            in_force_from=date(1872, 9, 1),
            commencement=None,
            source_url="https://www.indiacode.nic.in/",
            publisher=_LEGISLATIVE,
            successor="BSA",
            repealed_on=_NEW_LAWS_COMMENCED,
            repealed_by="BSA, section 170",
        ),
    )
}


def get_act(code: str) -> Act:
    for key, act in ACTS.items():
        if key.lower() == code.lower():
            return act
    raise KeyError(f"unknown act code: {code}")


def provision_status(code: str, section: str, *, on: date | None = None) -> str:
    """'in_force', 'not_in_force', 'repealed' or 'not_yet_in_force' for a section on a date."""
    act = get_act(code)
    on = on or date.today()
    if on < act.in_force_from:
        return "not_yet_in_force"
    if act.repealed_on and on >= act.repealed_on:
        return "repealed"
    if any(section == excluded or excluded.startswith(f"{section}(") for excluded in act.not_in_force):
        return "partly_in_force" if section not in act.not_in_force else "not_in_force"
    return "in_force"
