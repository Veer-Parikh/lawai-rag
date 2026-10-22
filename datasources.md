# LawAI ingestion data sources

This document defines the recommended corpus for the first LawAI release. The
system is intentionally focused on **central Indian criminal law**: the BNS,
BNSS, BSA, their historical equivalents, legally operative notifications, and
a small curated set of judgments for interpretation.

## Source priority

When sources disagree, preserve both records but apply this priority:

1. **Gazette of India** — controls enactment, commencement, amendment, repeal,
   and other legally operative notifications.
2. **India Code** — canonical consolidated text and section structure.
3. **Ministry of Home Affairs (MHA)** — official criminal-law publications and
   explanatory material.
4. **Supreme Court of India** — authoritative judgments and interpretations.
5. **Department of Justice, PIB, Law Commission, and Parliament** —
   explanatory, legislative-history, or policy context; do not treat these as
   substitutes for the enacted text.

Every ingested document should retain its original file, normalized text,
source URL, publisher, retrieval timestamp, document hash, publication date,
effective date when known, language, document type, and extraction/OCR status.

## 1. Current primary statutes — mandatory

Ingest the complete English text, schedules, footnotes, and available
amendment history for:

| Document | What to ingest | Official source |
|---|---|---|
| Bharatiya Nyaya Sanhita, 2023 (BNS) | Full Act, all sections, explanations, illustrations, schedules, commencement status, and amendments | [India Code](https://www.indiacode.nic.in/) · [MHA New Criminal Laws](https://www.mha.gov.in/en/commoncontent/new-criminal-laws) |
| Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) | Full Act, all sections, provisos, schedules, commencement status, and amendments | [India Code](https://www.indiacode.nic.in/) · [MHA New Criminal Laws](https://www.mha.gov.in/en/commoncontent/new-criminal-laws) |
| Bharatiya Sakshya Adhiniyam, 2023 (BSA) | Full Act, all sections, illustrations, schedules, commencement status, and amendments | [India Code](https://www.indiacode.nic.in/) · [MHA New Criminal Laws](https://www.mha.gov.in/en/commoncontent/new-criminal-laws) |

Use the India Code search portal when a direct Act handle changes:

- [India Code search](https://www.indiacode.nic.in/handle/123456789/1362/simple-search)
- [BNS search query](https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=bharatiya+nyaya+sanhita)
- [BNSS search query](https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=bharatiya+nagarik+suraksha+sanhita)
- [BSA search query](https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=bharatiya+sakshya+adhiniyam)

## 2. Historical statutes for crosswalks — mandatory

These are required to answer “old section to new section” questions and to
preserve historical meaning. Ingest the enacted text plus relevant amendments,
not just a modern summary.

| Document | Official source |
|---|---|
| Indian Penal Code, 1860 (IPC) | [India Code — IPC](https://www.indiacode.nic.in/handle/123456789/2263) |
| Code of Criminal Procedure, 1973 (CrPC) | [India Code — CrPC](https://www.indiacode.nic.in/handle/123456789/15098) |
| Indian Evidence Act, 1872 (IEA) | [India Code](https://www.indiacode.nic.in/) — search for “Indian Evidence Act, 1872” |
| Repealing and Amending Acts affecting the legacy statutes | [India Code](https://www.indiacode.nic.in/) and [e-Gazette](https://egazette.gov.in/) |

Also create structured crosswalk records for IPC → BNS, CrPC → BNSS, and IEA →
BSA. A crosswalk is supporting metadata; it must not replace retrieval from
the actual provision text.

## 3. Gazette and legally operative instruments — mandatory

Use the Gazette as the legal-effect source. Ingest the original PDFs and
extract the notification number, date, issuing authority, subject, affected
provisions, territory, conditions, and effective interval.

- [e-Gazette of India](https://egazette.gov.in/)
- [e-Gazette home/search](https://egazette.gov.in/(S(pupiain1jkne1uzbexcpbn3h))/default.aspx)

At minimum, search for and ingest:

- BNS, BNSS, and BSA enactment publications from 2023.
- Commencement notifications bringing provisions into force, including
  notifications covering 1 July 2024 and any exceptions or staged provisions.
- Central amendment Acts and amendment notifications from 2023 onward.
- Repeal, savings, transitional, and adaptation orders.
- Rules, regulations, forms, and notifications made under BNS, BNSS, or BSA.
- Notifications that change commencement dates, territorial application, or
  provision-level exceptions.
- Directly related Central Government and MHA statutory orders.

Useful Gazette search terms:

`Bharatiya Nyaya Sanhita`, `Bharatiya Nagarik Suraksha Sanhita`,
`Bharatiya Sakshya Adhiniyam`, `commencement`, `S.O. 850(E)`, `repeal`,
`amendment`, `savings`, `transitional`, `rules`.

## 4. Ministry of Home Affairs — mandatory supporting corpus

Ingest official MHA material that explains or operationalizes the three new
criminal laws:

- [MHA New Criminal Laws — English](https://www.mha.gov.in/en/commoncontent/new-criminal-laws)
- [MHA home page](https://www.mha.gov.in/)
- [MHA documents and notifications](https://www.mha.gov.in/en/documents)
- [MHA Parliament questions and answers](https://www.mha.gov.in/en/divisionofmha/parliament-questions)

Include:

- Official Act PDFs when published by MHA.
- Implementation notices, FAQs, handbooks, advisories, and explanatory
  material.
- Official replies to Parliament about commencement, implementation, or
  interpretation.
- Central notifications and circulars that directly affect the new codes.

Mark this material as **explanatory** unless it is itself a Gazette
notification or another primary legal instrument.

## 5. Supreme Court judgments — curated mandatory corpus

Do not ingest every judgment initially. Start with a reviewed collection of
judgments that directly interpret:

- IPC, CrPC, and IEA provisions that map to BNS, BNSS, or BSA.
- The new criminal codes and their commencement or transitional provisions.
- Arrest, bail, remand, investigation, charge, trial, evidence, sentencing,
  and procedural safeguards relevant to the pilot.
- Constitutional or statutory decisions that materially affect criminal-law
  interpretation.

Official sources:

- [Supreme Court eSCR judgment portal](https://judgments.ecourts.gov.in/supremecourt)
- [Supreme Court of India](https://www.sci.gov.in/)
- [Supreme Court judgments page](https://www.sci.gov.in/judgements/)

For each judgment, ingest the judgment PDF/HTML, metadata, case number,
judgment date, bench, parties, citations, paragraph anchors, and any official
corrigendum. Prefer the official judgment source over a third-party copy.

## 6. Department of Justice and legal-policy material — recommended

Use these sources for legislative context and official policy documents:

- [Department of Justice — Acts and Policies](https://doj.gov.in/documents/acts-and-policies)
- [Department of Justice](https://doj.gov.in/)
- [Law Commission reports](https://lawcommissionofindia.nic.in/reports.htm)
- [Law Commission reports by subject](https://lawcommissionofindia.nic.in/category-wise-reports/)

Prioritize criminal-law reports, consultation papers, and reports that explain
the reforms behind the BNS, BNSS, BSA, IPC, CrPC, or IEA. Store these as
**contextual sources**, not controlling law.

## 7. Parliamentary and legislative history — recommended

Ingest only material directly relevant to the three new codes and their
amendments:

- Bills and introduced text.
- Statement of Objects and Reasons.
- Parliamentary debates and committee reports.
- Standing committee reports.
- Amendment schedules and voting records.
- Official answers explaining implementation or transitional issues.

Source portals:

- [Digital Sansad](https://sansad.in/)
- [Lok Sabha](https://loksabha.nic.in/)
- [Rajya Sabha](https://rajyasabha.nic.in/)

Keep bill text, enacted text, and committee commentary as separate document
types. Legislative history can explain intent but cannot override the enacted
statute or Gazette.

## 8. Press Information Bureau — optional explanatory material

Use PIB only for dated government announcements and implementation context:

- [Press Information Bureau](https://pib.gov.in/)
- [PIB search](https://pib.gov.in/PressReleasePage.aspx)

Do not use a PIB release as proof of the legal commencement or wording of a
provision when a Gazette notification is available.

## 9. Research/bootstrap datasets — optional and non-authoritative

These can help bootstrap discovery, deduplication, or evaluation, but every
record should be linked back to an official source before it is used in a
user-facing answer:

- [AWS Open Data Registry](https://registry.opendata.aws/)
- [Indian Supreme Court judgment datasets on AWS](https://registry.opendata.aws/indian-supreme-court-judgments/)
- [Kaggle](https://www.kaggle.com/datasets) — use only datasets with verified
  provenance and redistribution rights.

Do not rank an unofficial dataset above India Code, the Gazette, MHA, or an
official court judgment.

## 10. Languages and versions

For the initial release:

- Ingest English primary sources first.
- Add Hindi versions only when the same document can be matched to an English
  source ID and the language/version metadata is preserved.
- Keep every retrieved version; never overwrite a previous document.
- Record `valid_from`, `valid_to`, `publication_date`, `effective_date`, and
  `status` (`enacted`, `notified`, `in_force`, `superseded`, `repealed`, or
  `unknown`).
- Preserve page numbers, section numbers, subsection labels, paragraph
  numbers, schedules, and notification clauses as retrieval anchors.

## Do not ingest in the first release

Exclude these until the central criminal-law corpus and provenance workflow are
stable:

- State-specific criminal statutes and state amendments.
- Every High Court judgment or unreviewed case-law dump.
- Civil, tax, corporate, family, labour, or property law.
- Unverified legal blogs, commentary, or user-generated summaries as
  authoritative sources.
- Private client documents or personally identifiable case records.

Before enabling automated refresh, verify the source terms of use, rate limits,
robots/access requirements, PDF redistribution rights, and whether the source
permits local text extraction and indexing.
