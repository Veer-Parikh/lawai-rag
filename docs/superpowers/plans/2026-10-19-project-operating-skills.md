# Project Operating Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repository-level `skills/` directory that gives future agents durable, actionable instructions for the India Criminal-Law Change Monitor.

**Architecture:** Keep the skill files as concise Markdown operating contracts. `copilot-instructions.md` defines universal rules, `decisions.md` records dated choices and reasons, and the frontend/backend/RAG files define subsystem-specific invariants. They document the approved design without prematurely scaffolding the Next.js migration or retrieval rewrite.

**Tech Stack:** Markdown, existing Python/FastAPI/Pinecone RAG repository, planned Next.js frontend, BM25/HNSW/RRF hybrid retrieval, OpenTelemetry-compatible observability.

**Spec:** `docs/superpowers/specs/2026-10-19-india-criminal-law-change-monitor-design.md`

## Global Constraints

- “Legal-reviewable citations and temporal correctness take priority over answer fluency.”
- “Official source availability, terms, rate limits, and redistribution rights must be verified before automated production ingestion.”
- “The first release covers central criminal-law materials” focused on BNS, BNSS, BSA and historical IPC/CrPC/Indian Evidence Act crosswalks.
- “Personalized legal advice” is outside scope.
- English primary sources are the initial language boundary.

## Review Focus

- A future agent must not treat a semantically similar stale section as current; `skills/rag.md` must require explicit temporal metadata.
- A future agent must not cite explanatory material as legal authority; `skills/copilot-instructions.md` and `skills/rag.md` must define source precedence.
- A future agent must not silently answer when evidence conflicts or is incomplete; `skills/backend.md` and `skills/rag.md` must define abstention.
- A future agent must not add the Next.js frontend by expanding Streamlit in place; `skills/frontend.md` must define the migration boundary.
- A future agent must not call lightweight JSONL telemetry “production observability”; `skills/rag.md` and `skills/backend.md` must require correlated traces and metrics.

### Task 1: Create repository operating skill files

**Files:**
- Create: `skills/copilot-instructions.md`
- Create: `skills/decisions.md`
- Create: `skills/frontend.md`
- Create: `skills/backend.md`
- Create: `skills/rag.md`

**Interfaces:**
- Consumes: approved design in `docs/superpowers/specs/2026-10-19-india-criminal-law-change-monitor-design.md`.
- Produces: Markdown instructions future coding agents can follow without reopening the brainstorming conversation.

- [ ] **Step 1: Write the five Markdown files with explicit scope, invariants, and commands**
- [ ] **Step 2: Verify every required topic is present with targeted text search**
- [ ] **Step 3: Verify Markdown files are non-empty and the worktree diff is clean of whitespace errors**

### Task 2: Record the implementation decision

**Files:**
- Modify: `skills/decisions.md`

**Interfaces:**
- Consumes: the approved product boundary and existing repository structure.
- Produces: an append-only decision log with dates, rationale, alternatives, and consequences.

- [ ] **Step 1: Record the product boundary, source precedence, frontend migration, hybrid RAG, and observability decisions**
- [ ] **Step 2: Verify decisions are dated and each includes a reason**

