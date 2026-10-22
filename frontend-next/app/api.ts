// Typed client for the FastAPI service. Every call attaches the session token.

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "lawai.session";
const DEVICE_KEY = "lawai.device";

export type Role = "researcher" | "admin";

export type Session = {
  token: string;
  name: string;
  role: Role;
  expires_at: number;
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export function readStoredSession(): Session | null {
  try {
    const raw = window.localStorage.getItem(TOKEN_KEY);
    if (!raw) return null;
    const session = JSON.parse(raw) as Session;
    return session.expires_at * 1000 > Date.now() ? session : null;
  } catch {
    return null;
  }
}

export function storeSession(session: Session | null) {
  try {
    if (session) window.localStorage.setItem(TOKEN_KEY, JSON.stringify(session));
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage can be unavailable (private mode); the session then lasts for this tab only.
  }
}

/** A random per-browser ID so a demo researcher gets their own matters back next time. */
function deviceId(): string | undefined {
  try {
    let id = window.localStorage.getItem(DEVICE_KEY);
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem(DEVICE_KEY, id);
    }
    return id;
  } catch {
    return undefined;
  }
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const session = typeof window === "undefined" ? null : readStoredSession();
  const headers = new Headers(init.headers);
  if (session) headers.set("Authorization", `Bearer ${session.token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, `The research service at ${API_URL} is not reachable. Start the API and try again.`);
  }
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      // Non-JSON error body; keep the status message.
    }
    throw new ApiError(response.status, detail);
  }
  return response;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await send(path, init);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });

// ── Auth ─────────────────────────────────────────────────────────────────────

export const signInDemo = () => request<Session>("/auth/demo", json("POST", { device_id: deviceId() }));

export const signInAdmin = (password: string) => request<Session>("/auth/admin", json("POST", { password }));

export const fetchMe = () => request<Omit<Session, "token">>("/auth/me");

// ── Document comparison ─────────────────────────────────────────────────────

export type DocumentSummary = {
  document_id: string;
  title: string;
  act_code: string;
  publisher: string;
  source_url: string;
  status: string;
  publication_date?: string | null;
  effective_date?: string | null;
  source_authority: string;
};

export type DocumentChangeResponse = {
  from_document_id: string;
  to_document_id: string;
  changed_sections: string[];
  added_sections: string[];
  removed_sections: string[];
};

export const listDocuments = () => request<DocumentSummary[]>("/documents");

export const compareDocuments = (fromDocumentId: string, toDocumentId: string) =>
  request<DocumentChangeResponse>(
    `/documents/${encodeURIComponent(fromDocumentId)}/compare/${encodeURIComponent(toDocumentId)}`,
  );

// ── Research ─────────────────────────────────────────────────────────────────

export type Source = {
  chunk_id: string;
  document_id?: string | null;
  source_file: string;
  source_url?: string | null;
  publisher?: string | null;
  source_authority?: string | null;
  doc_type: string;
  act_name?: string | null;
  case_name?: string | null;
  citation?: string | null;
  section?: string | null;
  publication_date?: string | null;
  effective_date?: string | null;
  effective_until?: string | null;
  legal_status?: string | null;
  page_anchor?: string | null;
  score: number;
};

export type CitationReference = {
  text: string;
  verified: boolean;
  source_chunk_ids: string[];
  source_urls: string[];
  page_anchors: string[];
};

export type ChatResponse = {
  answer: string;
  relevant_sections: string[];
  legal_explanation: string;
  disclaimer: string;
  citation_warnings: string[];
  citations: CitationReference[];
  confidence_tier: "HIGH" | "MEDIUM" | "LOW";
  sources: Source[];
  domain_detected: string;
  domain_router_stage: string;
  latency_ms: number;
  context_chunks_used: number;
};

export const askQuestion = (body: { query: string; as_of?: string; session_id?: string }) =>
  request<ChatResponse>("/chat", json("POST", { domain: "criminal", top_k: 5, ...body }));

export type FeedbackKind = "helpful" | "wrong_source" | "stale_status" | "missing_authority" | "poor_explanation";

export type FeedbackPayload = {
  query: string;
  kind: FeedbackKind;
  comment?: string;
  share_question?: boolean;
  answer?: string;
  citations?: string[];
  sources?: { title: string; section?: string | null; status?: string | null }[];
  confidence_tier?: string;
};

export const sendFeedback = (body: FeedbackPayload) =>
  request<{ recorded: boolean; queued_for_review: boolean }>("/feedback", json("POST", body));

// ── Legal reference: Acts, crosswalk, provisions, timelines ──────────────────

export type ActSummary = {
  code: string;
  title: string;
  act_number: string;
  status: "in_force" | "repealed";
  in_force_from: string;
  repealed_on: string | null;
  repealed_by: string | null;
  successor: string | null;
  predecessor: string | null;
  commencement: string | null;
  not_in_force: string[];
  sections_in_catalog: number;
  catalog_built_at: string | null;
};

export type CrosswalkRow = {
  id: string;
  from_code: string | null;
  from_section: string | null;
  from_heading: string | null;
  to_code: string | null;
  to_section: string | null;
  to_heading: string | null;
  relation: "equivalent" | "modified" | "omitted" | "new";
  note: string;
  verified: boolean;
  verified_by: string | null;
  verified_at: string | null;
};

export type CrosswalkResult = { note: string; total: number; verified: number; rows: CrosswalkRow[] };

export type Provision = {
  code: string;
  section: string;
  act_title: string;
  heading: string | null;
  text: string;
  page_anchor: string | null;
  status: string;
  status_label: string;
  source_url: string;
  extracted_from: string;
};

export type TimelineEvent = {
  date: string;
  kind: string;
  title: string;
  detail: string;
  code: string | null;
  section: string | null;
  instrument: string | null;
  url: string | null;
};

export type Timeline = {
  code: string;
  section: string;
  act_title: string;
  heading: string | null;
  status: string;
  status_label: string;
  source_url: string;
  text: string | null;
  page_anchor: string | null;
  events: TimelineEvent[];
  related: CrosswalkRow[];
};

const sectionPath = (code: string, section: string) =>
  `${encodeURIComponent(code)}/${encodeURIComponent(section.trim())}`;

export const listActs = () => request<ActSummary[]>("/acts");

export const searchCrosswalk = (params: { q?: string; code?: string; relation?: string }) => {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v) as [string, string][]);
  return request<CrosswalkResult>(`/crosswalk?${query}`);
};

export const fetchProvision = (code: string, section: string) =>
  request<Provision>(`/provisions/${sectionPath(code, section)}`);

export const fetchTimeline = (code: string, section: string) =>
  request<Timeline>(`/timeline/${sectionPath(code, section)}`);

export const verifyCrosswalkRow = (id: string, verified: boolean) =>
  request<CrosswalkRow>(`/admin/crosswalk/${encodeURIComponent(id)}`, json("PATCH", { verified }));

// ── Changes ──────────────────────────────────────────────────────────────────

export type ChangeEvent = {
  id: number;
  source_id: string;
  title: string;
  url: string | null;
  authority: string | null;
  outcome: "new" | "changed" | "unchanged" | "failed" | "added";
  checked_at: string;
  content_hash: string | null;
  previous_hash: string | null;
  error: string | null;
  detail: string | null;
};

export const fetchChanges = (limit = 50) => request<{ events: ChangeEvent[] }>(`/changes?limit=${limit}`);

// ── Matters ──────────────────────────────────────────────────────────────────

export type SavedAnswer = Pick<
  ChatResponse,
  "answer" | "confidence_tier" | "citations" | "sources" | "citation_warnings" | "legal_explanation" | "disclaimer"
>;

export type MatterEntry = {
  id: string;
  matter_id: string;
  created_at: string;
  question: string;
  as_of: string | null;
  answer: SavedAnswer;
  note: string;
};

export type Matter = {
  id: string;
  title: string;
  reference: string;
  notes: string;
  created_at: string;
  updated_at: string;
  entry_count: number;
  entries: MatterEntry[];
};

export const listMatters = () => request<Matter[]>("/matters");
export const fetchMatter = (id: string) => request<Matter>(`/matters/${encodeURIComponent(id)}`);
export const createMatter = (body: { title: string; reference?: string; notes?: string }) =>
  request<Matter>("/matters", json("POST", body));
export const updateMatter = (id: string, body: { title?: string; reference?: string; notes?: string }) =>
  request<Matter>(`/matters/${encodeURIComponent(id)}`, json("PATCH", body));
export const deleteMatter = (id: string) => request<void>(`/matters/${encodeURIComponent(id)}`, { method: "DELETE" });
export const saveToMatter = (id: string, body: { question: string; as_of?: string | null; answer: SavedAnswer; note?: string }) =>
  request<MatterEntry>(`/matters/${encodeURIComponent(id)}/entries`, json("POST", body));
export const updateEntryNote = (matterId: string, entryId: string, note: string) =>
  request<MatterEntry>(`/matters/${encodeURIComponent(matterId)}/entries/${encodeURIComponent(entryId)}`, json("PATCH", { note }));
export const deleteEntry = (matterId: string, entryId: string) =>
  request<void>(`/matters/${encodeURIComponent(matterId)}/entries/${encodeURIComponent(entryId)}`, { method: "DELETE" });

export async function downloadMemo(matterId: string): Promise<{ filename: string; text: string }> {
  const response = await send(`/matters/${encodeURIComponent(matterId)}/memo.md`);
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename = /filename="([^"]+)"/.exec(disposition)?.[1] ?? "memo.md";
  return { filename, text: await response.text() };
}

// ── Admin ────────────────────────────────────────────────────────────────────

export type DailyPoint = {
  date: string;
  total: number;
  high: number;
  medium: number;
  low: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
};

export type Analytics = {
  window_days: number;
  window_start: string;
  window_end: string;
  totals: {
    requests: number;
    answered: number;
    abstained: number;
    abstention_rate: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    tokens_used: number;
    citation_warnings: number;
    rewritten_queries: number;
    sessions: number;
    feedback: number;
  };
  confidence: Record<"HIGH" | "MEDIUM" | "LOW", number>;
  domains: Record<string, number>;
  router_stages: Record<string, number>;
  daily: DailyPoint[];
  recent: {
    timestamp: string;
    query_ref: string;
    domain: string;
    confidence_tier: string;
    top_chunk_score: number;
    chunks_used: number;
    citation_warnings: number;
    latency_ms: number;
    answered: boolean;
  }[];
  feedback: {
    by_kind: Record<string, number>;
    recent: { timestamp: string; kind: string; query_ref: string; comment: string | null }[];
  };
  http: {
    counters: { name: string; labels: Record<string, string>; value: number }[];
    observations: { name: string; labels: Record<string, string>; count: number; mean: number }[];
  };
};

export type LedgerEntry = {
  path: string;
  doc_type: string;
  status: "ingested" | "dry_run" | "failed";
  chunks_created: number;
  chunks_uploaded: number;
  error: string | null;
  job_id: string | null;
  sha256: string | null;
  ingested_at: string;
};

export type CorpusFile = {
  path: string;
  name: string;
  folder: string;
  topic: string;
  kind: "pdf" | "html";
  ingestible: boolean;
  empty: boolean;
  size_bytes: number;
  modified_at: string;
  title: string | null;
  publisher: string | null;
  authority: string | null;
  source_url: string | null;
  acquired_via: string | null;
  retrieved_at: string | null;
  sha256: string | null;
  ingestion: LedgerEntry | null;
};

export type Corpus = {
  summary: {
    files: number;
    pdfs: number;
    ingested: number;
    failed: number;
    pending: number;
    empty: number;
    chunks_uploaded: number;
    bytes: number;
    unavailable_sources: number;
  };
  files: CorpusFile[];
  unavailable: { source_id: string; title: string; url: string; authority: string; error: string }[];
  destinations: { folder: string; label: string; authority: string }[];
  history: LedgerEntry[];
};

export type IngestJob = {
  job_id: string;
  status: "started" | "running" | "completed" | "failed";
  target_path: string;
  dry_run: boolean;
  requested_by: string;
  started_at: string | null;
  finished_at: string | null;
  progress_done: number;
  progress_total: number;
  current_file: string;
  files_found: number;
  files_processed: number;
  total_chunks_created: number;
  total_chunks_uploaded: number;
  errors: string[];
};

export type BackgroundJob<R = unknown> = {
  job_id: string;
  kind: string;
  status: "running" | "completed" | "failed";
  requested_by: string;
  started_at: string;
  finished_at: string | null;
  progress_done: number;
  progress_total: number;
  current: string;
  result: R | null;
  error: string | null;
};

export type IndexStats =
  | { available: true; index: string; dimension: number; total_vectors: number; namespaces: Record<string, number> }
  | { available: false; index: string; error: string };

export const fetchAnalytics = (days: number) => request<Analytics>(`/admin/analytics?days=${days}`);
export const fetchCorpus = () => request<Corpus>("/admin/corpus");
export const fetchIndexStats = () => request<IndexStats>("/admin/index");
export const fetchJobs = () => request<IngestJob[]>("/admin/ingest/jobs");

export const startIngest = (target_path: string, options: { dry_run?: boolean; skip_ingested?: boolean } = {}) =>
  request<IngestJob>("/admin/ingest", json("POST", { target_path, doc_type: "all", dry_run: false, skip_ingested: false, ...options }));

export const uploadDocument = (form: FormData) =>
  request<{ path: string; size_bytes: number }>("/admin/corpus/upload", { method: "POST", body: form });

export type CatalogBuildResult = { code: string; status: "built" | "skipped" | "failed"; sections: number; reason: string | null };

export const startCatalogBuild = (codes?: string[]) =>
  request<BackgroundJob<CatalogBuildResult[]>>("/admin/catalog/build", json("POST", { codes: codes ?? null }));
export const fetchCatalogBuild = () => request<{ job: BackgroundJob<CatalogBuildResult[]> | null }>("/admin/catalog/build");

export type WatchedSource = {
  source_id: string;
  title: string;
  url: string;
  authority: string;
  document_type: string;
  last_check: ChangeEvent | null;
  last_change: ChangeEvent | null;
};

export type SourcesOverview = {
  sources: WatchedSource[];
  not_watched: { source_id: string; title: string; authority: string | null; reason: string }[];
  interval_hours: number;
  job: BackgroundJob<Record<string, number>> | null;
};

export const fetchSources = () => request<SourcesOverview>("/admin/sources");
export const startSourceCheck = (source_ids?: string[]) =>
  request<BackgroundJob<Record<string, number>>>("/admin/sources/check", json("POST", { source_ids: source_ids ?? null }));

export type ReviewStatus = "open" | "confirmed" | "dismissed" | "promoted";

export type ReviewItem = {
  id: number;
  created_at: string;
  kind: Exclude<FeedbackKind, "helpful">;
  query_hash: string;
  question: string | null;
  answer: string | null;
  citations: string[];
  sources: { title: string; section?: string | null; status?: string | null }[];
  confidence_tier: string | null;
  comment: string | null;
  status: ReviewStatus;
  reviewer_note: string | null;
  expected_sections: string[];
  must_abstain: boolean;
  resolved_by: string | null;
  resolved_at: string | null;
  eval_case_id: string | null;
};

export type EvalDataset = {
  name: string;
  cases: number;
  abstention_cases: number;
  categories: Record<string, number>;
  from_review: number;
};

export const fetchReview = (status?: ReviewStatus) =>
  request<{ counts: Record<ReviewStatus, number>; items: ReviewItem[] }>(`/admin/review${status ? `?status=${status}` : ""}`);
export const resolveReview = (
  id: number,
  body: { status: "open" | "confirmed" | "dismissed"; reviewer_note?: string; expected_sections?: string[]; must_abstain?: boolean },
) => request<ReviewItem>(`/admin/review/${id}`, json("PATCH", body));
export const promoteReview = (id: number) =>
  request<{ eval_case: { id: string }; dataset: string }>(`/admin/review/${id}/promote`, { method: "POST" });
export const fetchEvalDatasets = () => request<{ datasets: EvalDataset[] }>("/admin/evals");
