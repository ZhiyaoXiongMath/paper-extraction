export interface Provider {
  id: string;
  label: string;
  configured: boolean;
  key_env: string;
  text_model: string;
  vision_model: string;
}
export interface Config {
  providers: Provider[];
  default_provider?: string;
  max_import_files?: number;
  concurrency?: number;
}
export interface SourcePaper {
  id: string;
  name: string;
  kind: string;
  entry: string;
  entries: string[];
  files?: string[];
  file_count?: number;
}
export interface ImportPreview {
  id: string;
  papers: SourcePaper[];
}
export interface BatchSummary {
  id: string;
  created_at: string;
  provider: string;
  counts: Record<string, number>;
  total: number;
}
export interface Candidate {
  candidate_id?: string;
  independent_question?: string | null;
  local_kind?: string;
  local_decision?: string;
  local_reason?: string;
  local_evidence_ids?: string[];
  selection_evidence_ids?: string[];
  needed_by?: string[];
  represented_by?: string;
  context_for?: string[];
  scope_group_id?: string;
  scope_role?: string;
  scope_excerpt?: string;
  scope_evidence_id?: string;
  scope_evidence_version?: string;
  scope_reason_type?: string;
  scope_central_open?: boolean;
  kind: string;
  claim: string;
  assumptions: string;
  significance: string;
  decision?: "admitted" | "excluded" | "unresolved";
  source_number?: string;
  dependencies?: string[];
  decision_reason?: string;
  evidence_ids: string[];
}
export interface Review {
  stage: string;
  attempt: number;
  passed: boolean;
  issues: string[];
  route?: "extract" | "draft" | "pass";
  evidence_ids?: string[];
  defects?: EditorialDefect[];
  repair_targets?: { draft_field: string; current_text: string; diagnostics: string[] }[];
  page?: number;
}
export interface EditorialDefect {
  kind?: string;
  draft_field?: string;
  current_excerpt?: string;
  evidence_id?: string;
  source_excerpt?: string;
  required_change?: string;
  operation?: string;
  replacement_text?: string;
}
export interface EvidenceLink {
  block_id: string;
  evidence_ids: string[];
}
export interface DraftGroup {
  definitions: string[];
  theorem: string;
  paragraph?: string;
}
export interface Draft {
  title: string;
  authors: string[];
  paragraph: string;
  history: string[];
  shared_definitions: string[];
  groups: DraftGroup[];
  references: { key: string; text: string }[];
  evidence_links?: EvidenceLink[];
}
export interface Paper {
  id: string;
  name: string;
  kind: string;
  status: string;
  stage: string;
  reason?: string;
  draft?: Draft;
  links?: EvidenceLink[];
  candidates: Candidate[];
  reviews: Review[];
  events: { stage: string; data: Record<string, unknown>; time: string }[];
}
export interface BatchDetail extends BatchSummary {
  papers: Paper[];
}
export interface Evidence {
  transcribed_from_image?: boolean;
  evidence_type?: string;
  id: string;
  label: string;
  text: string;
  page?: number;
  file?: string;
  line?: number;
}
export interface Probe {
  ok: boolean;
  message: string;
}
export interface ReadBlock {
  id: string;
  heading: string;
  text: string;
}
