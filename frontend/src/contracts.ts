export interface Provider {
  id: string;
  label: string;
  configured: boolean;
  text_model: string;
  vision_model: string;
}
export interface Config {
  providers: Provider[];
  default_provider?: string;
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
  page?: number;
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
