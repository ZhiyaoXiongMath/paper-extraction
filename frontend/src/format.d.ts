import type { BatchSummary, Draft, Paper, ReadBlock } from "./contracts";
export function paperOutcome(paper: Paper): {
  hasDraft: boolean;
  title: string;
  summary: string;
  initialTab: string;
};
export function label(status?: string): string;
export function tone(status?: string): string;
export function terminal(status?: string): boolean;
export function canDeleteBatch(batch: BatchSummary, allowPaused?: boolean): boolean;
export function eventSummary(event: {
  stage: string;
  data: Record<string, unknown>;
  time: string;
}): string;
export function stageName(stage?: string): string;
export function metrics(batches: BatchSummary[]): {
  total: number;
  queued: number;
  running: number;
  completed: number;
  failed: number;
  paused?: number;
  cancelled?: number;
};
export function formatDate(v?: string): string;
export function shortId(s?: string): string;
export function kindName(v: string): string;
export function readBlocks(draft: Partial<Draft>): ReadBlock[];
