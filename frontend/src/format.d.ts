import type { BatchSummary, Draft, ReadBlock } from "./contracts";
export function label(status?: string): string;
export function tone(status?: string): string;
export function terminal(status?: string): boolean;
export function eventSummary(event: {
  stage: string;
  data: Record<string, unknown>;
  time: string;
}): string;
export function stageName(stage?: string): string;
export function metrics(batches: BatchSummary[]): {
  total: number;
  working: number;
  good: number;
  bad: number;
};
export function formatDate(v?: string): string;
export function shortId(s?: string): string;
export function kindName(v: string): string;
export function readBlocks(draft: Partial<Draft>): ReadBlock[];
