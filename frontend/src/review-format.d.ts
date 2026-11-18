import type { Draft, EditorialDefect, Review } from "./contracts";
export function reviewField(field?: string): string;
export function evidenceLabel(id: string): string;
export function defectKind(kind?: string): string;
export function editLabel(operation?: string): string;
export function readableReviewText(raw: string, references?: Draft["references"]): string;
export function reviewDetails(review: Review): { defects: EditorialDefect[]; plain: string[] };
