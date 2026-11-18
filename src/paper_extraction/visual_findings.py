"""Classify minor prose wrapping without suppressing unknown visual defects."""
import re

from .models import Review


# Each issue is a complete finding. Mixed findings remain actionable, even when
# they also mention an ordinary line break. Unknown descriptions stay actionable.
_MATERIAL = re.compile(
    r"clip\w*|truncat\w*|overlap\w*|overprint\w*|collid\w*|collision|overflow\w*|overfull|"
    r"out(?:side| of) (?:the )?(?:page|margin|frame|text)|cut[ -]?off|off[ -]?page|"
    r"unreadable|illegible|obscur\w*|missing|lost|disappear\w*|cropp\w*|"
    r"wrong|incorrect|dangling|detach\w* (?:sub|super)script|split (?:sub|super)script|"
    r"display(?:ed)? (?:math|formula|equation)|alignment (?:is )?(?:broken|wrong)|"
    r"page break|across pages|next page|截断|裁切|重叠|溢出|遮挡|缺失|不可读|跨页",
    re.I,
)
_CONTINUATION = re.compile(
    r"(?:\b(?:breaks?|wraps?|ends?)\b.{0,180}\b(?:start|begin)\w* (?:of |on |at )?(?:the )?next line|"
    r"\b(?:breaks?|wraps?)\b.{0,110}across lines.{0,120}starts? the next\b|"
    r"\b(?:single|isolated|lone) word\b.{0,80}(?:last|final|own) line|"
    r"\b(?:widow|orphan) word\b.{0,70}(?:last|final) line|"
    r"(?:单词|词语|行内符号).{0,20}(?:单独成行|出现在下一行开头))",
    re.I | re.S,
)


def actionable_page_review(review: Review) -> Review:
    issues = [issue for issue in review.issues if _MATERIAL.search(issue) or not _CONTINUATION.search(issue)]
    if issues == review.issues:
        return review
    return Review(passed=not issues, issues=issues,
                  locations=[location for location in review.locations if location.issue in issues])
