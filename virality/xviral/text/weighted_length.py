"""X's weighted text length and external-URL count.

Line-for-line port of TextMetrics::of in home-mixer/models/content_features.rs.
Rust walks the UTF-8 bytes: ASCII bytes weigh 1 and only \\t..\\r and space are
whitespace; non-ASCII characters use char_weight and Unicode White_Space.
A token that starts with http:// or https:// is a URL: its characters are not
counted, and each external URL adds URL_WEIGHTED_LEN instead. Attachment URLs
(the t.co link X appends for media) are subtracted from the URL count.
"""

from __future__ import annotations

from dataclasses import dataclass

from xviral.constants import SINGLE_WEIGHT_CODEPOINT_RANGES, URL_WEIGHTED_LEN

_ASCII_WS = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20}
# Unicode White_Space property minus ASCII, which is what Rust's char::is_whitespace uses.
_UNICODE_WS = {
    0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
}


def char_weight(cp: int) -> int:
    if cp <= SINGLE_WEIGHT_CODEPOINT_RANGES[0][1]:
        return 1
    if cp in (0xFE0E, 0xFE0F) or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F:
        return 0
    for lo, hi in SINGLE_WEIGHT_CODEPOINT_RANGES:
        if lo <= cp <= hi:
            return 1
    return 2


def _is_url(text: str, i: int) -> bool:
    return text.startswith("https://", i) or text.startswith("http://", i)


@dataclass(frozen=True)
class TextMetrics:
    weighted_len: int
    external_urls: int


def text_metrics(text: str | None, attachment_urls: int = 0) -> TextMetrics:
    if not text:
        return TextMetrics(0, 0)
    weighted = 0
    urls = 0
    in_token = False
    in_url = False
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp < 0x80:
            weight, is_ws = 1, cp in _ASCII_WS
        else:
            weight, is_ws = char_weight(cp), cp in _UNICODE_WS
        if is_ws:
            weighted += weight
            in_token = False
        else:
            if not in_token:
                in_token = True
                in_url = _is_url(text, i)
                urls += int(in_url)
            if not in_url:
                weighted += weight
    external = max(urls - attachment_urls, 0)
    return TextMetrics(weighted + URL_WEIGHTED_LEN * external, external)


def newline_count(text: str | None) -> int:
    """content_features.rs counts '\\n' only."""
    return (text or "").count("\n")
