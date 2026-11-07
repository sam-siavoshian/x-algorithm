"""Language resolution.

X's own `lang` tag is what X's systems use (home-mixer hydrates language_code
from it), so it is trusted when present and linguistic. Our detector is only a
fallback for sources without a tag and for X's 'no linguistic content' tags.
"""

from __future__ import annotations

import re
from functools import lru_cache

_STRIP_RE = re.compile(r"https?://\S+|@\w+|#\w+|\$[A-Za-z]+")
_LETTERS_RE = re.compile(r"[^\W\d_]", re.UNICODE)

_LANGS = (
    "ENGLISH", "SPANISH", "PORTUGUESE", "FRENCH", "GERMAN", "ITALIAN", "DUTCH", "TURKISH",
    "RUSSIAN", "UKRAINIAN", "POLISH", "JAPANESE", "KOREAN", "CHINESE", "ARABIC", "HINDI",
    "INDONESIAN", "VIETNAMESE", "THAI", "PERSIAN", "HEBREW", "SWEDISH", "TAGALOG",
)


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    langs = [getattr(Language, n) for n in _LANGS]
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


def strip_for_lang(text: str | None) -> str:
    return _STRIP_RE.sub(" ", text or "").strip()


def is_non_linguistic(text: str | None, min_letters: int = 3) -> bool:
    """Posts that are only links, mentions, numbers or emoji carry no language signal."""
    return len(_LETTERS_RE.findall(strip_for_lang(text))) < min_letters


def detect(texts: list[str | None]) -> list[str | None]:
    """ISO 639-1 codes, 'zxx' for non-linguistic text, None when the detector abstains."""
    det = _detector()
    out: list[str | None] = []
    cleaned = [strip_for_lang(t) for t in texts]
    todo_idx = [i for i, t in enumerate(texts) if not is_non_linguistic(t)]
    results = det.detect_languages_in_parallel_of([cleaned[i] for i in todo_idx]) if todo_idx else []
    by_idx = dict(zip(todo_idx, results))
    for i, t in enumerate(texts):
        if i not in by_idx:
            out.append("zxx")
            continue
        lang = by_idx[i]
        out.append(lang.iso_code_639_1.name.lower() if lang is not None else None)
    return out


def resolve(lang_src: list[str | None], texts: list[str | None], undetermined: set[str]) -> list[str | None]:
    """Trust X's tag when it is linguistic; otherwise detect."""
    need = [i for i, l in enumerate(lang_src) if not l or l in undetermined]
    detected = detect([texts[i] for i in need]) if need else []
    out = list(lang_src)
    for i, d in zip(need, detected):
        out[i] = d
    return out
