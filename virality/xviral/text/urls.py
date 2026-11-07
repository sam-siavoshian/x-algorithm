"""URL handling: domain extraction, domain classes, media-link detection, optional expansion.

Sources already carry expanded URLs (USC `links`/`media`, Community Archive
`tweet_urls`), so expansion over the network is only needed for third-party
shorteners and is off by default.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

_TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.jp", "co.in", "co.kr",
    "com.br", "com.cn", "com.mx", "co.za", "com.sg", "com.tr", "co.nz",
}

SHORTENERS = {
    "bit.ly", "buff.ly", "ow.ly", "tinyurl.com", "t.ly", "lnkd.in", "dlvr.it", "rebrand.ly",
    "shorturl.at", "cutt.ly", "is.gd", "goo.gl", "tiny.cc", "rb.gy", "trib.al", "linktr.ee",
}

DOMAIN_CLASSES: dict[str, set[str]] = {
    "x_internal": {"x.com", "twitter.com", "t.co", "pic.x.com", "pic.twitter.com"},
    "code": {"github.com", "gitlab.com", "huggingface.co", "arxiv.org", "kaggle.com", "colab.research.google.com", "npmjs.com", "pypi.org"},
    "video_host": {"youtube.com", "youtu.be", "vimeo.com", "loom.com", "twitch.tv", "wistia.com"},
    "app_store": {"apps.apple.com", "apple.com", "play.google.com", "chromewebstore.google.com", "testflight.apple.com"},
    "launch_platform": {"producthunt.com", "ycombinator.com", "news.ycombinator.com", "indiehackers.com", "betalist.com"},
    "writing": {"substack.com", "medium.com", "notion.site", "notion.so", "mirror.xyz", "beehiiv.com", "ghost.io", "dev.to", "hashnode.dev"},
    "social": {"linkedin.com", "instagram.com", "tiktok.com", "reddit.com", "facebook.com", "threads.net", "bsky.app", "discord.gg", "discord.com", "t.me"},
    "news": {"nytimes.com", "wsj.com", "bloomberg.com", "techcrunch.com", "theverge.com", "wired.com", "reuters.com", "ft.com", "cnbc.com", "theinformation.com", "axios.com", "businessinsider.com", "forbes.com", "cnn.com", "bbc.co.uk", "bbc.com", "washingtonpost.com", "theguardian.com"},
    "shortener": SHORTENERS,
}
DOMAIN_CLASS_NAMES = ("none", *DOMAIN_CLASSES.keys(), "other")

_STATUS_MEDIA_RE = re.compile(r"(?:x|twitter)\.com/[^/]+/status/\d+/(?:photo|video)/\d+")


def registrable_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlsplit(url if "://" in url else "http://" + url).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    # Keep known multi-label hosts whole (apps.apple.com, play.google.com, news.ycombinator.com).
    for names in DOMAIN_CLASSES.values():
        if host in names:
            return host
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def domain_class(domain: str | None) -> str:
    if not domain:
        return "none"
    for cls, names in DOMAIN_CLASSES.items():
        if domain in names or any(domain.endswith("." + n) for n in names):
            return cls
    return "other"


def is_media_link(expanded: str | None) -> bool:
    """True for the t.co link X appends to a post pointing at its own photo or video."""
    return bool(expanded and _STATUS_MEDIA_RE.search(expanded))


class UrlExpander:
    """Resolves third-party shorteners with HEAD requests, cached on disk, rate limited.

    Only used for SHORTENERS; everything else is taken as given by the source.
    """

    def __init__(self, cache_path: str | Path, requests_per_second: float = 2.0, timeout: float = 8.0):
        self.cache_path = Path(cache_path)
        self.cache: dict[str, str | None] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.min_interval = 1.0 / max(requests_per_second, 1e-6)
        self.timeout = timeout
        self._last = 0.0
        self._lock = threading.Lock()

    def expand(self, url: str) -> str | None:
        if registrable_domain(url) not in SHORTENERS:
            return url
        if url in self.cache:
            return self.cache[url]
        import requests

        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        try:
            r = requests.head(url, allow_redirects=True, timeout=self.timeout)
            final = r.url
        except requests.RequestException:
            final = None
        self.cache[url] = final
        return final

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
