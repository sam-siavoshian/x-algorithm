"""Pipeline configuration.

One YAML file drives every stage. Every field has a default, so a config only
needs to name the paths that differ. Thresholds that come from the X repository
are not configurable here; they live in `xviral.constants` with citations.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    work: Path = Path("data/work")
    """Root for every stage output. Each stage writes to work/<stage>/."""
    usc_root: Path | None = None
    """Checkout of sinking8/usc-x-24-us-election (part_*/**.csv.gz)."""
    community_archive: Path | None = None
    """Directory holding Community Archive tables as Parquet (downloaded or bulk)."""
    user_scrape: list[Path] = Field(default_factory=list)
    """JSONL files from the 2025 tech scrape. Listed newest first."""
    analytics: list[Path] = Field(default_factory=list)
    """X analytics CSV exports, one per account."""
    seed_accounts: Path | None = None
    """Optional text file of handles that define the target audience, one per line."""

    def stage(self, name: str) -> Path:
        p = self.work / name
        p.mkdir(parents=True, exist_ok=True)
        return p


class CommunityArchiveCfg(BaseModel):
    rest_url: str = "https://fabxmporizzqflnftavs.supabase.co/rest/v1"
    anon_key_env: str = "CA_ANON_KEY"
    """The archive's public anon key, documented in its docs/api-doc.md. Read from this env var."""
    min_created: str = "2023-01-01"
    page_size: int = 1000
    requests_per_second: float = 4.0
    tables: list[str] = Field(
        default_factory=lambda: [
            "all_account",
            "tweets",
            "tweet_media",
            "tweet_urls",
            "quote_tweets",
            "followers",
            "following",
            "profile",
            "likes",
            "liked_tweets",
            "archive_upload",
        ]
    )


class FilterCfg(BaseModel):
    min_created: str = "2023-01-01"
    """Views are public only for posts after Dec 2022."""
    maturity_hours: float = 72.0
    """Minimum hours between creation and count snapshot. X serves posts for 48 h."""
    keep_kinds: list[str] = Field(default_factory=lambda: ["original", "quote"])
    keep_langs: list[str] = Field(default_factory=lambda: ["en"])
    undetermined_langs: list[str] = Field(
        default_factory=lambda: ["und", "zxx", "qme", "qht", "qam", "qct", "qst", "art"]
    )
    """X language tags that mean 'no linguistic content'; these fall back to our detector."""
    max_counts_over_views: float = 2.0
    """Rows where likes exceed views by this factor are corrupt snapshots."""


class HygieneCfg(BaseModel):
    max_originals_per_24h: int = 200
    min_follow_ratio: float = 0.05
    min_following_for_ratio_rule: int = 5000
    max_same_domain_share: float = 0.8
    min_posts_for_domain_rule: int = 20
    max_copypasta_events: int = 20


class RelevanceCfg(BaseModel):
    embedder: str = "Qwen/Qwen3-Embedding-0.6B"
    """Cheap text encoder for the relevance pre-pass. 'hashing' uses a CPU fallback (tests)."""
    negatives_sample: int = 500_000
    threshold: float | None = None
    """Post-level probability threshold. None means choose by target precision on the audit set."""
    target_precision: float = 0.85
    account_keep_whole: float = 0.6
    account_keep_partial: float = 0.3
    account_window: int = 50
    max_rows_per_account: int = 2000
    max_rows_per_dup_cluster: int = 50
    audit_sample: int = 1000
    political_blocklist: list[str] = Field(
        default_factory=lambda: [
            "trump", "biden", "kamala", "harris", "maga", "democrat", "republican",
            "gop", "election", "ballot", "vance", "walz", "rnc", "dnc", "senate",
            "congress", "liberal", "conservative", "abortion", "immigration",
        ]
    )
    """Cheap pre-thin applied to the USC corpus only, before the embedding pass."""
    blocklist_min_hits: int = 1
    topic_margin_min: float = 0.0
    """Keep a post only if it is closer to a target-topic description than to any off-target one."""
    heldout_negative_fraction: float = 0.1


class LabelCfg(BaseModel):
    full_weight_views: int = 1000
    min_views_for_rate: int = 1
    pair_window_days: float = 7.0
    pair_tie_margin: float = 0.2
    max_pairs_per_author: int = 200
    trailing_window: int = 50
    recency_weights: dict[int, float] = Field(default_factory=lambda: {2023: 0.25, 2024: 0.5, 2025: 1.0})
    """Weight by creation year; years after the last key take the last key's weight."""
    analytics_weight_multiplier: float = 2.0


class SplitCfg(BaseModel):
    test_time_fraction: float = 0.10
    val_author_fraction: float = 0.10
    seed: int = 17


class MediaCfg(BaseModel):
    max_images: int = 300_000
    max_videos: int = 30_000
    requests_per_second: float = 5.0
    concurrency: int = 8
    timeout_s: float = 20.0
    vlm_side: int = 448
    siglip_side: int = 384
    jpeg_quality: int = 90
    embed_frames: int = 4
    """Frames per video for the X-faithful embedding (X's v8 plan uses up to 4)."""
    temporal_frames: int = 16
    """Additional ~1 fps frames for SigLIP temporal features, plus first and last."""
    max_video_seconds_for_asr: int = 600
    user_agent: str = "xviral-research/0.1"
    keep_source_bytes: bool = False


class EmbedCfg(BaseModel):
    mm_backend: str = "transformers"
    """'transformers' (local GPU) or 'http' (SGLang server), as in phoenix/reference/mm_encoder.py."""
    mm_endpoint: str | None = None
    mm_max_images: int = 8
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    whisper_model: str = "large-v3-turbo"
    device: str = "cuda"
    batch_size: int = 32
    sid_levels: int = 6
    sid_centroids: int = 256
    sid_iters: int = 25
    sid_sample: int | None = 2_000_000


class Config(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    community_archive: CommunityArchiveCfg = Field(default_factory=CommunityArchiveCfg)
    filters: FilterCfg = Field(default_factory=FilterCfg)
    hygiene: HygieneCfg = Field(default_factory=HygieneCfg)
    relevance: RelevanceCfg = Field(default_factory=RelevanceCfg)
    labels: LabelCfg = Field(default_factory=LabelCfg)
    splits: SplitCfg = Field(default_factory=SplitCfg)
    media: MediaCfg = Field(default_factory=MediaCfg)
    embed: EmbedCfg = Field(default_factory=EmbedCfg)
    workers: int = 4

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
