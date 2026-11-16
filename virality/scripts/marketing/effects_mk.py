#!/usr/bin/env python3
"""Outside marketing claims tested on the corpus with the launch track's method (scripts/launch/effects.py).

    python scripts/marketing/effects_mk.py      # reads data/marketing/table.parquet, writes data/marketing/effects_mk.json

Same outcome, estimators, scopes and rule as the launch track; the estimator functions below are copied from
scripts/launch/effects.py unchanged (that file runs on import, so it cannot be imported).
Outcome: lift = log((likes+1)/(account's trailing median likes+1)); secondary: reposts likewise.
Per trait: pairs (same account, <=365 days apart, exactly one has the trait, lifts differ by 1.25x+, posting order balanced,
<=50 pairs per account), ratio (accounts with both, precision-weighted), adjusted (account fixed effects, the launch
track's traits plus every marketing text trait). 95% intervals from resampling accounts (2,000 draws; 500 for adjusted).

RULE, fixed before any outcome was read (identical to scripts/launch/build_launch_evidence.py):
  MAIN = launch-type posts of company and founder accounts; STRICT = those whose opening is an announcement verb followed
  by a product name; ALL = launch-type posts of every account.
  too_thin   under 200 pairs or 25 accounts in MAIN
  survives   same direction in MAIN and STRICT, every test passes in one of them and at least one passes in the other
  main_only  every test passes in MAIN, same direction in STRICT
  wider_only every test passes in ALL, same direction as MAIN
  leaning    same direction everywhere and at least one test passes somewhere
  none       otherwise
  If MAIN is too thin the same rule is applied one level wider (ALL as the main set, strict posts of all accounts as the
  strict set); a result reached that way is marked pooled and can be at most grade C.
Verdict against the outside claim (which states a direction): confirmed = survives / main_only / wider_only in the claimed
direction; contradicted = the same in the opposite direction; not_confirmed = tested, no effect that passes (with the
leaning if any); untestable = too thin at both levels, or the trait cannot be read from the corpus, or the comparison is
circular (the trait is a consequence of the outcome).
"""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/marketing")
from traits import TEXT_TRAITS
rng = np.random.default_rng(0)
T = pl.read_parquet("data/marketing/table.parquet")
T = T.with_columns((pl.col("chars") >= 280).alias("long280"), (pl.col("lines") >= 3).alias("lines3"), (pl.col("dow") >= 6).alias("weekend"),
                   pl.col("hour_utc").is_between(18, 23).alias("h18_23"), pl.col("hour_utc").is_between(0, 12).alias("h00_12"),
                   (pl.col("gap_prev_h") < 2).fill_null(False).alias("prev2h"), (pl.col("gap_next_h") < 2).fill_null(False).alias("next2h"),
                   (pl.col("media_kind") == "video").alias("video"), (pl.col("media_kind") == "image").alias("image"), (pl.col("media_kind") != "none").alias("media"),
                   (pl.col("ext_link") & ~pl.col("points_to_reply")).alias("link_in_post"), (pl.col("photo_n") >= 2).alias("multi_image"))
STRICT = r"(?:[Ii]ntroducing|[Aa]nnouncing|[Uu]nveiling|[Mm]eet|[Ll]aunching|[Ll]aunch(?:ed)?|[Rr]eleasing|[Rr]eleased?|[Oo]pen[- ]?sourc(?:ed|ing)|INTRODUCING|ANNOUNCING)[:\s]+(?:the |our |my |a |an )?(?:new |all-new )?[@A-Z0-9][\w.@-]+"
T = T.with_columns(pl.col("text").fill_null("").str.replace_all(r"https?://\S+", "").str.slice(0, 300).str.contains(STRICT).alias("strict"))
LAUNCH = ["launch", "open_source"]; CUT = 1776902400000
CA = pl.col("source") == "community_archive"
isl, cf = pl.col("ptype").is_in(LAUNCH), pl.col("akind") != "other"
SCOPES = {"launch_cf": isl & cf, "launch_strict_cf": isl & cf & pl.col("strict"), "launch_all": isl, "launch_strict_all": isl & pl.col("strict"),
          "launch_all_before_cutoff": isl & (pl.col("created_at_ms") < CUT), "launch_all_after_cutoff": isl & (pl.col("created_at_ms") >= CUT),
          "launch_all_under_100k": isl & pl.col("tier").is_in(["<1k", "1k-10k", "10k-100k"]), "launch_all_100k_plus": isl & (pl.col("tier") == "100k+"),
          "launch_ai_model_all": isl & pl.col("ai_model_release")}
c = pl.col
# name: (description, with, restrict-to or None, kind)   kind: text | timing | context | circular
A = {k: (d, c(k), None, "text") for k, (d, _) in TEXT_TRAITS.items()}
A |= {
    "waitlist_vs_available_now": ("the post points to a waitlist / early access, against one that says the product is available now", c("cta_waitlist"), c("cta_waitlist") | (c("availability_now") & ~c("cta_waitlist")), "text"),
    "benchmark_with_image": ("a benchmark claim with an image attached (a chart), against an image without a benchmark claim", c("benchmark_claim"), CA & c("image"), "text"),
    "demo_video": ("a video with demonstration wording, against a video without it", c("demo_words"), CA & c("video"), "text"),
    "video_in_model_release": ("a video or GIF against no media, model releases only", c("video"), CA & c("ai_model_release") & (c("video") | ~c("media")), "text"),
    "image_in_model_release": ("an image against no media, model releases only", c("image"), CA & c("ai_model_release") & (c("image") | ~c("media")), "text"),
    "multi_image": ("two or more images, against one image", c("multi_image"), CA & c("image"), "text"),
    "open_source_in_model_release": ("open source / open weights wording, model releases only", c("offer_open_source"), c("ai_model_release"), "text"),
    "pt_weekday_morning": ("posted Monday to Friday 06:00 to 11:59 Pacific time, against any other time", c("pt_weekday_morning"), None, "timing"),
    "pt_weekday_8_10": ("posted Monday to Friday 08:00 to 10:59 Pacific time, against any other time", c("pt_weekday_8_10"), None, "timing"),
    "pt_tue_thu_morning": ("posted Tuesday to Thursday 06:00 to 11:59 Pacific, against any other time", c("pt_tue_thu_morning"), None, "timing"),
    "pt_morning_vs_afternoon": ("weekday 06:00 to 11:59 Pacific, against weekday 12:00 to 17:59 Pacific", c("pt_weekday_morning"), c("pt_weekday_morning") | c("pt_weekday_afternoon"), "timing"),
    "pt_morning_vs_evening_night": ("weekday 06:00 to 11:59 Pacific, against 18:00 to 05:59 Pacific on any day", c("pt_weekday_morning"), c("pt_weekday_morning") | c("pt_evening_night"), "timing"),
    "pt_tuesday": ("posted on a Tuesday (Pacific), against Monday, Wednesday, Thursday or Friday", c("pt_tuesday"), ~c("pt_weekend"), "timing"),
    "pt_weekend": ("posted on Saturday or Sunday (Pacific)", c("pt_weekend"), None, "timing"),
    "teaser_72h": ("the account posted a teaser 1 to 72 hours before the launch post", c("teaser_72h"), None, "context"),
    "teaser_36h": ("the account posted a teaser 1 to 36 hours before the launch post", c("teaser_36h"), None, "context"),
    "quiet_prev24h": ("the account posted nothing in the 24 hours before the launch post", c("quiet_prev24h"), None, "context"),
    "launch_in_prev_week": ("the account made another launch-type post 1 to 7 days earlier (launch week, repeated launches)", c("launch_in_prev_week"), None, "context"),
    "self_quote_day2_7": ("the account quote-posts its own launch 1 to 7 days later (follow-up)", c("self_quote_day2_7"), None, "circular"),
    "followup_announce_day2_4": ("the account posts another announcement (launch, feature, milestone, demo) 1 to 4 days after", c("followup_announce_day2_4"), None, "circular"),
    "active_day2_4": ("the account posts three or more times 1 to 4 days after", c("active_day2_4"), None, "circular"),
    "quoted_1h": ("another corpus account quote-posts the launch within 1 hour", c("quoted_1h"), None, "circular"),
    "team_quoted_1h": ("a first-person (same team) quote post by another corpus account within 1 hour", c("team_quoted_1h"), None, "circular"),
    "team_quoted_1h_vs_later": ("a same-team quote post within 1 hour, against launches whose first same-team quote came 1 to 24 hours after", c("team_quoted_1h"), c("team_quoted_24h"), "circular"),
    "quoted_1h_vs_later": ("a quote post by another corpus account within 1 hour, against launches first quoted 1 to 48 hours after", c("quoted_1h"), c("quoted_48h"), "circular"),
}
ADJ = ["video", "image", "link_in_post", "points_to_reply", "thread_marker", "open_announce", "open_problem", "open_result", "open_number", "names_product_first_line", "long280", "lines3", "tags", "hashtag",
       "is_quote", "weekend", "h18_23", "h00_12", "prev2h", "next2h"] + [k for k in TEXT_TRAITS] + ["teaser_72h", "launch_in_prev_week"]
ADJ_KEY = {"waitlist_vs_available_now": "cta_waitlist", "pt_weekend": "weekend"}

def boot_ci(stat, n, B=2000):
    idx = rng.integers(0, n, (B, n))
    b = np.array([stat(i) for i in idx]); b = b[np.isfinite(b)]
    return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) > B / 2 else (float("nan"), float("nan"))

def pairs_est(q, y):
    q = q.filter(pl.col(y).is_not_null()).select("author_id", "created_at_ms", "post_id", "a", pl.col(y).alias("y"))
    w, n = q.filter(pl.col("a")), q.filter(~pl.col("a"))
    p = w.join(n, on="author_id", suffix="_n").filter(((pl.col("created_at_ms") - pl.col("created_at_ms_n")).abs() <= 365 * 86_400_000) & ((pl.col("y") - pl.col("y_n")).abs() >= np.log(1.25)))
    if p.height == 0: return None
    p = p.with_columns((pl.col("y") > pl.col("y_n")).alias("win"), (pl.col("created_at_ms") < pl.col("created_at_ms_n")).alias("wf"))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 50)
    n1, n2 = p.filter(pl.col("wf")).height, p.filter(~pl.col("wf")).height
    per = p.group_by("author_id").agg((pl.col("win") & pl.col("wf")).sum().alias("w1"), pl.col("wf").sum().alias("n1"), (pl.col("win") & ~pl.col("wf")).sum().alias("w2"), (~pl.col("wf")).sum().alias("n2")).sort("author_id")
    w1, m1, w2, m2 = (per[c].to_numpy().astype(float) for c in ("w1", "n1", "w2", "n2"))
    def bal(i):
        a, b = m1[i].sum(), m2[i].sum()
        if a < 1 or b < 1: return np.nan
        return 0.5 * (w1[i].sum() / a + w2[i].sum() / b)
    rate = bal(np.arange(len(w1))); lo, hi = boot_ci(bal, len(w1))
    return {"win_rate": None if not np.isfinite(rate) else round(float(rate), 4), "lo": round(lo, 4), "hi": round(hi, 4), "pairs": int(p.height), "accounts": int(per.height), "pairs_with_first": int(n1), "pairs_with_second": int(n2)}

def ratio_est(q, y):
    q = q.filter(pl.col(y).is_not_null())
    per = q.group_by("author_id").agg(pl.col(y).filter(pl.col("a")).mean().alias("m1"), pl.col("a").sum().alias("n1"), pl.col(y).filter(~pl.col("a")).mean().alias("m0"), (~pl.col("a")).sum().alias("n0")).filter((pl.col("n1") > 0) & (pl.col("n0") > 0)).sort("author_id")
    if per.height < 5: return None
    dlt = (per["m1"] - per["m0"]).to_numpy(); w = (per["n1"] * per["n0"] / (per["n1"] + per["n0"])).to_numpy().astype(float)
    st = lambda i: float(np.sum(w[i] * dlt[i]) / np.sum(w[i]))
    lo, hi = boot_ci(st, len(w))
    return {"ratio": round(float(np.exp(st(np.arange(len(w))))), 3), "ratio_lo": round(float(np.exp(lo)), 3), "ratio_hi": round(float(np.exp(hi)), 3), "accounts_with_both": int(per.height), "posts_with": int(per["n1"].sum()), "posts_without": int(per["n0"].sum())}

def adjusted(S, y, names):
    q = S.filter(pl.col(y).is_not_null() & CA)
    extra = [pl.col("ptype") == t for t in sorted(set(q["ptype"].unique()) - {"launch"})]
    q = q.with_columns([pl.col(c).cast(pl.Float64) for c in names] + [e.cast(pl.Float64).alias(f"_t{j}") for j, e in enumerate(extra)])
    names = names + [f"_t{j}" for j in range(len(extra))]
    q = q.filter(pl.len().over("author_id") >= 2)
    X = q.select(names).to_numpy(); Y = q[y].to_numpy(); g = q["author_id"].to_numpy()
    u, inv = np.unique(g, return_inverse=True)
    def dm(M):
        s = np.zeros((len(u),) + M.shape[1:]); np.add.at(s, inv, M); cn = np.bincount(inv).astype(float)
        return M - (s / cn.reshape(-1, *([1] * (M.ndim - 1))))[inv]
    Xd, Yd = dm(X), dm(Y)
    beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
    rows = [np.where(inv == k)[0] for k in range(len(u))]
    B = []
    for _ in range(500):
        pick = rng.integers(0, len(u), len(u)); ii = np.concatenate([rows[k] for k in pick])
        B.append(np.linalg.lstsq(Xd[ii], Yd[ii], rcond=None)[0])
    B = np.array(B)
    return {n: {"ratio": round(float(np.exp(beta[j])), 3), "ratio_lo": round(float(np.exp(np.quantile(B[:, j], 0.025))), 3), "ratio_hi": round(float(np.exp(np.quantile(B[:, j], 0.975))), 3)} for j, n in enumerate(names) if not n.startswith("_t")} | {
        "_n": {"posts": int(len(Y)), "accounts": int(len(u))}}

out = {"scopes": {}, "kinds": {k: v[3] for k, v in A.items()}, "descriptions": {k: v[0] for k, v in A.items()}}
for sk, sf in SCOPES.items():
    S = T.filter(sf & pl.col("lift_likes").is_not_null() & CA)
    res = {"posts": S.height, "accounts": S["author_id"].n_unique(), "effects": {}}
    for k, (desc, fa, restrict, kind) in A.items():
        q = (S.filter(restrict) if restrict is not None else S).with_columns(fa.fill_null(False).alias("a"))
        res["effects"][k] = {"posts_with": q.filter(pl.col("a")).height, "accounts_with": q.filter(pl.col("a"))["author_id"].n_unique(),
                             "likes": {"pairs": pairs_est(q, "lift_likes"), "within_account": ratio_est(q, "lift_likes")},
                             "reposts": {"pairs": pairs_est(q, "lift_reposts"), "within_account": ratio_est(q, "lift_reposts")}}
    res["adjusted_likes"] = adjusted(S, "lift_likes", ADJ)
    out["scopes"][sk] = res
    print(f"\n== {sk}: {S.height} posts, {res['accounts']} accounts")
    print(f"{'trait':<30}{'n+':>5} {'win':>6} {'interval':>12} {'pairs':>6} {'acc':>4} | {'ratio':>5} {'interval':>11} {'acc':>4} | {'adj':>5} {'interval':>11} | {'RP win':>6} {'interval':>12}")
    for k, e in res["effects"].items():
        p, r, a, rp = e["likes"]["pairs"], e["likes"]["within_account"], res["adjusted_likes"].get(ADJ_KEY.get(k, k)), e["reposts"]["pairs"]
        ps = f"{100*p['win_rate']:5.1f}% {100*p['lo']:5.1f}-{100*p['hi']:<5.1f} {p['pairs']:>6} {p['accounts']:>4}" if p and p["win_rate"] is not None else " " * 31
        rs = f"{r['ratio']:5.2f} {r['ratio_lo']:5.2f}-{r['ratio_hi']:<5.2f} {r['accounts_with_both']:>4}" if r else " " * 22
        as_ = f"{a['ratio']:5.2f} {a['ratio_lo']:5.2f}-{a['ratio_hi']:<5.2f}" if a else " " * 17
        rps = f"{100*rp['win_rate']:5.1f}% {100*rp['lo']:5.1f}-{100*rp['hi']:<5.1f}" if rp and rp["win_rate"] is not None else ""
        print(f"{k:<30}{e['posts_with']:>5} {ps} | {rs} | {as_} | {rps}")
json.dump(out, open("data/marketing/effects_mk.json", "w"), indent=1)
