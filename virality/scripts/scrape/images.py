#!/usr/bin/env python3
"""Does what the picture shows predict views beyond the text and the account, across accounts? And do the image-kind
findings of the images track hold on views in the home-timeline scrape?

    python scripts/scrape/images.py            # data/scrape_track/images.json, image_kinds.parquet

Part A (prediction, accounts held out). Five folds by account, every number out of fold.
  stage 1   gradient boosting of log10 views on what is known without reading the post: followers, following, post and
            media counts, blue check, age at capture, hour and weekday, capture day, media type and count, text length.
            With `acct`: plus the account's level, the mean stage-1 residual of the account's OTHER posts (accounts with 5+).
  stage 2   ridge on embeddings to predict the stage-1 residual: text; text + image; text + description; all three.
  metrics   share of residual variance explained, and same-account pairs ranked correctly in held-out accounts
            (differences with intervals from resampling accounts).
Part B (kinds). Kinds are read from X's picture description with rules (KINDS below; 20 per kind were read by hand, see
REPORT). Effects are measured inside accounts on the stage-1 residual (so age at capture and follower count are out),
for views and for likes, with intervals from resampling accounts.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

D, OUT = Path("data/scrape/converted"), Path("data/scrape_track")
R: dict = {}
rng = np.random.default_rng(0)

KINDS = [  # first match wins, on the first 160 characters of the description, lower case
    ("chart_table", r"\b((line|bar|pie|area|candlestick|price|stock|scatter) (graph|chart|plot)s?|graph|chart|plot|histogram|leaderboard|benchmark|table (showing|comparing|listing|titled|displaying)|comparison table|spreadsheet|heatmap)\b"),
    ("post_chat_shot", r"\b(screenshot|screenshots|image) (of|displays|shows|showing) (an? |the |two |several )?(\w+ ){0,3}(tweet|x post|post on x|twitter|social media post|reply|thread|message|messaging|chat|conversation|text exchange|dm|direct message|email|whatsapp|imessage|discord|slack|telegram|reddit|comment|notification)"),
    ("article_shot", r"\b(screenshot|image) (of|displays|shows) (an? |the )?(\w+ ){0,3}(article|headline|news|blog|newsletter|press release|paper|document|pdf|page from|abstract)"),
    ("code_terminal", r"\b(code editor|terminal (window|output|screen|interface|session)|(screenshot|image) of (a |an |the )?(\w+ )?terminal|command line|code snippet|source code|python code|javascript code|console output|ide\b|github (repo|repository|page))"),
    ("product_ui", r"\b(screenshot|screenshots|screen recording|interface|dashboard|app screen|homepage|landing page|website|web page|user interface|\bui\b|mobile app|browser window|laptop screen (showing|displaying))"),
    ("text_card", r"^(a |an |the )?((first )?image (shows|displays|features) )?(an? )?(\w+[ -]){0,3}(graphic|banner|poster|flyer|title card|slide|quote card|background|text|lettering|logo)\b.{0,90}\b(text|title|heading|headline|words|lettering|reads|reading|stating|saying|titled)\b"),
    ("diagram_infographic", r"\b(infographic|diagram|flowchart|flow chart|schematic|architecture|mind map|timeline graphic|illustrated guide|map of)\b"),
    ("meme_cartoon", r"\b(meme|cartoon|comic|anime|stick figure|spongebob|simpsons|pepe|wojak|illustration|drawing|pixel art|digital art|animated|caricature|3d render)"),
    ("person", r"\b(man|woman|men|women|person|people|boy|girl|guy|selfie|portrait|group of|crowd|couple|player|speaker|founder|team|child|baby|athlete)\b"),
]


def pair_stats(acct, y, s, margin=0.1):
    """Same-account pairs: per-account (pairs, correct)."""
    o = np.argsort(acct, kind="stable")
    acct, y, s = acct[o], y[o], s[o]
    st = np.flatnonzero(np.r_[True, acct[1:] != acct[:-1]])
    en = np.r_[st[1:], len(acct)]
    n, c, ids = [], [], []
    for a, b in zip(st, en):
        if b - a < 2:
            continue
        k = min(b - a, 60)                       # at most 60 posts of one account
        idx = np.arange(a, b) if b - a <= 60 else rng.choice(np.arange(a, b), 60, replace=False)
        yy, ss = y[idx], s[idx]
        i, j = np.triu_indices(k, 1)
        m = np.abs(yy[i] - yy[j]) >= margin
        if not m.any():
            continue
        ag = ((ss[i] > ss[j]) == (yy[i] > yy[j]))[m].astype(float)
        n.append(m.sum()); c.append(ag.sum()); ids.append(acct[a])
    return np.array(ids), np.array(n, float), np.array(c, float)


def boot_diff(n, c1, c2, B=1000):
    k = len(n)
    d = []
    for _ in range(B):
        i = rng.integers(0, k, k)
        d.append((c2[i].sum() - c1[i].sum()) / n[i].sum())
    return [float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100)]


def boot_acc(n, c, B=1000):
    k = len(n)
    d = [c[i].sum() / n[i].sum() for i in (rng.integers(0, k, k) for _ in range(B))]
    return [float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100)]


raw = pl.read_parquet(D / "posts.parquet", columns=["id_str"]).with_row_index("ri").unique("id_str", keep="first")
posts = pl.read_parquet(OUT / "posts.parquet").filter(pl.col("kind") == "original").with_columns(pl.col("post_id").cast(pl.String).alias("id_str")).join(raw, on="id_str")
meta = pl.read_parquet(OUT / "post_meta.parquet").select("post_id", "lang", "tech_strong", "off", "akind")
posts = posts.join(meta, on="post_id", how="left").filter(pl.col("lang") == "en")
posts = posts.with_columns(
    pl.col("author_handle").str.to_lowercase().alias("handle"),
    ((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) / 3.6e6).clip(lower_bound=0.01).alias("age_h"),
    pl.col("media").list.len().alias("n_media"), pl.col("media").list.first().struct.field("type").fill_null("none").alias("mtype"),
    pl.col("media").list.first().struct.field("alt").alias("desc"),
    pl.col("views").clip(lower_bound=1).log10().alias("y_views"), (pl.col("likes") + 1).log10().alias("y_likes"))
posts = posts.with_columns(pl.col("handle").cast(pl.Categorical).to_physical().alias("acct"), pl.len().over("handle").alias("acct_n"),
                           ((pl.col("tech_strong").mean().over("handle") >= 0.15) & (pl.col("off").mean().over("handle") <= pl.col("tech_strong").mean().over("handle")) & (pl.len().over("handle") >= 5)).alias("audience"))
n = posts.height
acct = posts["acct"].to_numpy().astype(np.int64)
fold = (acct * 2654435761 % 2**32) % 5
ri = posts["ri"].to_numpy()
R["population"] = {"english_original_posts": n, "accounts": int(len(np.unique(acct)))}

base = posts.select(
    pl.col("author_followers").clip(lower_bound=1).log10().alias("lf"), pl.col("author_following").clip(lower_bound=1).log10().alias("lg"),
    pl.col("author_posts_count").clip(lower_bound=1).log10().alias("lp"), pl.col("author_media_count").clip(lower_bound=1).log10().alias("lm"),
    pl.col("author_verified").cast(pl.Float32).alias("ver"), pl.col("age_h").log10().alias("lage"),
    pl.from_epoch("created_at_ms", "ms").dt.hour().alias("hour"), pl.from_epoch("created_at_ms", "ms").dt.weekday().alias("dow"),
    ((pl.col("snapshot_at_ms") - pl.col("snapshot_at_ms").min()) / 86.4e6).alias("day"), pl.col("n_media"),
    pl.col("mtype").replace_strict({"none": 0, "photo": 1, "video": 2, "gif": 3}, return_dtype=pl.Int32).alias("mt"),
    pl.col("text").str.len_chars().alias("tlen")).to_numpy().astype(np.float32)
P = dict(objective="regression", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=int(__import__("os").environ.get("THREADS", 4)))


def oof_gbm(X, y, rows=None, rounds=300):
    rows = np.arange(len(y)) if rows is None else rows
    pred = np.full(len(y), np.nan, np.float32)
    for f in range(5):
        tr, te = rows[fold[rows] != f], rows[fold[rows] == f]
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr]), rounds)
        pred[te] = m.predict(X[te])
    return pred


res1 = {}
for tgt in ("views", "likes"):
    y = posts[f"y_{tgt}"].to_numpy().astype(np.float32)
    p1 = oof_gbm(base, y)
    r1 = y - p1
    # account level: mean residual of the account's other posts
    s = np.bincount(acct, weights=r1); c = np.bincount(acct)
    loo = (s[acct] - r1) / np.maximum(c[acct] - 1, 1)
    has = c[acct] >= 5
    Xb = np.column_stack([base, np.where(has, loo, np.nan), np.log10(c[acct])]).astype(np.float32)
    rows5 = np.flatnonzero(has)
    p1b = oof_gbm(Xb, y, rows5)
    res1[tgt] = {"y": y, "p1": p1, "r1": r1, "p1b": p1b, "r1b": y - p1b}
    R[f"stage1_{tgt}"] = {"r2_without_account_level": float(1 - np.var(r1) / np.var(y)), "sd_y_log10": float(np.std(y)), "sd_residual": float(np.std(r1)),
                          "accounts_5plus_posts": int(len(rows5)), "r2_with_account_level_on_5plus": float(1 - np.var((y - p1b)[rows5]) / np.var(y[rows5])),
                          "r2_without_account_level_on_5plus": float(1 - np.var(r1[rows5]) / np.var(y[rows5])), "sd_residual_with_account_level": float(np.std((y - p1b)[rows5]))}
print(json.dumps(R, indent=1), flush=True)

# ---- Part A: embeddings on the residual ----
T = np.load(D / "text_emb.npy", mmap_mode="r")
I, Ih = np.load(D / "img_emb.npy", mmap_mode="r"), np.load(D / "img_emb_has.npy")
Dm, Dh = np.load(D / "imgdesc_emb.npy", mmap_mode="r"), np.load(D / "imgdesc_emb_has.npy")
ipos, dpos = np.cumsum(Ih) - 1, np.cumsum(Dh) - 1
has_img = Ih[ri]
has_desc = posts["desc"].is_not_null().to_numpy() & Dh[ri]
mtype = posts["mtype"].to_numpy()


def ridge_oof(X, r, rows, alphas=(1.0, 3.0, 10.0, 30.0, 100.0)):
    """Out-of-fold ridge predictions; alpha chosen inside each training fold on a held-out fifth of its accounts."""
    pred = np.zeros(len(rows), np.float32)
    f = fold[rows]
    a_of = acct[rows]
    chosen = []
    for k in range(5):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X[tr] - mu) / sd
        inner = (a_of[tr] * 40503 % 5) == 0
        G, b = Z[~inner].T @ Z[~inner], Z[~inner].T @ (r[tr][~inner] - r[tr][~inner].mean())
        best, ba = 1e9, alphas[0]
        for al in alphas:
            w = np.linalg.solve(G + al * (~inner).sum() / 100 * np.eye(G.shape[0], dtype=np.float32), b)
            e = np.mean((r[tr][inner] - r[tr][~inner].mean() - Z[inner] @ w) ** 2)
            if e < best:
                best, ba = e, al
        chosen.append(ba)
        w = np.linalg.solve(Z.T @ Z + ba * len(tr) / 100 * np.eye(Z.shape[1], dtype=np.float32), Z.T @ (r[tr] - r[tr].mean()))
        pred[te] = r[tr].mean() + ((X[te] - mu) / sd) @ w
    return pred, chosen


def part_a(name, rows, use_desc, tgt="views", stage="r1"):
    y = res1[tgt]["y"][rows]
    r = res1[tgt][stage][rows].astype(np.float32)
    p_base = y - r
    Xt = np.asarray(T[ri[rows]], np.float32)
    Xi = np.asarray(I[ipos[ri[rows]]], np.float32)
    sets = {"text": Xt, "image": Xi, "text+image": np.hstack([Xt, Xi])}
    if use_desc:
        Xd = np.asarray(Dm[dpos[ri[rows]]], np.float32)
        sets.update({"description": Xd, "text+description": np.hstack([Xt, Xd]), "text+image+description": np.hstack([Xt, Xi, Xd])})
    out = {"posts": int(len(rows)), "accounts": int(len(np.unique(acct[rows]))), "sd_residual": float(np.std(r)), "sets": {}}
    ids0, n0, c0 = pair_stats(acct[rows], y, p_base)
    out["pairs"] = int(n0.sum()); out["pair_accounts"] = int(len(ids0))
    out["base_only_pair_accuracy"] = float(c0.sum() / n0.sum() * 100)
    keep = {}
    for k, X in sets.items():
        pr, al = ridge_oof(X, r, rows)
        ids, nn, cc = pair_stats(acct[rows], y, p_base + pr)
        assert (ids == ids0).all()
        keep[k] = cc
        out["sets"][k] = {"residual_variance_explained": float(1 - np.mean((r - pr) ** 2) / np.var(r)), "corr": float(np.corrcoef(r, pr)[0, 1]),
                          "pair_accuracy": float(cc.sum() / nn.sum() * 100), "pair_accuracy_ci": boot_acc(nn, cc), "alpha": al}
    for a, b in (("text", "text+image"), ("text", "text+description"), ("text", "text+image+description"), ("text+description", "text+image+description")):
        if a in keep and b in keep:
            out[f"{b} minus {a}, points"] = {"diff": float((keep[b].sum() - keep[a].sum()) / n0.sum() * 100), "ci": boot_diff(n0, keep[a], keep[b])}
    out["text minus base, points"] = {"diff": float((keep["text"].sum() - c0.sum()) / n0.sum() * 100), "ci": boot_diff(n0, c0, keep["text"])}
    R.setdefault("part_a", {})[name] = out
    print(name, json.dumps(out, indent=1), flush=True)


photo_desc = np.flatnonzero(has_img & has_desc & (mtype == "photo"))
part_a("photos with description, views, followers controlled", photo_desc, True)
has5 = ~np.isnan(res1["views"]["p1b"])
part_a("photos with description, views, followers and account level controlled (accounts 5+)", np.flatnonzero(has_img & has_desc & (mtype == "photo") & has5), True, stage="r1b")
part_a("all photos, views, followers controlled", np.flatnonzero(has_img & (mtype == "photo")), False)
part_a("video first frames, views, followers controlled", np.flatnonzero(has_img & (mtype == "video")), False)
aud = posts["audience"].to_numpy()
part_a("photos with description, audience accounts, views, account level controlled", np.flatnonzero(has_img & has_desc & (mtype == "photo") & has5 & aud), True, stage="r1b")
part_a("photos with description, likes, followers and account level controlled (accounts 5+)", np.flatnonzero(has_img & has_desc & (mtype == "photo") & has5), True, tgt="likes", stage="r1b")
(OUT / "images.json").write_text(json.dumps(R, indent=1))

# ---- Part B: kinds, inside accounts ----
head = posts["desc"].fill_null("").str.to_lowercase().str.slice(0, 160)
kind = pl.lit("other_photo")
for k, pat in reversed(KINDS):
    kind = pl.when(head.str.contains(pat)).then(pl.lit(k)).otherwise(kind)
posts = posts.with_columns(pl.when(pl.col("desc").is_null()).then(None).otherwise(kind).alias("img_kind"))
posts.filter(pl.col("img_kind").is_not_null()).select("post_id", "handle", "img_kind", "desc", "views", "likes", "age_h", "author_followers").write_parquet(OUT / "image_kinds.parquet")
R["kinds"] = {"counts": posts.filter(pl.col("img_kind").is_not_null())["img_kind"].value_counts(sort=True).to_dicts()}
startup = posts["akind"].is_not_null().to_numpy()


def within(mask, groups: dict[str, np.ndarray], ref: np.ndarray, tgt: str, B=1000):
    """Each group against `ref`, inside accounts: mean over accounts of (mean residual of the group's posts - mean
    residual of the reference posts), accounts weighted by min(n_group, n_ref); and the share of pairs the group wins."""
    r = res1[tgt]["r1"]
    out = {}
    for g, gm in groups.items():
        a_g, a_r = acct[mask & gm], acct[mask & ref]
        both = np.intersect1d(a_g, a_r)
        if len(both) < 15:
            out[g] = {"accounts": int(len(both)), "note": "too few accounts"}
            continue
        d, w, win, npair = [], [], [], []
        gi, rf = np.flatnonzero(mask & gm & np.isin(acct, both)), np.flatnonzero(mask & ref & np.isin(acct, both))
        og, orr = gi[np.argsort(acct[gi], kind="stable")], rf[np.argsort(acct[rf], kind="stable")]
        sg, sr = np.searchsorted(acct[og], both), np.searchsorted(acct[orr], both)
        eg, er = np.searchsorted(acct[og], both, "right"), np.searchsorted(acct[orr], both, "right")
        for i in range(len(both)):
            x, z = r[og[sg[i]:eg[i]]][:40], r[orr[sr[i]:er[i]]][:40]
            d.append(x.mean() - z.mean()); w.append(min(len(x), len(z)))
            win.append((x[:, None] > z[None, :]).sum()); npair.append(len(x) * len(z))
        d, w, win, npair = np.array(d), np.array(w, float), np.array(win, float), np.array(npair, float)
        bs, bw = [], []
        for _ in range(B):
            i = rng.integers(0, len(d), len(d))
            bs.append(10 ** (np.sum(d[i] * w[i]) / w[i].sum())); bw.append(win[i].sum() / npair[i].sum())
        out[g] = {"accounts": int(len(both)), "posts_in_group": int(len(gi)), "reference_posts": int(len(rf)), "pairs": int(npair.sum()),
                  "ratio": float(10 ** (np.sum(d * w) / w.sum())), "ratio_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "win_share": float(win.sum() / npair.sum()), "win_share_ci": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))]}
    return out


ik = posts["img_kind"].to_numpy()
allm = np.ones(n, bool)
single_photo = (mtype == "photo")
none = (mtype == "none")
R["kinds"]["media_vs_no_media"] = {}
for pop_name, pop in (("all accounts", allm), ("audience accounts", aud), ("startup-list accounts", startup)):
    for tgt in ("views", "likes"):
        R["kinds"]["media_vs_no_media"][f"{pop_name}, {tgt}"] = within(pop, {"photo": mtype == "photo", "video": mtype == "video", "gif": mtype == "gif"}, none, tgt)
desc_photo = single_photo & (ik != None)  # noqa: E711
R["kinds"]["kind_vs_other_described_photos"] = {}
for pop_name, pop in (("all accounts", allm), ("audience accounts", aud), ("startup-list accounts", startup)):
    for tgt in ("views", "likes"):
        groups = {k: (ik == k) for k, _ in KINDS} | {"other_photo": ik == "other_photo"}
        res = {}
        for g, gm in groups.items():
            res.update(within(pop & desc_photo, {g: gm}, desc_photo & ~gm, tgt))
        R["kinds"]["kind_vs_other_described_photos"][f"{pop_name}, {tgt}"] = res
# across accounts: the kind's mean stage-1b residual (followers, age and account level out), accounts resampled
R["kinds"]["across_accounts_views_residual"] = {}
r1b = res1["views"]["r1b"]
for k in [k for k, _ in KINDS] + ["other_photo"]:
    m = desc_photo & (ik == k) & has5
    oth = desc_photo & (ik != k) & has5
    if m.sum() < 50:
        continue
    ua = np.unique(acct[m | oth])
    sm, cm = np.bincount(acct[m], weights=r1b[m], minlength=acct.max() + 1), np.bincount(acct[m], minlength=acct.max() + 1)
    so, co = np.bincount(acct[oth], weights=r1b[oth], minlength=acct.max() + 1), np.bincount(acct[oth], minlength=acct.max() + 1)
    bs = []
    for _ in range(500):
        i = rng.choice(ua, len(ua))
        bs.append(10 ** (sm[i].sum() / max(cm[i].sum(), 1) - so[i].sum() / max(co[i].sum(), 1)))
    R["kinds"]["across_accounts_views_residual"][k] = {"posts": int(m.sum()), "accounts": int(len(np.unique(acct[m]))), "ratio_to_other_described_photos": float(10 ** (r1b[m].mean() - r1b[oth].mean())),
                                                       "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
(OUT / "images.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R["kinds"], indent=1))
