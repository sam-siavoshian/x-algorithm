"""Draft scoring: raw history + drafts -> features -> stored transforms -> report."""

import numpy as np
import polars as pl
import pytest
import torch

from tests.conftest import frame, make_post
from xviral import serve
from xviral.embed.store import EmbeddingStore
from xviral.model import features, train
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable

T0 = 1_760_000_000_000
HOUR = 3_600_000


def embed(texts: list[str]) -> np.ndarray:
    """A stand-in encoder: one direction for launch posts, another for the rest, plus a little text-dependent noise."""
    out = np.zeros((len(texts), 16), np.float32)
    for i, t in enumerate(texts):
        out[i, 0 if "launch" in t else 1] = 1.0
        out[i, 2 + len(t) % 14] = 0.2
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def corpus(rng) -> pl.DataFrame:
    rows, pid = [], 1
    for a in range(1, 25):
        base = float(rng.uniform(1.0, 2.2))                       # the account's own level, log10 views
        t = T0 + int(rng.integers(0, 24)) * HOUR
        for _ in range(40):
            t += int(rng.integers(5, 30)) * HOUR
            good = rng.random() < 0.4
            views = int(10 ** (base + (0.8 if good else 0.0) + rng.normal(0, 0.15)) * 20)
            text = ("launch day: we shipped the new agent" if good else "some thoughts on the week") + f" {pid}"
            rows.append(make_post(pid, a, t, text, views=views, likes=max(views // 40, 0), replies=views // 400, reposts=views // 300,
                                  author_followers=int(2000 * a), source="community_archive"))
            pid += 1
    return frame(rows)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A nano model trained on a corpus with a planted signal: posts that say 'launch' do several times better."""
    tmp_path = tmp_path_factory.mktemp("serve")
    rng = np.random.default_rng(0)
    posts = corpus(rng)
    df = serve.feature_frame(posts)
    df = df.with_columns(pl.when(pl.col("author_id") > 21).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split"))
    store = EmbeddingStore(tmp_path / "emb", dim=16)
    store.write(df["post_id"].to_numpy(), embed(features.render_texts(df)))
    features.build(df, tmp_path / "table", emb=store)
    table = FeatureTable.load(tmp_path / "table")
    mcfg = train.model_cfg_for(table, "nano", dropout=0.0)
    train.train(table, tmp_path / "run", mcfg, TrainCfg(steps=300, batch_size=32, lr=2e-3, warmup_steps=30, eval_every=150, ema_decay=0.98), log_every=10_000)

    return tmp_path, posts, table, mcfg


def test_draft_scoring_end_to_end(trained):
    tmp_path, posts, table, mcfg = trained
    scorer = serve.DraftScorer(tmp_path / "run" / "best.pt", embed)
    history = posts.filter(pl.col("author_id") == 23)
    when = int(history["created_at_ms"].max()) + 30 * HOUR
    account = {"handle": "user23", "followers": 46_000, "following": 300}
    drafts = [{"text": "some thoughts on the week ahead"}, {"text": "launch day: we shipped the new agent today"}]

    # Stored transforms: history rows come out of serving exactly as they sit in the training table.
    t2, rows2 = scorer.featurize(history, drafts, account, when)
    assert t2.n == history.height + 2 and rows2["source"].to_list()[-2:] == ["draft", "draft"]
    # Each draft is the account's next post: neither sees the other as a previous post.
    assert rows2["history_n"].to_list()[-2:] == [history.height] * 2 and rows2["hours_since_prev_original"].to_list()[-2:] == pytest.approx([30.0, 30.0])
    ids = t2.t["post_id"][:-2]
    pos = {int(p): i for i, p in enumerate(table.t["post_id"].tolist())}
    at = torch.tensor([pos[int(p)] for p in ids])
    assert torch.allclose(t2.t["num"][:-2], table.t["num"][at], atol=1e-5)
    assert torch.equal(t2.t["cat"][:-2], table.t["cat"][at]) and torch.equal(t2.t["counts"][:-2], table.t["counts"][at])
    assert (t2.t["counts"][-2:] == -1).all()                       # drafts carry no outcome

    reports = scorer.score(history, drafts, account, when)
    plain, launch = reports
    assert plain["text"] == drafts[0]["text"] and plain["history_posts_used"] == min(40, mcfg.max_history)
    assert launch["views_24h"]["q50"] > 1.5 * plain["views_24h"]["q50"], (plain["views_24h"], launch["views_24h"])
    assert launch["p_own_top_10pct"]["views"] > plain["p_own_top_10pct"]["views"]
    assert plain["views_24h"]["q10"] <= plain["views_24h"]["q50"] <= plain["views_24h"]["q90"]
    # Alternatives are isolated: a draft's score does not depend on which other drafts it is compared with.
    alone = scorer.score(history, drafts[1:], account, when)[0]
    assert abs(alone["views_24h"]["q50"] - launch["views_24h"]["q50"]) < 1e-3 * launch["views_24h"]["q50"]
    assert scorer.score(history, drafts[::-1], account, when)[0]["views_24h"]["q50"] == launch["views_24h"]["q50"]   # order-free
    again = scorer.score(history, drafts, account, when)
    assert again[1]["views_24h"] == launch["views_24h"]
    # Deterministic gates ride along with every report.
    g = {x["gate"]: x for x in launch["gates"]}
    assert g["in_network_store"]["status"] == "pass" and g["cold_start_slot"]["status"] == "pass" and g["expiry"]["hours"] == 48
    assert g["author_diversity"]["hours_since_previous"] == pytest.approx(30.0) and g["author_diversity"]["second_post_multiplier"] == pytest.approx(0.625)
    assert g["exploration_target"]["views_needed"]["24h"] == round(0.03 * 46_000) and g["out_of_network_factor"]["factor"] == 0.75
    big = scorer.score(history, drafts[:1], {**account, "followers": 900_000}, when)[0]
    assert {x["gate"]: x for x in big["gates"]}["cold_start_slot"]["status"] == "info"
    # With a feature cache, a second scorer reuses the model-independent part and returns the same report.
    scorer.feature_cache = {}
    first_pass = scorer.score(history, drafts, account, when)
    other = serve.DraftScorer(tmp_path / "run" / "best.pt", lambda texts: (_ for _ in ()).throw(AssertionError("embedded again")))
    other.feature_cache = scorer.feature_cache
    assert len(other.feature_cache) == 1 and other.score(history, drafts, account, when)[1]["views_24h"] == first_pass[1]["views_24h"]
    scorer.feature_cache = None
    # Posting-time sweep and link variants reuse the cached history embeddings.
    calls = []
    scorer._embed_raw = lambda texts: (calls.append(len(texts)), embed(texts))[1]
    sweep = scorer.timing(history, drafts[1], account, start_ms=when, hours_ahead=12, step_hours=4)
    assert len(sweep) == 4 and sorted(r["hours_from_start"] for r in sweep) == [0, 4, 8, 12]
    assert sweep[0]["views_q50"] == max(r["views_q50"] for r in sweep) and sum(calls) == 0       # nothing new to embed
    link = {"text": "launch day: the new agent is live https://foo.dev read more below"}
    lv = scorer.link_variants(history, link, account, when)
    assert [r["variant"] for r in lv] == ["as written", "link moved to the end", "link removed (put it in a reply)"]
    assert lv[1]["text"].endswith("https://foo.dev") and "http" not in lv[2]["text"] and scorer.link_variants(history, drafts[0], account, when) == []
    # An account with no history still gets a report; the lift fields are empty.
    cold = scorer.score(history.head(0), drafts[:1], {"handle": "newco", "followers": 150, "author_id": 999}, when)[0]
    assert cold["lift_vs_own_median"]["views"] is None and cold["views_24h"]["q50"] >= 0


def test_advice_gates(trained):
    from xviral import advise

    tmp_path, posts, _, _ = trained
    ck = tmp_path / "run" / "best.pt"
    scorer = serve.DraftScorer([ck, ck], embed)                       # a two-member ensemble, so the agreement gate has members to read
    history = posts.filter(pl.col("author_id") == 23)
    when = int(history["created_at_ms"].max()) + 30 * HOUR
    account = {"handle": "user23", "followers": 46_000, "following": 300}
    adv = advise.Advisor(scorer, scales={"views": 6.0, "likes": 6.0})
    # The planted signal: the user's launch version is recommended over a plain draft, with members in agreement.
    plain = {"text": "some thoughts on the week ahead"}
    out = adv.advise(history, plain, account, when, alternatives=[{"text": "launch day: we shipped the new agent today", "label": "launch version"}])
    lv = {c["change"]: c for c in out["changes"]}["launch version"]
    assert lv["status"].startswith("recommended") and lv["p_better"] > 0.7 and lv["members_agree"] is True and lv["predicted_ratio"] > 1.5
    assert out["changes"][0]["change"] == "launch version" and out["outcome_compared"] == "likes"
    # Every applicable operator is tried, each as one change, and each carries its measured evidence.
    draft = {"text": "https://foo.dev some thoughts on the week. It was long. It was good. #monday"}
    ops = {c["operator"]: c for c in adv.advise(history, draft, account, when)["changes"] if c["operator"]}
    assert set(ops) == {"attach_image", "link_out_of_first_line", "remove_hashtags"}     # the link leads, so line layout is not applied on top
    assert ops["attach_image"]["adds_media"] and ops["attach_image"]["evidence"]["supported"] and ops["attach_image"]["evidence"]["pairs"] > 1000
    assert ops["link_out_of_first_line"]["text"].endswith("https://foo.dev") and not ops["link_out_of_first_line"]["text"].startswith("http")
    # A change the model likes but that has no measured effect is never recommended.
    assert not ops["remove_hashtags"]["evidence"]["supported"] and not ops["remove_hashtags"]["status"].startswith("recommended")
    assert adv._status({"p_better": 0.9, "members_agree": True}, {"supported": False})[0] == "untested"
    assert adv._status({"p_better": 0.9, "members_agree": True}, {"supported": True})[0] == "recommended_likely"
    assert adv._status({"p_better": 0.65, "members_agree": True}, {"supported": True})[0] == "recommended"
    assert adv._status({"p_better": 0.9, "members_agree": False}, {"supported": True})[0] == "no_confident_difference"
    assert adv._status({"p_better": 0.55, "members_agree": True}, {"supported": True})[0] == "no_confident_difference"
    assert adv._status({"p_better": 0.9, "members_agree": True}, {"supported": True}, meaning_ok=False)[0] == "no_confident_difference"
    assert adv._status({"p_better": 0.1, "members_agree": True}, {"supported": True})[0] == "predicted_worse"
    assert advise.Advisor(scorer, scales={}).advise(history, draft, account, when)["changes"][0]["status"] == "uncalibrated"
    # Disagreeing members are detected from their medians.
    cmp = adv._compare({"views_24h": {"q50": 100}, "likes": {"q50": 5}, "members": [{"views_q50": 100, "likes_q50": 5}, {"views_q50": 100, "likes_q50": 5}]},
                       {"views_24h": {"q50": 400}, "likes": {"q50": 9}, "members": [{"views_q50": 900, "likes_q50": 9}, {"views_q50": 60, "likes_q50": 9}]}, "views")
    assert cmp["members_agree"] is False and cmp["p_better"] > 0.9
    # Spacing: a draft 30 minutes behind the previous post is compared with itself two hours behind.
    soon = adv.advise(history, {"text": "some thoughts on the week"}, account, int(history["created_at_ms"].max()) + HOUR // 2)
    w = [c for c in soon["changes"] if c["operator"] == "wait_for_gap"]
    assert len(w) == 1 and w[0]["evidence"]["ratio_when_closer"] < 1 and any(g["gate"] == "author_diversity" for g in soon["rules"])
    # The account's own record states only differences its history supports, and the text report holds together.
    rec = advise.own_record(history.with_columns(pl.when(pl.col("text").str.contains("launch")).then(pl.lit([dict(advise.PHOTO)], dtype=history.schema["media"]))
                                                 .otherwise(pl.col("media")).alias("media")))
    assert [f["attribute"] for f in rec] == ["an image or video"] and rec[0]["ratio"] > 3 and rec[0]["ratio_lo"] > 1
    assert advise.own_record(history) == []
    text = advise.render(out)
    assert "better (" in text and "launch version" in text and "Also checked:" in text and "Changes we stand behind:" in text
    assert "No single change we can stand behind" in advise.render(advise.Advisor(scorer, scales={}).advise(history, draft, account, when))


def test_operators_change_one_thing():
    from xviral import advise

    ops = {o.name: o for o in advise.OPERATORS}
    d = {"text": "We shipped a thing. It is fast. It is cheap.\nhttps://x.dev/blog"}
    lines = ops["one_point_per_line"].apply(d, {})["text"]
    assert lines.split("\n\n") == ["We shipped a thing.", "It is fast.", "It is cheap.", "https://x.dev/blog"]
    assert ops["link_out_of_first_line"].apply(d, {}) is None                      # the link is not in the first line: nothing to move
    one = {"text": "We shipped a thing. It is fast. It is cheap. https://x.dev/blog"}
    assert ops["link_out_of_first_line"].apply(one, {})["text"] == "We shipped a thing. It is fast. It is cheap.\n\nhttps://x.dev/blog"
    assert ops["one_point_per_line"].apply(one, {}) is None                        # would move the link too: two changes at once
    assert ops["one_point_per_line"].apply({"text": "a\nb\nc"}, {}) is None and ops["one_point_per_line"].apply({"text": "Short. Two."}, {}) is None
    assert ops["attach_image"].apply({"text": "x", "media": [dict(advise.PHOTO)]}, {}) is None and ops["remove_hashtags"].apply({"text": "no tags"}, {}) is None
    grow = lambda text, instruction: text + " It cuts build time from 90 to 54 seconds on a 200-file project, with no config change."   # noqa: E731
    assert ops["add_detail"].apply({"text": "Builds are faster now."}, {"rewrite_fn": grow})["text"].startswith("Builds are faster now.")
    assert ops["add_detail"].apply({"text": "Builds are faster now."}, {"rewrite_fn": lambda t, i: t + " Really."}) is None   # not the tested change
    ev = advise.load_evidence()
    assert ev["media"]["supported"] and ev["lines"]["supported"] and ev["longer"]["supported"] and ev["link_not_first"]["supported"]
    assert not ev["question_open"]["supported"] and not ev["no_hashtag"]["supported"]


def test_product_service_and_http(trained):
    """The launch advisor: versions compared for a known account, a pasted history, and a new account; then over HTTP."""
    from fastapi.testclient import TestClient

    from xviral.app import server, service
    from xviral.text import post_type

    tmp_path, posts, table, mcfg = trained
    scorer = serve.DraftScorer(tmp_path / "run" / "best.pt", embed)
    posts.write_parquet(tmp_path / "posts.parquet")
    svc = service.Service(scorer, {"likes": 6.0, "views": 6.0}, posts=tmp_path / "posts.parquet")
    plain, launch = {"text": "some thoughts on the week ahead"}, {"text": "launch day: we shipped the new agent today", "image": True}

    r = svc.analyze({"handle": "@User23", "versions": [plain, launch]})          # found in the corpus, followers taken from it
    assert r["account"]["history_posts"] == 40 and r["account"]["history_source"] == "our corpus of public posts" and r["account"]["followers"] == 46_000
    assert r["pick"]["version"] == 2 and r["versions"][1]["vs_version_1"]["status"].startswith("recommended")
    assert r["versions"][1]["likes_median"] > r["versions"][0]["likes_median"]
    media = {i["key"]: i for i in r["versions"][0]["checklist"]}["media"]
    assert media["ok"] is False and "2,247 pairs" in media["measured"]
    assert r["versions"][1]["type"]["type"] == "launch"                                # a launch post gets the launch checklist
    assert {i["key"]: i for i in r["versions"][1]["checklist"]}["video"]["text"] == "Has an image, not a video or GIF."

    r = svc.analyze({"handle": "user23", "versions": [launch, plain]})           # the better one first: the pick stays on it
    assert r["pick"]["version"] == 1

    past = [{"text": f"launch day: we shipped the new agent {i}" if i % 3 == 0 else f"some thoughts on the week {i}", "posted_at": f"2026-09-{1 + i:02d}T15:00:00Z",
             "likes": 400 if i % 3 == 0 else 60, "views": 16_000 if i % 3 == 0 else 2_400} for i in range(24)]
    r = svc.analyze({"handle": "newco", "followers": 9_000, "versions": [plain], "past_posts": past, "when": "2026-10-05T15:00:00Z"})
    assert r["account"]["history_posts"] == 24 and r["account"]["history_source"] == "posts you supplied" and r["pick"] is None and not r["account"]["new_account"]

    r = svc.analyze({"handle": "brandnew", "followers": 500, "versions": [plain]})  # no history at all
    assert r["account"]["new_account"] and r["versions"][0]["likes_median"] >= 0
    with pytest.raises(ValueError):
        svc.analyze({"handle": "brandnew", "versions": [plain]})                  # unknown account and no follower count

    assert post_type.classify("Introducing Acme Agents: build one in a minute")["type"] == "launch"
    assert post_type.classify("We raised a $12M Series A led by Example Ventures")["type"] == "funding"
    assert post_type.classify("We just crossed 10,000 users")["type"] == "milestone"
    assert post_type.classify("We're hiring a founding engineer")["type"] == "hiring"
    assert post_type.classify("You can now sign in with SSO on every plan")["type"] == "feature"
    assert post_type.classify("coffee was good today")["type"] == "other"
    tr = post_type.traits("Introducing Atlas 2: agents that file your taxes\n\nIt reads receipts.\nIt fills the forms. https://atlas.example/launch", [{"type": "video"}])
    assert tr["names_product_first_line"] and tr["lines3"] and tr["video"] and tr["ext_link"] and not tr["open_problem"]
    assert post_type.traits("Tired of slow builds?\nWe fixed it.")["open_problem"]

    r = svc.analyze({"handle": "user23", "versions": [{"text": "Tired of slow builds? We are launching today, link below https://example.com/x"}]})
    assert r["post_type"]["type"] == "launch" and "2,156 posts" in r["type_expectation"]["statement"]
    cl = {i["key"]: i for i in r["versions"][0]["checklist"]}
    assert not cl["names_product"]["ok"] and cl["names_product"]["grade"] == "A" and "1,672 pairs" in cl["names_product"]["measured"]
    assert not cl["announcement_first"]["ok"] and not cl["no_link"]["ok"] and not cl["video"]["ok"]
    assert cl["quiet_day"]["grade"] == "B" and "1,510 pairs" in cl["quiet_day"]["measured"] and cl["only_launch_this_week"]["ok"]

    client = TestClient(server.create_app(svc))
    assert "Launch advisor" in client.get("/").text and client.get("/api/health").json()["loaded"]
    res = client.post("/api/analyze", json={"handle": "user23", "versions": [plain, launch]})
    assert res.status_code == 200 and res.json()["pick"]["version"] == 2
    assert client.post("/api/analyze", json={"handle": "nobody", "versions": [plain]}).status_code == 422
    assert client.post("/api/analyze", json={"versions": []}).status_code == 422
