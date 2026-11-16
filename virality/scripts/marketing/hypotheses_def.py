"""Outside claims as testable hypotheses. Written before the outcome tables were read for the direction and the trait;
verdict_override is used only where the test lives in the launch track's results or the comparison is circular.

sources: ids of records in research_records.json (R: peer-reviewed review, X: platform code / statements / vendor studies,
PB: launch playbooks). source_grade is the best grade among them for THIS claim as it applies to a launch post on X.
direction: +1 the claim says posts WITH the trait do better, -1 worse.
"""
def h(id, claim, sources, grade, ai, trait=None, direction=1, frm="marketing", override=None, note=None):
    d = {"id": id, "claim": claim, "sources": sources, "source_grade": grade, "ai_startup_specific": ai, "trait": trait, "direction": direction, "from": frm}
    if override: d["verdict_override"] = override
    if note: d["note"] = note
    return d

HYPOTHESES = [
    # ---- calls to action
    h("H01", "A launch post should end with a call to action to use the product (try it, get started, sign up).", ["PB050", "PB072", "R:K07"], "anecdote", False, "cta_try", 1,
      note="Lee, Hosanagar and Nair (R:K07/R:B01, 106k Facebook brand posts) point the other way for overtly persuasive content."),
    h("H02", "Gating access behind a waitlist, invites or a research preview makes the launch post travel further (scarcity).", ["PB012", "PB036", "PB029", "PB046", "PB047", "R:P04"], "anecdote", True, "cta_waitlist", 1),
    h("H03", "A waitlist launch does better than an available-now launch.", ["PB012", "PB036"], "anecdote", True, "waitlist_vs_available_now", 1),
    h("H04", "Asking for a reply or repost in exchange for access ('comment X and I'll DM you', 'RT for an invite') lifts the post.", ["PB076", "X:S08", "X:C14"], "anecdote", False, "cta_engagement_bait", 1,
      note="X's code labels engagement bait as spam (no out-of-network distribution) and X said in 2026 that Grok pushes solicitation down; no effectiveness data exists outside either."),
    h("H05", "Explicitly asking readers to share or support ('please RT', 'would love your support') raises sharing.", ["R:W02", "R:K06"], "controlled study", False, "cta_any_ask", 1,
      note="Tan, Lee and Pang 2014: same author, same link, 11,404 pairs; 2014 Twitter."),
    # ---- offers
    h("H06", "Saying the product is free lifts the launch post (zero-price effect).", ["R:P03", "PB008"], "controlled study", False, "offer_free", 1, note="The controlled evidence is on purchase choice in lab experiments, not on social posts."),
    h("H07", "Open-source / open-weights releases travel further than closed launches.", ["PB006", "PB007", "PB008", "PB009", "PB010", "PB025", "PB030", "R:B08"], "anecdote", True, "offer_open_source", 1,
      note="R:B08 (controlled, 2,370 repositories) is about tweets raising GitHub stars, not about open-source posts getting more engagement."),
    h("H08", "Offering credits, a discount, a launch deal or a prize lifts the launch post.", ["PB022", "PB073", "R:B01"], "anecdote", False, "offer_credits_discount", 1,
      note="R:B01 (large observational, Facebook) found price and deal content LOWERS engagement; our result agrees with that study and contradicts the playbook."),
    h("H09", "Stating the price lowers engagement with a brand post.", ["R:B01"], "large observational", False, "price_stated", -1),
    # ---- proof
    h("H10", "Naming customers or users (social proof) lifts the launch post.", ["R:B07", "PB010", "PB028"], "large observational", False, "names_customer", 1, note="R:B07 is Kickstarter pitch wording (45k projects), not X."),
    h("H11", "Naming investors or an accelerator (backed by, YC) lifts the launch post.", ["PB079", "PB080", "PB055"], "anecdote", False, "backed_by", 1),
    h("H12", "A concrete number anywhere in the post (informativeness) makes it travel further.", ["R:W03", "R:P01", "PB060"], "controlled study", False, "number_anywhere", 1,
      note="Launch track: a number in the FIRST line passed at 55.8% (50.5 to 60.9), its weakest finding."),
    h("H13", "Leading with a benchmark result (state of the art, x times faster, a leaderboard rank) makes an AI launch travel.", ["PB005", "PB006", "PB037", "PB004"], "anecdote", True, "benchmark_claim", 1),
    h("H14", "A benchmark chart image does better than another image.", ["PB005", "PB006", "PB004"], "anecdote", True, "benchmark_with_image", 1, note="Text proxy: a benchmark claim in a post with an image. What the image shows is not read here (media track)."),
    h("H15", "Comparing against a named competitor or incumbent lifts the launch post.", ["R:P02", "PB037", "PB030", "PB006"], "controlled study", False, "compares_named", 1, note="R:P02 is a meta-analysis of comparative advertising, not social posts."),
    h("H16", "Pitching by analogy to a known product ('Cursor for X') lifts the post.", ["PB030"], "anecdote", False, "analogy_pitch", 1),
    h("H17", "A 'first / best / most capable' claim (the first AI software engineer, our most capable model) makes the launch travel.", ["PB027", "PB005"], "anecdote", True, "superlative_first", 1),
    # ---- voice
    h("H18", "Opening with stated excitement ('excited to announce') hurts; state the news instead.", ["PB072", "R:W05"], "anecdote", False, "open_excited", -1, note="R:W05 (controlled) says sentiment words help in general; the playbooks say this particular opener is wasted."),
    h("H19", "Telling the backstory (how long it took, why it was built) lifts the post (build in public).", ["PB060", "PB063", "R:B01"], "small observational", False, "backstory", 1),
    h("H20", "Addressing the reader directly (you / your) in the first line lifts the post.", ["R:K07"], "small observational", False, "addresses_you", 1),
    h("H21", "Pointing at a demonstration (demo, watch, see it in action) lifts the launch post.", ["PB027", "PB072", "PB036"], "anecdote", True, "demo_words", 1),
    h("H22", "A launch video that is framed as a demo does better than another launch video.", ["PB027", "PB072"], "anecdote", True, "demo_video", 1),
    h("H23", "Saying the product is available now (labs' availability line) lifts the post.", ["PB008", "PB014"], "anecdote", True, "availability_now", 1),
    h("H24", "Announcing something that is not yet available ('coming soon') does worse.", ["PB039", "PB042"], "anecdote", False, "coming_soon", -1),
    h("H25", "Celebration emoji (rocket, party, fire) lift the post.", ["R:B01"], "large observational", False, "rocket_emoji", 1),
    h("H26", "Mentioning the Product Hunt / Hacker News launch in the X post helps it.", ["PB057", "PB059", "PB020"], "anecdote", False, "product_hunt", 1),
    # ---- AI-specific media
    h("H27", "For a model release, a video of the model doing something beats no media.", ["PB027", "PB036", "PB041", "PB045"], "anecdote", True, "video_in_model_release", 1),
    h("H28", "For a model release, an image (typically a benchmark chart) beats no media.", ["PB005", "PB006", "R:M01"], "anecdote", True, "image_in_model_release", 1),
    h("H29", "Several images do better than one.", ["R:M01"], "large observational", False, "multi_image", 1),
    h("H30", "Within model releases, the open-weights ones travel further.", ["PB006", "PB008", "PB009"], "anecdote", True, "open_source_in_model_release", 1),
    h("H31", "Model releases are the launches that travel furthest for an AI account.", ["PB001", "PB006", "PB009"], "anecdote", True, "ai_model_release", 1, note="A topic, not something a writer chooses; reported so that the model can condition on it."),
    # ---- timing
    h("H32", "Launch on a weekday morning Pacific time (06:00 to 11:59 PT).", ["X:P08", "X:P10", "PB069", "PB001", "PB020"], "large observational", False, "pt_weekday_morning", 1, note="Vendor timing studies are pooled across accounts, in the poster's local time, and disagree with each other (X:P09, X:P11)."),
    h("H33", "The 8 to 10am Pacific weekday slot is best.", ["PB069", "PB070", "PB001"], "large observational", False, "pt_weekday_8_10", 1, note="PB070 (79 founder tweets) claims 3 to 5 times the median impressions for the Tuesday / Wednesday morning slot."),
    h("H34", "Tuesday to Thursday morning Pacific is best.", ["X:P08", "X:P09", "X:P10"], "large observational", False, "pt_tue_thu_morning", 1),
    h("H35", "Weekday morning beats weekday afternoon Pacific.", ["X:P08", "X:P09"], "large observational", False, "pt_morning_vs_afternoon", 1, note="Buffer says morning, Sprout says noon to 6pm."),
    h("H36", "Weekday morning beats evening and night Pacific.", ["X:P08", "X:P11"], "large observational", False, "pt_morning_vs_evening_night", 1, note="Metricool says 9pm is best."),
    h("H37", "Tuesday is the best launch day.", ["X:P08", "PB058", "PB070"], "large observational", False, "pt_tuesday", 1),
    h("H38", "Weekend launches do worse.", ["X:P08", "X:P09", "PB070"], "large observational", False, "pt_weekend", -1),
    # ---- before and after
    h("H39", "A teaser in the days before ('tomorrow', 'something is coming') lifts the launch post.", ["X:S12", "PB001", "PB011", "PB042", "PB081"], "anecdote", True, "teaser_72h", 1),
    h("H40", "A teaser the day before lifts the launch post.", ["PB001", "PB081"], "anecdote", True, "teaser_36h", 1),
    h("H41", "Do not post in the 24 hours before the launch (an account's posts compete with each other).", ["X:C04", "X:L02", "R:T01"], "platform code", False, "quiet_prev24h", 1, note="The code decays an author's 2nd and 3rd post within one feed request; the launch track found the 2-hour version (grade C)."),
    h("H42", "Launch repeatedly / run a launch week: several launch posts within days of each other each do well.", ["PB019", "PB021", "PB048", "PB023"], "anecdote", False, "launch_in_prev_week", 1,
      note="The playbook claim is about total reach over the week; what we can test is each post against the account's other launch posts."),
    h("H43", "Follow up in the days after by quote-posting your own launch (a milestone, '24 hours later').", ["PB082", "PB018", "PB012", "X:S12"], "anecdote", False, "self_quote_day2_7", 1, note="Circular: launches that did well get followed up. See _context_extra for how the follow-up posts themselves do."),
    h("H44", "Follow-up announcements in the days after help the launch.", ["PB082", "X:S12"], "anecdote", False, "followup_announce_day2_4", 1),
    h("H45", "Keep posting in the days after the launch.", ["PB082", "X:P07"], "anecdote", False, "active_day2_4", 1),
    # ---- amplification
    h("H46", "Other accounts quote-posting within the first hour lifts the launch (early engagement compounds).", ["R:C07", "R:C08", "R:T01", "PB020", "PB028", "PB073", "R:B08"], "controlled study", True, "quoted_1h", 1,
      note="The controlled evidence (random early upvotes, donations) is from other platforms. In our corpus the comparison is circular: launches that are taking off get quoted sooner."),
    h("H47", "Employees and founders quote-posting the company's launch within the first hour lifts the launch.", ["PB020", "PB028", "PB015", "PB079"], "anecdote", True, "team_quoted_1h", 1),
    h("H48", "When the team does quote-post, doing it within the first hour beats doing it later the same day.", ["PB020", "R:T01"], "anecdote", True, "team_quoted_1h_vs_later", 1,
      override=None, note="The least circular version available: both sides were quoted by the team. Still partly circular and thin."),
    h("H49", "A quote by another account within the first hour beats a first quote that comes later.", ["R:T01", "R:C05"], "large observational", False, "quoted_1h_vs_later", 1),
    # ---- claims already tested by the launch track (scripts/launch), restated against their outside sources
    h("H50", "A link in the post body costs reach.", ["R:K01", "R:K03", "X:P04", "X:P17", "X:S01", "X:S02"], "large observational", False, "no_link_in_post", 1, "launch", "confirmed",
      "Launch posts without an external link won 58.2%. The size (1.29x likes) is far below the 4 to 8x views gap of R:K01, and X's product head denies any penalty (X:S03, X:S04); the code has none (X:C15)."),
    h("H51", "Put the link in a reply instead.", ["X:S02", "X:P23"], "platform statement", False, "points_to_reply_vs_link_in_post", 1, "launch", "untestable", "17 pairs; replies are not in the corpus."),
    h("H52", "A launch needs a video.", ["PB072", "PB027", "PB050", "X:S13", "X:S14"], "anecdote", True, "video_vs_none", 1, "launch", "confirmed", "Video against no media 65.4%, against an image 62.8% (grade B; weaker after April 2026)."),
    h("H53", "Text posts beat video posts on X (2025).", ["X:P05", "X:P03", "PB071", "PB060"], "large observational", False, "video_vs_none", -1, "launch", "contradicted",
      "For launch posts inside the same account, video wins 65.4% against no media. Buffer's result is engagement per impression pooled across accounts; ours is likes against the account's own usual."),
    h("H54", "Adding an image raises engagement.", ["R:M01", "X:P16", "X:L01"], "large observational", False, "image_vs_none", 1, "launch", "not_confirmed", "59.5% in pairs but likes ratio 1.17x (0.95 to 1.45) and 48.0% among clear launch posts."),
    h("H55", "Write the launch as a thread.", ["X:P21", "X:P20", "PB015"], "small observational", False, "thread_marker", 1, "launch", "not_confirmed", "Only the head post's thread marker is visible: 51.2% (45.5 to 56.9). The thread body is not in the corpus."),
    h("H56", "Hashtags help (1 to 2 hashtags, +21%).", ["X:P22", "R:K04"], "anecdote", False, "no_hashtag", -1, "launch", "untestable", "41 launch posts with a hashtag; the version without won 60.2% (43.4 to 76.6). X's code has no hashtag term and X's owner told users to stop (X:S10)."),
    h("H57", "Tagging other accounts lowers sharing.", ["R:W04"], "controlled study", False, "no_tags", 1, "launch", "not_confirmed", "Same direction as Tan, Lee and Pang: no tags won 59.1% (52.9 to 64.4) but the likes ratio 1.18x (0.99 to 1.41) includes no effect."),
    h("H58", "Open the launch on the pain point, not on the product ('hook first, product second').", ["PB072", "PB050", "PB027"], "anecdote", False, "announcement_not_problem_first", -1, "launch", "contradicted",
      "The playbooks say this about the launch VIDEO. For the post TEXT the opposite holds: opening with the announcement won 72.5% (59.5 to 81.8) over opening with a problem or question, and naming the product in the first line won 64.0% (grade A)."),
    h("H59", "Informative, specific first lines win (proper nouns, numbers).", ["R:W03"], "controlled study", False, "names_product_first_line", 1, "launch", "confirmed", "First line names the product: 64.0% (59.0 to 69.0), confirmed before and after April 2026."),
    h("H60", "Shorter is better (trim 10 to 20%).", ["R:W07"], "controlled study", False, "long280", -1, "launch", "not_confirmed", "Length shows no effect either way (280+ characters: adjusted 0.95x); layout on 3+ lines helps (59.2%)."),
    h("H61", "Posts close together from one account suppress each other.", ["X:C04", "X:L02"], "platform code", False, "two_hours_after_previous", 1, "launch", "confirmed", "Launch within 2 hours of the previous post: 0.71x (0.58 to 0.87) pooled over all accounts (grade C)."),
    h("H62", "Quote-posting your own announcement instead of posting it as an original costs reach.", ["X:C05", "X:C06"], "platform code", False, "original_not_quote", 1, "launch", "confirmed", "Original against quote: 72.1% (60.7 to 81.1), grade B."),
    # ---- cannot be tested on this corpus
    h("H63", "Launch video length of 45 to 90 seconds is best; over 10 seconds is needed for the video-quality-view action.", ["PB072", "PB050", "X:C09"], "anecdote", True, override="untestable", note="duration_ms is empty for all 852 video launch posts in the corpus. The code's video weights are all zero in any case (X:C09)."),
    h("H64", "Replies from well-known accounts and replying to commenters lift the post.", ["PB016", "X:P06", "X:C03", "X:C22"], "large observational", True, override="untestable", note="Replies are not in the corpus."),
    h("H65", "Post the launch from the founder's account, not the company's.", ["PB078", "PB017", "PB032"], "anecdote", False, override="untestable",
      note="Team membership is not observable. What the launch track could measure points the other way: launch posts get 2.2x the usual on company accounts and 2.3x on founder accounts, and a team member's first-person quote got more likes than the company's own post in 10% of 152 cases."),
    h("H66", "Paid amplification (clippers, creator blocks, ads) is what produces the large view counts.", ["PB034", "PB035", "PB073", "PB074", "PB075"], "anecdote", False, override="untestable", note="Paid and organic views are not separable in public counts, and views exist for only 205 launch posts here."),
    h("H67", "Controversy / rage bait is distribution.", ["PB032", "PB033", "PB040", "R:E04", "R:C11"], "large observational", False, override="untestable", note="No reliable label for provocation; negative reactions are not observable (replies, mutes, blocks), and the code weights those at -31 to -234 (X:C02)."),
    h("H68", "Images with a human face lift engagement (38 to 291% on Twitter).", ["R:M01"], "large observational", False, override="untestable", note="Needs image content; left to the media track."),
    h("H69", "Premium accounts get about 10x reach.", ["X:P01", "R:C13", "X:C10"], "large observational", False, override="untestable", note="Cannot be measured inside an account; outside sources conflict (Buffer 10x pooled; a 2023 feed audit found no boost; no multiplier in the 2026 code)."),
    h("H70", "A curator or well-known account sharing the work is the largest lever for AI work (2 to 3x citations).", ["R:B11", "PB006", "PB055"], "small observational", True, override="untestable", note="Reposts are not in the corpus; quote posts by large accounts are circular (see H46)."),
]

# Traits the model should receive as inputs. tier 1 = a measured effect in this corpus (launch or marketing track);
# tier 2 = needed to condition on (context, not a lever); tier 3 = outside evidence is strong enough to carry the trait
# although this corpus cannot measure it yet. "source" says where it is computed.
MODEL_TRAITS = {
 "_note": "Text traits are the regular expressions in scripts/marketing/traits.py (TEXT_TRAITS) and scripts/launch/build_table.py; they run on a draft with no outcome data. Give them only for posts the launch labeller types as an own announcement; elsewhere they are untested.",
 "tier_1_measured_effect": [
  {"name": "post_type", "values": "launch, open_source, feature, funding, milestone, hiring, other", "why": "launch posts get 2.2x the account's usual and the model predicts 1.03x (launch track)", "source": "xviral/text/launch_labels.py"},
  {"name": "names_product_first_line", "why": "64.0% (59.0-69.0), grade A, launch track", "source": "scripts/launch/build_table.py"},
  {"name": "lines3", "why": "59.2%, grade A-, launch track", "source": "scripts/launch/build_table.py"},
  {"name": "media_kind", "values": "none, image, video", "why": "video 65.4% against none, grade B, launch track; strongest for model releases (65.3%)", "source": "media field"},
  {"name": "open_problem", "why": "announcement first 72.5%, grade B, launch track; contradicts the 'pain point first' playbook", "source": "scripts/launch/build_table.py"},
  {"name": "is_quote", "why": "original 72.1%, grade B, launch track", "source": "kind"},
  {"name": "link_in_post", "why": "no link 58.2%, grade B, launch track; agrees with the outside link-penalty studies in direction, not in size", "source": "urls"},
  {"name": "open_number", "why": "55.8%, grade B weakest, launch track", "source": "scripts/launch/build_table.py"},
  {"name": "offer_open_source", "why": "this track: open source / open weights wording, see marketing_evidence.json", "source": "scripts/marketing/traits.py"},
  {"name": "superlative_first", "why": "this track: first / best / most capable claim, see marketing_evidence.json", "source": "scripts/marketing/traits.py"},
  {"name": "offer_credits_discount", "why": "this track: negative, pooled over all accounts (grade C)", "source": "scripts/marketing/traits.py"},
  {"name": "hours_since_previous_own_post", "why": "under 2 h: 0.71x (launch track, C); nothing in the previous 24 h: this track", "source": "account history"},
  {"name": "own_launch_posts_in_previous_7_days", "why": "this track: a launch that follows another within a week does worse", "source": "account history + post_type"},
 ],
 "tier_2_context_not_a_lever": [
  {"name": "ai_model_release", "why": "model releases do 1.2 to 1.5x the account's other launches; a topic the writer does not choose", "source": "scripts/marketing/traits.py"},
  {"name": "account_kind", "values": "company, founder, other", "why": "feature releases and hiring posts behave differently on company and founder accounts (launch track)", "source": "scripts/launch/accounts.py"},
  {"name": "follower_tier", "why": "video against none is 74.9% under 100k followers and 65.2% above (launch track)", "source": "author_followers"},
  {"name": "period", "why": "video and several B findings are weaker after April 2026; the platform changed ranking systems in 2026", "source": "created_at"},
 ],
 "tier_3_outside_evidence_only": [
  {"name": "first_hour_quotes_by_other_accounts (count, same-team count, followers of quoters)", "why": "controlled evidence elsewhere that early support compounds (R:C07, R:C08); here 69.2% and a dose-response, but circular. Useful as a post-launch signal, never as advice", "source": "quote graph"},
  {"name": "video_duration_seconds", "why": "playbooks and the code's 10-second threshold; empty in this corpus, to be filled by the media track", "source": "media.duration_ms"},
  {"name": "image_content (benchmark chart, product screenshot, face)", "why": "Li and Xie 2020 on faces; AI playbooks on benchmark charts; text proxy showed nothing (H14)", "source": "media track"},
  {"name": "link_in_first_reply", "why": "platform statements in both directions; replies not collected", "source": "replies"},
  {"name": "engagement_bait_request", "why": "X's code and 2026 statements penalise it; 10 launch posts here", "source": "scripts/marketing/traits.py"},
 ],
 "do_not_add": ["posting hour or weekday in Pacific time (no effect in any of seven comparisons)", "call to action to try the product", "free", "waitlist", "benchmark claim in the text", "number anywhere in the text", "teaser before the launch", "celebration emoji",
                "you / your in the first line", "availability line", "demo wording"],
}
