"""Text traits that turn outside marketing claims into measurable properties of a post. Shared by the marketing-track
scripts; written so the product can compute the same traits on a draft (plain regular expressions, no outcome used).

All patterns run on the text with links removed, lower-cased, curly apostrophes straightened.
"""
import polars as pl

NUMU = r"\d[\d,.]*\s?(k|m|mm|b|bn|x|%|million|billion|ms|s|sec|seconds|minutes|mins|hours|hrs|days|weeks|months|years|tokens|params|parameters|gb|mb|fps|loc|lines|stars|users|customers|teams|companies|developers|downloads|languages|models|countries)\b"
# named competitors / incumbents a launch is compared with (the launching account's own name is not excluded: a self-mention
# is caught as "comparison" only together with a comparison word, see compares_named)
RIVALS = (r"(openai|chatgpt|gpt-?[345][\w.-]*|o[134](-mini|-pro)?|claude|anthropic|gemini|google|deepmind|llama|meta|mistral|deepseek|grok|copilot|cursor|windsurf|devin|replit|lovable|bolt|v0|perplexity|midjourney|dall-?e|sora|runway|pika|"
          r"elevenlabs|notion|figma|slack|zoom|excel|salesforce|zapier|stripe|shopify|canva|photoshop|adobe|github|vercel|aws|azure|nvidia|apple|siri|alexa|microsoft|docusign|jira|linear|airtable|superhuman|gmail|chrome|whisper|stable diffusion|langchain|hugging ?face)")
CMP = r"(\bvs\.?\b|\bversus\b|\bcompared (to|with)\b|\b(better|faster|cheaper|smaller|more accurate|more capable) than\b|\boutperform(s|ed|ing)?\b|\bbeat(s|ing)?\b|\bsurpass(es|ed|ing)?\b|\b(alternative|rival|competitor|answer) to\b|\bunlike\b|\bkiller\b|\breplac(e|es|ing) (your )?|\bswitch(ing)? from\b|\bmigrat(e|ing) from\b|\bon par with\b|\bmatch(es|ing)? \b)"

TEXT_TRAITS = {
    # --- calls to action
    "cta_try": ("an imperative call to action to use the product (try it, get started, sign up, download, install, check it out, start building)",
                r"(\btry (it|it out|it now|it free|it today|it here|it yourself|now|today|out)\b|\bget started\b|\bsign up\b|\bdownload (it|now|here|today|the app)\b|\binstall (it|now|with)\b|\bcheck it out\b|\bstart (building|using|creating)\b|\bgive it a (try|spin|go)\b|\bplay with it\b|\bgo (try|build|play)\b|\btry [\w.-]+ (now|today|free|here|for free)\b|\bavailable (now|today) (at|on|in)\b|\bget (it|yours|access) (now|here|today)\b|\bbook a demo\b|\bapply (now|here|today)\b|\bjoin (us|now|today|here)\b|\bread (more|the (blog|post|paper|docs))\b|\blearn more\b|👉|➡️)"),
    "cta_waitlist": ("the post points to a waitlist, early access, an invite or a request-access form",
                     r"(\bwait ?list\b|\bearly access\b|\brequest access\b|\binvite[- ]only\b|\binvite codes?\b|\binvites?\b.{0,20}\b(rolling|going out|available|left)\b|\bprivate (beta|preview|alpha)\b|\bclosed (beta|alpha)\b|\bsign up for (the )?(beta|access)\b|\bapply for (early )?access\b|\bjoin the (beta|waitlist|wait list)\b|\b(limited|research) preview\b)"),
    "cta_engagement_bait": ("the post asks for a reply, repost, like or follow in exchange for something (comment X and I'll send it, RT for access)",
                            r"(\b(comment|reply|drop|type)\b.{0,40}\b(and|&) (i'll|we'll|i will|we will|get|you'll)\b|\b(comment|reply) (with )?[\"“'‘]?\w+[\"”'’]? (below |and |to |for )|\b(rt|retweet|repost)\b.{0,30}\b(to|for|and) (win|get|enter|access|a chance|i'll|we'll)|\b(like|follow)\b.{0,6}(\+|&|and)\s?(rt|retweet|repost|comment|reply|follow)\b|\bdm (me|us)\b.{0,30}\b(for|to get|and i'll)|\bi'll dm you\b|\bwe'll dm you\b|\b(reply|comment) for (access|an invite|the link|a code|early access)\b|\bgiving away\b|\bgiveaway\b)"),
    "cta_any_ask": ("the post asks the reader for support (repost, share, upvote, feedback, follow)",
                    r"(\b(please|pls|plz)\b.{0,20}\b(rt|retweet|repost|share|upvote|support)\b|\b(rts?|retweets?|reposts?|shares?) (are |is |much |greatly |very )*(appreciated|welcome)\b|\bwould (love|appreciate) (your|any|some) (support|feedback|thoughts|upvote)|\bsupport us\b|\bshow (us )?some love\b|\bupvote\b|\bfeedback (is )?(welcome|appreciated)\b|\blet (me|us) know what you think\b|\bwould love (your|to hear)\b|\bhelp us spread\b|\bspread the word\b)"),
    # --- offers
    "offer_free": ("the post says the product is free (free, for free, free tier, no credit card, $0)",
                   r"(\bfor free\b|\bfree (to (use|try|start)|tier|plan|trial|forever|for (everyone|all|students|teams|individuals|personal|open)|access|beta|credits?|today|now|during)\b|\b(is|it's|its|are|completely|totally|100%|entirely) free\b|\bno credit card\b|\$0\b|\bfree and open\b|\bfree\b[.!]?$|\btry (it )?free\b|\bfree:)"),
    "offer_open_source": ("the post says the product is open source or open weights (open source, open weights, MIT / Apache licence, on GitHub / Hugging Face)",
                          r"(\bopen[- ]?sourc(e|ed|ing)\b|\bopen[- ]weights?\b|\b(mit|apache( 2(\.0)?)?|gpl|bsd)[- ]licen[cs]ed?\b|\b(mit|apache 2\.0) licen[cs]e\b|\bweights (are|is) (now )?(available|out|released|public|up|open|on)\b|\b(code|repo|weights|models?|dataset) (is|are)? ?(now )?(available |live |out |up )?on (github|hugging ?face|hf)\b|\bfully open\b|\boss\b|\bself[- ]host(ed|able)?\b|\bfork it\b|\bstar (it|us|the repo) on github\b)"),
    "offer_credits_discount": ("the post offers credits, a discount, a deal or a prize",
                               r"(\b(free )?credits\b|\b\d+% off\b|\bdiscount\b|\bpromo code\b|\bcoupon\b|\blifetime deal\b|\blaunch (price|pricing|special|offer|discount)\b|\bgiving away\b|\bgiveaway\b|\$\d[\d,.]*k? (in|of|worth of) (credits|prizes)|\b(1|one|2|two|3|three|6|six|12) (month|months|year)s? free\b|\bfree for (a|the first|one|1|3|6|12) (year|month|months|week))"),
    "price_stated": ("the post states a price in currency ($20/month, $0.50 per million tokens)",
                     r"[$€£]\s?\d[\d,.]*\s?(/|per |a )\s?(mo|month|year|yr|user|seat|m\b|1m|million|token|hour|hr|gb|image|minute|min|request|call)"),
    # --- proof
    "names_customer": ("the post names users or customers as proof (used by, trusted by, customers like, teams at, in production at)",
                       r"(\b(used|trusted|loved|adopted|relied on|powered|deployed|built) by (teams|engineers|developers|companies|people|researchers|folks|thousands|millions|hundreds|[\d,.]+[km+]?|the team|teams at|@\w+)|\bcustomers (like|such as|including)\b|\b(already )?(in production|live|deployed|running) (at|with|inside|across)\b|\bpowering\b.{0,30}@\w+|\bcompanies like\b|\b(early|design|launch) (customers|partners|users) (like|include|including|such as)\b|\bfortune 500\b|\b(case study|testimonial)\b|\bworking with @\w+)"),
    "backed_by": ("the post names investors or an accelerator (backed by, led by, YC batch, a16z, Sequoia)",
                  r"(\bbacked by\b|\bled by\b|\bfunded by\b|\b(yc|y combinator)\b|\(yc [wsfx]\d\d\)|\b[wsf]\d\d batch\b|\ba16z\b|\bandreessen\b|\bsequoia\b|\bfounders fund\b|\bkhosla\b|\bbenchmark capital\b|\bgreylock\b|\blightspeed\b|\baccel\b|\bindex ventures\b|\bgeneral catalyst\b|\bthrive capital\b|\bangels? (like|including)\b|\binvestors (like|including)\b|\bseed round\b|\bseries [a-e]\b)"),
    "number_anywhere": ("the post contains a quantified claim (a number with a unit, a percentage, a multiple or a money amount) anywhere in the text",
                        r"((^|[\s(~<>≈#])" + NUMU + r"|[$€£]\s?\d|\b\d+(\.\d+)?x\b|\d%)"),
    "benchmark_claim": ("the post cites a benchmark or a measured comparison anywhere (benchmark names, state of the art, x times faster, scores, leaderboard)",
                        r"(\bsota\b|\bstate[- ]of[- ]the[- ]art\b|\bbenchmarks?\b|\bleaderboards?\b|\b(swe-?bench|mmlu|humaneval|gpqa|arc-?agi|lmarena|lm arena|livebench|webarena|osworld|terminal-?bench|gaia|imagenet|mteb|elo|aider|frontiermath|simpleqa|hle|artificial analysis|evals?)\b|\b\d+(\.\d+)?x (faster|cheaper|smaller|better|more|fewer|less|lower|higher|the)\b|\boutperform(s|ed|ing)?\b|\b#1 (on|in)\b|\b(top|first|1st) (on|of) (the )?\w+ (leaderboard|arena|benchmark)|\bscor(es|ed|ing) \d|\bbeats\b|\b\d+(\.\d+)?% (on|accuracy|faster|cheaper|better|fewer|less|lower|higher|more|reduction|improvement)\b|\brecord\b|\bbest[- ]in[- ]class\b)"),
    "compares_named": ("the post compares the product with a named competitor or incumbent (vs, better than, alternative to, outperforms + a known product name; or 'the X for Y' / 'X killer')",
                       r"(" + CMP + r".{0,40}\b" + RIVALS + r"\b|\b" + RIVALS + r"\b.{0,25}" + CMP + r"|\b" + RIVALS + r"[- ]killer\b|\b(open[- ]source|free|local|private|cheaper|faster|better|oss|self-hosted) (alternative|version|clone|answer) (to|of)\b)"),
    "analogy_pitch": ("the pitch is an analogy to a known product ('Cursor for X', 'the Stripe of Y', 'like ChatGPT but')",
                      r"(\b(the |a |an |like |think )" + RIVALS + r" (for|of) \w+|\b" + RIVALS + r" for (every|your|the|all|any|\w+ers|\w+s)\b|\blike " + RIVALS + r",? but\b|\b" + RIVALS + r",? but (for|with|on|in)\b|\bthink (of it as )?" + RIVALS + r"\b)"),
    "superlative_first": ("the post claims to be the first, the best or the most of something (world's first, the first, the fastest, most powerful, best)",
                          r"(\bworld'?s (first|fastest|best|most|smallest|largest)\b|\bthe first (ever |truly |real |fully |open|ai|\w+ )?\w+|\bfirst[- ]ever\b|\b(the|our) (most (powerful|capable|advanced|intelligent|accurate)|fastest|smartest|best|largest|biggest|smallest|cheapest)\b|\bnew state of the art\b|\bgame[- ]chang(er|ing)\b|\brevolution(ary|iz)|\bbreakthrough\b|\bnever (been )?(done|seen|possible) before\b)"),
    # --- voice and framing
    "open_excited": ("the first line opens with stated excitement (excited / thrilled / proud / happy to announce or share)",
                     None),   # first-line pattern, set in add_text_traits
    "backstory": ("the post tells how long or why it was built (after N months, we spent a year, I quit my job, I built this because)",
                  r"(\b(after|for|over|spent|took|been) (the (last|past) )?(\d+|a|two|three|four|five|six|several|many|a few|a couple of) (week|month|year|day)s?\b.{0,40}\b(build|built|building|work|worked|working|research|develop|making|hack|grind|stealth)|\b(week|month|year)s (of|in) (work|building|research|development|stealth|the making)\b|\bi (quit|left) my (job|phd)\b|\bi built (this|it)\b|\bwe built (this|it) because\b|\bout of stealth\b|\bstarted as\b|\bside project\b)"),
    "addresses_you": ("the first line addresses the reader (you, your)", None),
    "demo_words": ("the post points at a demonstration (demo, watch, see it in action, here's how it works)",
                   r"(\bdemo\b|\bwatch\b|\bin action\b|\bhere'?s how it works\b|\bwalkthrough\b|\bsee (it|how|what)\b|\bhere'?s (a|what|how)\b|\bsound on\b|🔊|🎥|📹|\bvideo\b|👇)"),
    "availability_now": ("the post says the product can be used now (available now / today, live now, out now, rolling out today, now in the API)",
                         r"(\bavailable (now|today|to (all|everyone))\b|\b(is|are|it's) (now )?(live|out|here|available)\b|\bout now\b|\blive (now|today)\b|\brolling out (now|today|to)\b|\bnow (available|live|in (the )?(api|public beta|beta|preview|ga)|generally available|open to)\b|\bstarting today\b|\btoday\b)"),
    "coming_soon": ("the post says the product is not yet available (coming soon, launching soon, next week, stay tuned)",
                    r"(\bcoming soon\b|\blaunching (soon|next|tomorrow|on (mon|tue|wed|thu|fri|sat|sun))|\bstay tuned\b|\bsoon\b[.!]?\s*$|\b(launching|available|shipping|releasing|opening) tomorrow\b)"),
    "rocket_emoji": ("the post has a rocket or party emoji (🚀 🎉 🔥)", r"[🚀🎉🔥🥳]"),
    "product_hunt": ("the post says the product is on Product Hunt, Hacker News or Show HN", r"(\bproduct ?hunt\b|\b(live|launching|launched|up) on ph\b|\bhacker ?news\b|\bshow hn\b|\b#?producthunt\b)"),
    "ai_model_release": ("the post announces a model (model, weights, checkpoint, parameters, context window, fine-tune, LLM)",
                         r"(\b(new |our |a |the )?(foundation |language |video |image |speech |voice |embedding |reasoning |open |frontier |diffusion |text-to-\w+ )?models?\b|\bweights\b|\bcheckpoints?\b|\b\d+(\.\d+)?[bm] param|\bparameters\b|\bcontext window\b|\bfine-?tun(e|ed|ing)\b|\bllms?\b|\binference\b|\btokens?\b)"),
}
FIRST_LINE = {
    "open_excited": r"\b(excited|thrilled|proud|happy|delighted|pumped|stoked|honored|humbled|psyched) to\b|\b(super|so|very|beyond|incredibly) (excited|thrilled|proud|happy)\b|\bwe'?re excited\b|\bi'?m excited\b",
    "addresses_you": r"\b(you|your|you're|you'll|yourself)\b",
}
# the account's own posts in the days before a launch that tease it
TEASER = (r"(\btomorrow\b|\bsomething (big|new|special|exciting|is coming|cool)\b|\bstay tuned\b|\bcoming (soon|tomorrow|this week|next week|on (mon|tue|wed|thu|fri))|\b(launching|shipping|dropping|releasing|announcing|going live) (soon|tomorrow|this week|next week|on (mon|tue|wed|thu|fri)|in \d+ (hours|days|hrs))"
          r"|\b(big|huge|exciting|major) (news|announcement|launch|update|things?|release) (coming|tomorrow|soon|this week|next week|on the way|ahead)\b|\b(\d+|one|two|three) (more )?(days?|sleeps?|hours?) (to go|left|until|till)\b|\bsneak peek\b|\bteaser\b|\bcan't wait to (show|share)\b"
          r"|\bsee you (tomorrow|at \d|on (mon|tue|wed|thu|fri))|\btune in\b|\bmark your calendars?\b|\bsave the date\b|\b(any guesses|guess what'?s coming)\b|\bget ready\b|\bsoon[.!™]*\s*(👀)?\s*$|^\W*👀\W*$|\blaunch (day|week) (is )?(tomorrow|coming|starts)\b|\bwe have something\b|\bcooking\b)")


def add_text_traits(d: pl.DataFrame) -> pl.DataFrame:
    raw = pl.col("text").fill_null("").str.replace_all(r"https?://\S+", " ")
    lc = raw.str.to_lowercase().str.replace_all("’", "'").str.replace_all(r"[ \t]+", " ")
    first = raw.str.strip_chars().str.split("\n").list.first().fill_null("").str.to_lowercase().str.replace_all("’", "'")
    d = d.with_columns(lc.alias("_lcm"), first.alias("_first"))
    d = d.with_columns([pl.col("_lcm").str.contains("(?m)" + pat).alias(k) for k, (_, pat) in TEXT_TRAITS.items() if pat is not None] +
                       [pl.col("_first").str.contains(pat).alias(k) for k, pat in FIRST_LINE.items()])
    return d.drop("_lcm", "_first")
