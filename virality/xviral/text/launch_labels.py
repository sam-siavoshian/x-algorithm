"""Rules that label startup/company post types. Shared by every launch-track script.

A post is typed from its opening only (the first 300 characters, links removed, lower-cased): an announcement is made at
the top of a post, and matching anywhere in long posts labels essays and reprinted articles. First match wins, in the
order of ORDER. A typed post counts as the account's *own* announcement only when the opening speaks in the first person
(we / our / I / my) or, for a company account, opens with an announcement word.
"""
import polars as pl

MONEY = r"[$€£]\s?\d[\d,.]*\s?(k|m|mm|b|bn|million|billion)?\b"
NUM = r"\d[\d,.]*\s?(k|m|mm|b|million|billion)?\+?"
UNITS = r"(users|customers|downloads|stars|github stars|signups|sign-ups|developers|devs|installs|subscribers|teams|companies|members|paying customers|paid users|orders|merchants|creators|builders|apps|agents|requests|tokens)"
FP = r"\b(we|we're|we've|we are|we have|our|i|i'm|i've|my)\b"
EXC = r"(excited|thrilled|proud|happy|pumped|stoked|delighted)"

P = {
 "funding": r"(\b(we|i)('ve| have)? (just |recently |officially )?(raised|closed|secured)\b.{0,60}" + MONEY + r"|\b(we|i)('ve| have)? (just |recently |officially )?(raised|closed) (a|an|our|the) (\w+ ){0,3}(round|seed|series [a-e]\b)"
            r"|\b(our|my)( new| latest)? (" + MONEY + r" )?(seed|pre-seed|series [a-e]\b|funding round|fundraise|round)\b.{0,80}(" + MONEY + r"|led by|raise|raised|closed|announc)"
            r"|\b(announcing|announce) (our|a|the|my) (" + MONEY + r" )?(seed|pre-seed|series [a-e]\b|funding|fundraise|raise)|\bhas (raised|closed|secured) (a |an )?" + MONEY + r"|\braised " + MONEY + r".{0,80}\b(led by|from)\b)",
 "hiring": r"(\b(we're|we are|i'm|i am) (now |also |actively |still )?hiring\b|\bis (now |also |actively )?hiring\b|\bhiring (a|an|our|for|across|engineers|designers|founding|our first)\b|\bjoin (our|my) team\b|\b(we have|we've got|our) open (roles|positions)\b"
           r"|\bcome (work|build) with (us|me)\b|\b(we're|we are|i'm|i am) looking for (a|an|our|my) (first |founding |senior |staff |lead |next )?(\w+ ){0,2}(engineer|designer|developer|researcher|marketer|pm|product manager|intern|head of|chief of staff|account executive|recruiter|cto|specialist|lead)\b|\bwe're expanding our team\b)",
 "launch": r"(\b(introducing|announcing|unveiling)\b|^\W*(meet|say hello to|say hi to|presenting|launching|launched|launch day|new launch|now live|it's live|it's here)\b"
           r"|\b(we|i)('re| are|'m| am|'ve| have)? (so |super |very |beyond |incredibly )?(" + EXC + r" )?(to )?(finally |officially |just |now )?(launch|launching|launched|introduce|unveil|unveiling|announce|release|releasing|released)\b"
           r"|\b" + EXC + r" to (finally |officially )?(announce|launch|introduce|unveil|release|share that|share our)\b|\b(just|officially|finally) (launched|launching|released)\b|\blaunch(es|ing|ed)? today\b|\btoday,? (we|i)('re| are|'m| am)? (launch|releas|introduc|announc|unveil|open)"
           r"|\bis (now|finally|officially) (live|here|out|launching|available|public|open|generally available|ga)\b|\bare (now|finally|officially) (live|here|out|available)\b|\b(live|launching|launched) on (product ?hunt|ph|hacker news|show hn)\b"
           r"|\bout now\b|\bavailable (now|today)\b|\bnow (generally available|in (public|open|private) (beta|preview|alpha)|open to everyone|open to all)\b|\b(public|open|private) beta (is|starts|opens)\b|\b(waitlist|early access) (is )?(now )?(open|live)\b)",
 "open_source": r"(\b(we're|we are|i'm|i am|we|i|we've|i've|just|today|now|officially|fully) (are |have |just |now |also |fully |officially )?open[- ]?sourc(ed|ing)\b|\bopen[- ]?sourc(ing|ed) (our|my|it|this)\b|\bis (now |officially |fully )open[- ]?source\b|\bnow open[- ]?source\b"
                r"|\b(our|my|the) (code|weights|repo|model|models|dataset)( and \w+)? (is|are) (now )?(available|live|out|up|open|public|released) on (github|hugging ?face|hf)\b)",
 "feature": r"(^\W*(shipped|just shipped|changelog|release notes|what's new|whats new|new feature|new features|new this week|shipped this week|weekly drop|product update|new in [\w@. ]{2,25})\b.{0,15}[:\-–—!]|\bnow supports?\b|\byou can now\b|\bnow you can\b"
            r"|\b(we|i)('ve| have)? (just |recently |also )?(shipped|added|rolled out|pushed|redesigned|rebuilt|upgraded)\b|\bjust shipped\b|\b(we're|we are|is|are|i'm) (now )?rolling out\b|\brolling out (to|today|now)\b|\b(is|are) now available (in|on|for|to|via|with|through)\b|\bnow available (in|on|for|to|via)\b"
            r"|\bchangelog\b|\brelease notes\b|\bnew (feature|features)\b|\bv?\d+\.\d+(\.\d+)?( beta \d+)? (is|has) (now )?(out|live|here|available|released|landed|shipped)|\b(released|releasing|shipping|shipped) v?\d+\.\d+|\badded support for\b|\b(we|i) now (support|have|offer)\b|\bnow (works|integrates) with\b)",
 "milestone": r"((^\W*|\b(we|i|we've|i've|we're|our \w+|my \w+|[\w.]+ (just|has|have|now|officially)) (just |have |has |now |officially |recently )*)(hit|crossed|passed|reached|surpassed|grew to|went from)\b.{0,30}(\b" + NUM + r" " + UNITS + r"\b|" + MONEY + r"\s?(in )?(arr|mrr|revenue|run[- ]rate|gmv|sales))"
              r"|^\W*" + NUM + r" " + UNITS + r"\b|\b(we're|we are|i'm) (now )?(#1|number one|product of the (day|week|month)|profitable|cash[- ]flow positive|ramen profitable|trending)\b|\b(thank you|thanks) (for|to) (the |our )?" + NUM + r" " + UNITS + r")",
 "demo": r"(\bdemo\b|\b(watch|see) (it|this|how|me|us|the)\b|\bin action\b|\bhere's how it works\b|\bwalkthrough\b|\bscreen recording\b|\bsneak peek\b|\bquick (video|look|tour)\b|\bhere's (a|what|how)\b)",
}
NOT_LAUNCH = r"\b(partnership|partnering|partnered|partner with|collaboration with|joined|joining|recipients|winners|webinar|livestream|live stream|session|episode|podcast|keynote|conference|hackathon|meetup|summit|signatures|new paper|our paper|my paper|paper:|report|essay|blog post|lecture|talk|interview|giveaway|acquisition|acquired|acquiring)\b"
NOT_FEATURE = r"\b(new paper|paper:|report|essay|blog post|lecture|episode|podcast|article|newsletter|interview)\b"
ORDER = ["funding", "hiring", "launch", "open_source", "feature", "milestone", "demo"]
OPENER = r"^\W*(\d+/\s*\d*\s*)?(introducing|announcing|unveiling|meet|say hello to|new|launching|launched|now live|update|shipped|just shipped|changelog|today|big news|it's here|it's live)\b"
TECH = r"\b(app|api|sdk|model|models|agent|agents|ai|llm|gpt|open[- ]source|github|startup|founder|saas|product|feature|beta|waitlist|users|customers|developers|devs|platform|tool|software|code|coding|dashboard|plugin|extension|ios|android|web app|cli|library|framework|dataset|benchmark|inference|gpu|cloud|database|browser|workflow|integration|product hunt|yc|y combinator|seed|series [a-e]|arr|mrr|ship|shipped|shipping|launch|launched|build|building|built|release|v\d+\.\d+)\b"
REPRINT = r"\|\s*[\w .'-]{3,40},\s*[\w .'-]{3,40}\s{1,3}\S"          # "Headline | Author, Outlet  Body": a reprinted article


def clean(col="text"):
    return pl.col(col).fill_null("").str.replace_all(r"https?://\S+", " ").str.to_lowercase().str.replace_all("’", "'").str.replace_all(r"[ \t]+", " ")


def add_types(d: pl.DataFrame) -> pl.DataFrame:
    """Adds m_<type> flags, ptype_raw (first matching type, before the own-announcement test), techy, fp_head, opener."""
    d = d.with_columns(clean().alias("_lc")).with_columns(pl.col("_lc").str.slice(0, 300).alias("_head"))
    d = d.with_columns([pl.col("_head").str.contains(P[k]).alias("m_" + k) for k in ORDER] + [
        pl.col("_lc").str.contains(TECH).alias("techy"), pl.col("_head").str.contains(FP).alias("fp_head"), pl.col("_head").str.contains(OPENER).alias("opener"),
        pl.col("text").fill_null("").str.slice(0, 400).str.contains(REPRINT).alias("reprint"), pl.col("text").fill_null("").str.contains(r"^\s*@\w").alias("at_start")])
    d = d.with_columns((pl.col("m_launch") & ~pl.col("_head").str.contains(NOT_LAUNCH)).alias("m_launch"), (pl.col("m_feature") & ~pl.col("_head").str.contains(NOT_FEATURE)).alias("m_feature"),
                       (pl.col("m_demo") & ~pl.col("_head").str.contains(r"\bdemo day\b")).alias("m_demo"))
    e = pl.lit("other")
    for k in reversed(ORDER):
        e = pl.when(pl.col("m_" + k)).then(pl.lit(k)).otherwise(e)
    return d.with_columns(e.alias("ptype_raw"))


def own_expr():
    """The post is the account's own announcement (needs akind, has_video)."""
    own = pl.col("techy") & ~pl.col("reprint") & ~pl.col("at_start") & (pl.col("fp_head") | ((pl.col("akind") == "company") & pl.col("opener")))
    typed = pl.when(pl.col("ptype_raw") == "demo").then(pl.col("has_video")).otherwise(pl.col("ptype_raw") != "other")
    return own & typed
