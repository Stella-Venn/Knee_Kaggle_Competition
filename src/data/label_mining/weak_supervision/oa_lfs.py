"""Labeling functions (LFs) for the three OA compartment labels (Medial OA,
Lateral OA, PF OA) -- Step 1 of the weak-supervision pipeline described in
[redacted: private project notes]'s 2026-09-04 decision entry.

Built from the Phase 1 investigation ([redacted: private project notes], same date), restricted
to the 40-study dev subset of `data/splits/oa_investigation_split.json`. Two
findings from that investigation shaped this design:

- A bare "medial"/"lateral" adjective is not enough to attribute a finding to
  [redacted: quoted text from a restricted clinical dataset]
  patellofemoral (PF) finding despite containing "medial" (see study
  [redacted: quoted text from a restricted clinical dataset]
  finding, not a Medial OA one). So MEDIAL_TF_TERMS/LATERAL_TF_TERMS require a
  tibiofemoral anatomical noun (condyle, tibial plateau) attached to the
  laterality word, not the adjective alone; PF_TERMS matches on the anatomical
  site (patella, trochlea, patellar facet) regardless of any laterality
  modifier, since that modifier describes which facet, not which compartment.
- At least one case (`[redacted: study ID]` again) has a correctly-attributed,
  clearly-described finding that still disagrees with the gold label, for
  reasons this investigation didn't resolve. That's the whole reason this is
  an LF ensemble feeding a learned aggregator rather than one more hand-tuned
  rule -- no single LF here is trusted to be always right, including this one.

Design deviation from the original spec, flagged explicitly: a literal
"hedge LF that fires ABSTAIN on uncertain language" would return ABSTAIN
whether or not it actually detects hedging (ABSTAIN is also the default for
"nothing matched at all"), making it indistinguishable from a no-op in
Dawid-Skene-style aggregation -- an always-abstain-when-triggered column
carries no information for the label model to learn from. Implemented
instead as: (a) `make_keyword_proximity_lf` itself downgrades an
otherwise-POSITIVE vote to ABSTAIN when hedge language is nearby (hedging
actually changes this LF's vote), and (b) `make_hedge_flag_lf` is a separate
*diagnostic* function (not a vote-contributing LF) for Step 3's audit, to see
how often hedging co-occurs with OA mentions at all. `HEDGE_MARKERS` is a
first pass built from a single confirmed Fracture-investigation example
(Greek "could be attributed to") -- there is no proven, reusable hedge module
from Fracture to inherit; Fracture's own docstring explicitly declined to
generalize from one example. Treat this as equally unvalidated and expect
Step 3 to reveal whether it's worth keeping.

Vote convention (Snorkel's): POSITIVE = 1, NEGATIVE = -1, ABSTAIN = 0.

Three gaps found while writing tests against verbatim dev-subset text were
fixed after the initial version (see [redacted: private project notes] for the full account):
[redacted: quoted text from a restricted clinical dataset]
  stated once, findings following further than the tight proximity window
  reaches) is now also resolved via `_declared_compartment_before`, a
  state-tracking scope for section headers/paragraph openers, separate from
  the tight-window inline anatomical mentions.
- Forward negation: "Category:\nNone." (negation after the term, common in
  structured templates) is now caught via `rule_based.is_negated_forward`.
- Phrase adjacency: "[redacted: report text]" required strict adjacency and missed
  [redacted: quoted text from a restricted clinical dataset]
  "fissur"/"fisura" as a bare stem requiring a cartilage-specific companion
  nearby (not adjacent) instead.

Fixing these had a real, non-obvious side effect worth flagging rather than
treating the fixes as an unqualified win: Medial OA and Medial-negation
coverage/accuracy improved on the dev subset, but Lateral OA's
keyword_proximity accuracy got WORSE (50.0% -> 38.5%). The fixes correctly
find more real, well-attributed findings -- and more of those turn out to be
the same "correctly parsed, still disagrees with gold" pattern already
documented for study `[redacted: study ID]` (now 5 such cases, not 1-4). This isn't a
new bug to chase with a 6th hand-tuned exception; it's exactly the kind of
per-compartment LF unreliability the ensemble/learned-aggregation approach
(Steps 3-5) exists to detect and down-weight.
"""

import re

from data.label_mining.rule_based import NEGATION_MARKERS, is_negated, is_negated_forward
from data.text_preprocess import lower_report

POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0

COMPARTMENTS = ("Medial OA", "Lateral OA", "PF OA")

# Wide-net generic OA/degenerative/cartilage-damage terms (the "finding" half
# of the finding+compartment pair). Deliberately broad -- these terms alone
# don't distinguish compartments; that's COMPARTMENT_TERMS' job below.
OA_TERMS = [
    "osteoarthritis", "osteoarthrit", "joint space narrowing", "chondromalacia",
    "chondropathy", "chondrosis", "chondral", "cartilage loss", "cartilage thinning",
    "cartilage defect", "cartilage ulcer", "osteophyte",
    "articular cartilage loss", " oa ", " oa.", " oa,", "full thickness cartilage",
    "artrosis", "condropatía", "condropatia", "condromalacia", "úlcera condral",
    "ulcera condral", "adelgazamiento del cartílago", "osteofito", "condrosis",
    # Spanish counterpart of the English "chondral" stem -- "úlceras
    # condrales" (plural) never matched "úlcera condral" (2026-09-24,
    # dev studies [redacted: study ID]/[redacted: study ID]). "osteocondral" is still excluded
    # via _OA_EXCLUDE_PREFIXES, same as "osteochondral".
    "condral",
    # 2026-09-24: tr "gonartroz" (31 reports) and en "osteoarthrosis" (9)
    # matched nothing. "arthros"/"artroz" inside "osteo..." is caught by the
    # separate "osteoarthros" entry (the osteo- prefix exclusion below would
    # drop it); "hem(o)arthrosis"/"hämarthros" is excluded via _HEMARTHROSIS_PREFIXES.
    "osteoarthros", "arthros", "artroz",
    "pinzamiento",
    "osteoartrit", "kondromalazi", "kondropati", "kıkırdak kaybı", "kikirdak kaybi",
    "osteofit", "eklem aralığı daralması",
    "artroza", "hondromalacija", "hondropatija", "gubitak hrskavice", "suženj",
    "erozivne promjene",
    "αρθρίτιδα", "χόνδρωση", "χονδρομαλακία", "οστεόφυτο",
    "arthrose", "chondropathie", "chondromalazie", "knorpelschaden", "osteophyt",
    "gelenkspaltverschmälerung", "kraakbeenlijden", "kraakbeenverlies",
    "артроза", "хондропатия", "хондромалация", "остеофит",
    # Generic "[redacted: report text]" is real signal (e.g. "[redacted: report text]"
    # None."), but the bare stem ("degenerativ", "dejenerasyon", "εκφυλιστικ",
    # "дегенеративни") false-positived on meniscus/ligament degeneration --
    # [redacted: quoted text from a restricted clinical dataset]
    # finding, not cartilage/OA (confirmed against study [redacted: study ID], gold
    # Lateral OA=0). Requiring the compound phrase, not the bare root, is the
    # fix -- a real cartilage finding also uses one of the specific terms
    # above in practice.
    "degenerative change", "cambios degenerativos", "degenerativne promjene",
    "εκφυλιστική οστεοαρθρίτιδα", "дегенеративни промени", "degeneratieve veranderingen",
]

# Defense in depth alongside the compound-phrase fix above: if the ONLY thing
# nearby the OA-term match is a meniscus/ligament/tendon mention (not also a
# cartilage-specific word), it's describing that structure's degeneration,
# not the cartilage/compartment -- exclude it.
_MENISCUS_LIGAMENT_TERMS = [
    "meniscus", "menisco", "menisküs", "meniskus", "μηνίσκ", "менискус",
    # hr "meniska"/"menisku" -- "meniskus" isn't a substring of those (2026-09-24,
    # [redacted: quoted text from a restricted clinical dataset]
    "menisk",
    "ligament", "ligamento", "bağ", "ligaman", "σύνδεσμ", "връзка",
    "tendon", "tendón", "tendo",
]
_CARTILAGE_SPECIFIC_TERMS = [
    "cartilage", "cartílago", "cartilago", "chondr", "condr", "kıkırdak",
    "kikirdak", "hrskavic", "χόνδρ", "хрущял", "kraakbeen",
]

# "fissur"/"fisura" alone is ambiguous (a meniscal fissure isn't OA) --
# unlike the specific terms above, these bare stems only count with a
# cartilage-specific companion nearby. Originally encoded as the single
# literal phrase "[redacted: report text]", which required strict adjacency and
# [redacted: quoted text from a restricted clinical dataset]
# missing a genuine lateral finding in study [redacted: study ID]) -- this checks
# co-occurrence instead of adjacency.
_AMBIGUOUS_OA_TERMS = ["fissur", "fisura", "fisur"]

# The bare "chondral" stem (from "chondromalacia"/"chondropathy"/"chondrosis"/
# "chondral") matches inside "osteochondral" too (osteo+chondral) -- and
# osteochondral findings (OCD/cartilage-surface entities) are reliably
# excluded from Fracture for the same underlying reason (see rule_based.py's
# _FRACTURE_EXCLUDE, confirmed across 5+ languages there). Same exclusion
# applies here: a "chondral"-family term immediately preceded by "osteo" is
# excluded, confirmed against study [redacted: study ID] ("[redacted: report text]"),
# gold Lateral OA=0.
_OA_EXCLUDE_PREFIXES = ["osteo", "остео"]
# Blood in the joint, not OA: "hemarthrosis", "lipohemarthrosis", de
# "Hämarthros", tr "hemartroz" (~70 mentions in the pool).
_HEMARTHROSIS_PREFIXES = ("hem", "häm", "haem", "hema")
_OA_EXCLUDE_WINDOW = 5  # len("osteo") == len("остео"), immediately before the match

# "chondral"-family terms co-occurring with fracture language describe a
# [redacted: quoted text from a restricted clinical dataset]
# degenerative/OA finding -- unlike osteochondral above, "subchondral" alone
# is genuinely OA-relevant (subchondral cysts/edema/sclerosis are real OA
# features), so only fracture-adjacency is excluded, not "sub-" generally.
# Confirmed misattributing a Fracture-label finding to Lateral OA in study
# `[redacted: study ID]` during Step 3 investigation.
_OA_EXCLUDE_NEARBY_FRACTURE_TERMS = ["fractur", "fraktur", "kırık", "kırığ"]
_OA_EXCLUDE_FRACTURE_WINDOW = 20  # chars around a chondral-family match

# [redacted: quoted text from a restricted clinical dataset]
# together, NOT a bare "medial"/"lateral" -- see module docstring.
MEDIAL_TF_TERMS = [
    "medial femoral condyle", "medial tibial plateau", "medial compartment",
    "medial tibiofemoral",
    "cóndilo femoral medial", "platillo tibial medial",
    "compartimento femorotibial medial", "compartimento medial",
    "medial femorotibiaal", "mediaal femorotibiaal", "matig mediaal",
    "medijalnog kompartmenta femorotibijalnog", "medijalnog femoralnog kondila",
    "medijalnog tibijalnog platoa", "medijalnog kondila",
    "έσω διαμέρισμα", "έσω μηριαίο κόνδυλο", "έσω κνημιαίο πλατό",
    "медиалния кондил", "медиалното тибиално плато",
    "medialen femorotibialen", "medialen femurkondyle",
    # Turkish/Bulgarian: no direct evidence found in Phase 1 (only meniscus
    # degeneration mentions, not cartilage/OA attribution) -- best-effort
    # guesses, unvalidated, kept in specifically so Step 3's audit can
    # measure real reliability rather than silently having zero coverage.
    "medial femoral kondil", "medial tibial plato",
]
LATERAL_TF_TERMS = [
    "lateral femoral condyle", "lateral tibial plateau", "lateral compartment",
    "lateral tibiofemoral",
    "cóndilo femoral lateral", "platillo tibial lateral",
    "compartimento femorotibial lateral",
    "lateral femorotibiaal", "lateraal femorotibiaal",
    "lateralnog kompartmenta femorotibijalnog", "lateralnog femoralnog kondila",
    "lateralnog tibijalnog platoa", "lateralnog kondila", "lateralnog platoa",
    "έξω διαμέρισμα", "έξω μηριαίο κόνδυλο", "έξω κνημιαίο πλατό",
    "латералния кондил", "латералното тибиално плато",
    "lateralen femorotibialen", "lateralen femurkondyle",
    "lateral femoral kondil", "lateral tibial plato",
]
# PF attribution is by anatomical site, not laterality -- "[redacted: report text]"
# facet" and "[redacted: report text]" are BOTH PF findings.
PF_TERMS = [
    "patellofemoral", "patelofemoral", "femoropatellar", "femoropatelar",
    "femoropatellair", "patellofemoraal",
    "trochlea", "tróclea", "trohlea", "troclea", "troklea",
    "patellar facet", "faceta patelar", "faceta rotuliana",
    "rótula", "rotuliana", "patella", "patelar",
    "patelarni", "patelofemoralna",
    "μηριαία τροχιλία", "τροχιλία", "επιγονατίδα", "επιγονατιδομηριαί",
    "пателофеморал", "трохлея", "пателата",
    "patellofemoralen", "trochlear",
]

ALL_COMPARTMENT_TERMS = MEDIAL_TF_TERMS + LATERAL_TF_TERMS + PF_TERMS

COMPARTMENT_TERMS = {
    "Medial OA": MEDIAL_TF_TERMS,
    "Lateral OA": LATERAL_TF_TERMS,
    "PF OA": PF_TERMS,
}

# First-pass hedge markers -- see module docstring on how unvalidated this is.
HEDGE_MARKERS = [
    "could be", "possibly", "may represent", "may reflect", "questionable",
    "θα μπορούσε",  # el: "[redacted: report text]" -- the one confirmed Fracture example
    "podría representar", "podría corresponder", "podria representar",
]

MATCH_WINDOW = 60  # chars on either side of an OA-term match to look for compartment/hedge context

# Section-header / paragraph-opener phrases that DECLARE a compartment scope,
# as opposed to inline anatomical mentions ("[redacted: report text]") that
# describe one specific finding. Structured templates state these once, then
# describe findings across a longer span than MATCH_WINDOW reaches (confirmed
# missing a real negated finding in study [redacted: study ID], ~90 chars past "In the
# medial compartment,"). Narrower and more specific than COMPARTMENT_TERMS on
# purpose -- these are scope-setters, not general attribution vocabulary.
# Only the English structured/narrative convention is encoded: that's the
# only language this pattern was actually observed in during Phase 1/Step 1
# investigation. Not extended to other languages without evidence.
_COMPARTMENT_DECLARATIONS = {
    "Medial OA": ["medial compartment:", "in the medial compartment,", "the medial compartment,"],
    "Lateral OA": ["lateral compartment:", "the lateral compartment,", "in the lateral compartment,"],
    "PF OA": ["patellofemoral compartment:", "the patellofemoral joint,", "patellofemoral joint:"],
}
# Section headers that end a compartment declaration's scope even though
# they don't start a new one -- without these, a medial declaration would
# incorrectly "leak" all the way to the next compartment mention, however
# far away, once the window-limited check no longer applies.
_DECLARATION_RESET_HEADERS = [
    "cruciate ligament", "collateral ligament", "extensor mechanism",
    "muscles/tendons", "soft tissues", "impression:", "conclusion:",
    "intercondylar notch", "posteromedial", "posterolateral corner",
]
_DECLARATION_MAX_SCOPE = 500  # chars after a declaration before it stops applying


def _declared_compartment_before(text_lower: str, pos: int):
    """Return the compartment whose declaration most recently precedes `pos`
    (within _DECLARATION_MAX_SCOPE), or None if a reset header or a
    different compartment's declaration is closer, or nothing is found.
    """
    best_pos = -1
    best_compartment = None
    for compartment, phrases in _COMPARTMENT_DECLARATIONS.items():
        for phrase in phrases:
            idx = text_lower.rfind(phrase, 0, pos)
            if idx > best_pos:
                best_pos = idx
                best_compartment = compartment
    for phrase in _DECLARATION_RESET_HEADERS:
        idx = text_lower.rfind(phrase, 0, pos)
        if idx > best_pos:
            best_pos = idx
            best_compartment = None  # reset -- no compartment scope active
    if best_pos == -1 or pos - best_pos > _DECLARATION_MAX_SCOPE:
        return None
    return best_compartment


def _find_oa_matches(text_lower: str):
    """Yield (start_index, term_end_index) for every OA finding mention in
    text_lower, skipping matches that read as meniscus/ligament/tendon
    degeneration rather than cartilage/OA (see module docstring /
    _MENISCUS_LIGAMENT_TERMS), or as an osteochondral (OCD) entity (see
    _OA_EXCLUDE_PREFIXES). Ambiguous bare stems (_AMBIGUOUS_OA_TERMS) only
    yield when a cartilage-specific companion term is also nearby.
    """
    for term in OA_TERMS + _AMBIGUOUS_OA_TERMS:
        start = 0
        while True:
            idx = text_lower.find(term, start)
            if idx == -1:
                break
            start = idx + 1
            preceding = text_lower[max(0, idx - _OA_EXCLUDE_WINDOW):idx]
            if preceding in _OA_EXCLUDE_PREFIXES:
                continue
            if text_lower[max(0, idx - 4):idx].endswith(_HEMARTHROSIS_PREFIXES):
                continue
            fracture_window = text_lower[
                max(0, idx - _OA_EXCLUDE_FRACTURE_WINDOW):idx + len(term) + _OA_EXCLUDE_FRACTURE_WINDOW
            ]
            if any(ft in fracture_window for ft in _OA_EXCLUDE_NEARBY_FRACTURE_TERMS):
                continue
            local = _local_window(text_lower, idx)
            has_cartilage = any(t in local for t in _CARTILAGE_SPECIFIC_TERMS)
            if term in _AMBIGUOUS_OA_TERMS and not has_cartilage:
                continue
            has_ml = any(t in local for t in _MENISCUS_LIGAMENT_TERMS)
            if has_ml and not has_cartilage:
                continue
            yield idx, idx + len(term)


# A sentence ends at a period followed by whitespace, or at a blank line.
# NOT at a bare newline -- Croatian reports in this dataset hard-wrap lines
# [redacted: quoted text from a restricted clinical dataset]
# period after a digit or a standalone roman numeral -- Croatian/German
# write grades/ordinals as "[redacted: report text]" / "2. Grades", which would
# otherwise sever a finding from its own compartment term. A standalone
# header line ("\nImpresión:\n") is also a boundary -- found in study
# [redacted: quoted text from a restricted clinical dataset]
_SENTENCE_BOUNDARY = re.compile(
    r"(?<![0-9])(?<!\bi)(?<!\bii)(?<!\biii)(?<!\biv)\.\s|\n[ \t]*\n|\n[^\n.:]{1,40}:[ \t]*\n"
    # ...but a digit/numeral period followed by a NEWLINE is a sentence end
    # [redacted: quoted text from a restricted clinical dataset]
    # ("[redacted: report text]") -- 42 vs 21 occurrences in the full pool.
    r"|(?:\b(?:i|ii|iii|iv)|[0-9])\.\n(?![ \t]*stupnj)"
)


def _local_window(text_lower: str, idx: int, window: int = MATCH_WINDOW) -> str:
    """Up to `window` chars either side of idx, clipped to the sentence
    containing idx. Without the clip, a compartment term from an adjacent
    sentence bled into an unrelated finding's attribution -- confirmed in
    [redacted: quoted text from a restricted clinical dataset]
    rotuliana superficial." attributed the PF finding to Lateral OA),
    [redacted: quoted text from a restricted clinical dataset]
    `[redacted: study ID]` (a medial paragraph's "[redacted: report text]" attributed to
    [redacted: quoted text from a restricted clinical dataset]
    """
    lo, hi, _ = _window_bounds(text_lower, idx, window)
    return text_lower[lo:hi]


def _window_bounds(text_lower: str, idx: int, window: int = MATCH_WINDOW):
    """(lo, hi, sentence_end) for idx's local window -- see _local_window."""
    # Terms like " oa " start with whitespace, which puts idx one char BEFORE
    # the sentence boundary's end -- anchor on the first real character, or
    # the window reaches into the previous sentence (confirmed in study
    # [redacted: quoted text from a restricted clinical dataset]
    # " oa " match the previous sentence's medial attribution).
    anchor = idx
    while anchor < len(text_lower) and text_lower[anchor].isspace():
        anchor += 1
    start = 0
    for m in _SENTENCE_BOUNDARY.finditer(text_lower, 0, anchor):
        start = m.end()
    m = _SENTENCE_BOUNDARY.search(text_lower, anchor)
    end = m.start() + 1 if m else len(text_lower)
    return max(start, idx - window), min(end, idx + window), end


def _has_inline_term(text_lower: str, idx: int, terms) -> bool:
    """True if any of `terms` STARTS inside idx's local window, even if it
    runs past the window's right edge (but not past the sentence). A plain
    `term in _local_window(...)` check silently misses long multi-word terms
    [redacted: quoted text from a restricted clinical dataset]
    finding and was never matched in its own sentence.

    Also matches terms hard-wrapped across a line break ("cóndilo femoral\\n
    medial") -- the newline->space replace is char-for-char, so indices from
    _window_bounds (computed on the original text) still line up.
    """
    lo, hi, sentence_end = _window_bounds(text_lower, idx)
    text_flat = text_lower.replace("\n", " ")
    for t in terms:
        j = text_flat.find(t, lo, sentence_end)
        if j != -1 and j < hi:
            return True
    return False


# Severity qualifiers (Lever 2, 2026-09-24). Every keyword_proximity error on
# [redacted: quoted text from a restricted clinical dataset]
# nearly all of them were correctly-attributed findings described only as
# mild / low-grade / superficial / grade I-II -- e.g. two dev studies ([redacted: quoted text from a restricted clinical dataset]
# [redacted: quoted text from a restricted clinical dataset]
# report's full-thickness medial and high-grade PF findings are gold 1),
# [redacted: quoted text from a restricted clinical dataset]
# ). Consistent with the standard modified-
# Outerbridge split (grade 1-2 low, 3-4 high) -- the gold annotators appear
# not to call a compartment OA on low-grade chondral change alone. A finding
# is "low-grade only" if its local window has a low-grade qualifier and NO
# [redacted: quoted text from a restricted clinical dataset]
# at-least-moderate). Osteophytes/joint-space narrowing are at-least-moderate
# by definition (definite osteophytes = Kellgren-Lawrence >= 2).
# Word-START-anchored regexes, not plain substrings: "leve" (es: mild) is
# inside "level", "sever" inside "several", "early" inside "nearly".
# Entries are stems unless they end in \b.
_LOW_GRADE_RE_WORDS = re.compile(r"\b(?:" + "|".join([
    "mild", "low-grade", "low grade", "minimal", "superficial", "slight", "subtle", r"early\b",
    r"leves?\b", "incipiente", "discret[ao]", "liger[ao]", "mínim[ao]",
    "beginnend", "gering", "lichte", "leicht", "diskret",
    "niedriggradig", "erstgradig", "zweitgradig", "laaggradig", "lichtgradig",
    "blag", "početn",
    "ήπι", "αρχόμεν", "επιφανειακ",
    "hafif", "yüzeysel",
    r"лек[аи]\b", "начал", "повърхност",
]) + ")")
_AT_LEAST_MODERATE_RE_WORDS = re.compile(r"\b(?:" + "|".join([
    "high-grade", "high grade", "full thickness", "full-thickness", "near full", "near-full",
    "nearly full", "severe", "advanced", "moderate", "marked", "bone-on-bone", "denuded",
    "exposed bone", "osteophyt", "joint space narrowing",
    "moderad", r"sever[ao]s?\b", "avanzad", "importante", "espesor total", "prácticamente total",
    "osteofit", "pinzamiento",
    "matig", "ernstig", "gevorderd", "uitgesproken", "volledig", "osteofyt",
    "mäßig", "hochgradig", "fortgeschritten", "ausgeprägt", "vollschicht",
    "höhergradig", "mittelgradig", "drittgradig", "viertgradig", "hooggradig",
    "umjeren", "tešk", "uznapredoval", "izražen", "reduciran",
    "σοβαρ", "προχωρημέν", "μέτρι", "οστεόφυτ",
    "ileri derece", "orta derece", "şiddetli", "belirgin",
    # joint-space narrowing in the other OA_TERMS languages -- like the English
    # [redacted: quoted text from a restricted clinical dataset]
    "gelenkspaltverschmälerung", "eklem aralığı daralması", "suženj",
    "тежк", "умерен", "изразен", "напреднал", "остеофит",
]) + ")")
_GRADE_WORD = r"(?:grades?|grados?|grad|stadium|βαθμ\w*)"  # el "[redacted: report text]"
_HIGH_GRADE_RE = re.compile(
    _GRADE_WORD + r"\s*(?:(?:1|2|i|ii|uno|dos)\s*(?:[-–/]|y|a|to|and|or|o)\s*)?(?:3|4|iii|iv|tres|cuatro)\b"
    # optional ICRS sub-grade letter ("[redacted: report text]"); bg "степен" = degree
    r"|\b(?:iii|iv|3|4)\.?[ab]?\s*(?:°|\.?\s*(?:stupnj|степен))"
)
_LOW_GRADE_RE = re.compile(
    _GRADE_WORD + r"\s*(?:1|2|i|ii|uno|dos)\b"
    r"|\b(?:i|ii|1|2)\.?[ab]?\s*(?:°|\.?\s*(?:stupnj|степен))"
)

# What keyword_proximity votes when a compartment's ONLY genuine findings are
# low-grade: ABSTAIN (don't train on it either way) -- see [redacted: private project notes]
# 2026-09-24 for the ABSTAIN-vs-NEGATIVE comparison.
LOW_GRADE_ONLY_VOTE = NEGATIVE

# Disease-name OA terms are a radiologist's OA DIAGNOSIS, not a graded
# cartilage lesion -- "early/incipient osteoarthritis" still means OA is
# present. The low-grade downgrade (derived from chondral-lesion grading) is
# not applied to them. Gold dev `[redacted: study ID]`: "[redacted: report text]"
# [redacted: quoted text from a restricted clinical dataset]
# exempting these fixes it with no other dev change (2026-09-24).
_DISEASE_NAME_TERMS = frozenset([
    "osteoarthritis", "osteoarthrit", "osteoartrit", "osteoarthros", "arthros", "arthrose",
    "artrosis", "artroza", "artroz", " oa ", " oa.", " oa,", "αρθρίτιδα", "артроза",
])


def _is_low_grade_only(local: str) -> bool:
    if _HIGH_GRADE_RE.search(local) or _AT_LEAST_MODERATE_RE_WORDS.search(local):
        return False
    return bool(_LOW_GRADE_RE.search(local) or _LOW_GRADE_RE_WORDS.search(local))


def make_keyword_proximity_lf(compartment: str):
    """POSITIVE if any OA-term co-occurring with this compartment's
    attribution (inline anatomical mention, or falls within a compartment
    declaration's scope -- see _declared_compartment_before) is not negated
    and has no hedge language nearby; NEGATIVE only if every qualifying
    match found is negated (forward or backward) and none is a genuine
    positive; ABSTAIN if nothing qualifies at all. A genuine positive whose
    local context is low-grade only (see _is_low_grade_only) doesn't count
    as POSITIVE: if every genuine finding is low-grade only, the vote is
    LOW_GRADE_ONLY_VOTE instead.

    A genuine positive elsewhere always wins over an earlier negated match
    for the same compartment -- reports commonly rule out one specific
    finding while confirming another in the same breath ("[redacted: report text]"
    [redacted: quoted text from a restricted clinical dataset]
    positive if any of its described findings is. Confirmed missing this
    exact pattern (negated edema before a real, unnegated osteophyte
    finding) in study `[redacted: study ID]` during Step 3 investigation -- returning
    on the first match found, negated or not, was the bug.
    """
    compartment_terms = COMPARTMENT_TERMS[compartment]

    def lf(text: str) -> int:
        text_lower = lower_report(text)
        found_negative = False
        found_low_grade = False
        for idx, end in _find_oa_matches(text_lower):
            local = _local_window(text_lower, idx)
            inline_match = _has_inline_term(text_lower, idx, compartment_terms)
            declared_match = _declared_compartment_before(text_lower, idx) == compartment
            if not (inline_match or declared_match):
                continue
            if is_negated(text_lower, idx) or is_negated_forward(text_lower, end):
                found_negative = True
                continue  # keep looking -- a later genuine positive should win
            if any(h in local for h in HEDGE_MARKERS):
                continue  # hedged -- don't vote POSITIVE, keep looking
            if text_lower[idx:end] not in _DISEASE_NAME_TERMS and _is_low_grade_only(local):
                found_low_grade = True
                continue  # keep looking -- an at-least-moderate finding should win
            return POSITIVE
        if found_low_grade:
            return LOW_GRADE_ONLY_VOTE
        return NEGATIVE if found_negative else ABSTAIN

    return lf


def make_negation_lf(compartment: str):
    """NEGATIVE if a compartment-attributed OA-term match is negated (forward
    or backward) and no OTHER compartment-attributed match is a genuine
    (non-negated, non-hedged) positive; ABSTAIN otherwise. Redundant with
    make_keyword_proximity_lf's own negation check by design -- an
    independent LF voting the same way on negated cases is useful signal
    for the aggregator (agreement raises confidence), not wasted duplication.

    Only ever votes NEGATIVE or ABSTAIN, never POSITIVE (that's
    keyword_proximity's job) -- but a report commonly rules out one specific
    finding while confirming another in the same breath ("[redacted: report text]"
    [redacted: quoted text from a restricted clinical dataset]
    not override a genuine positive elsewhere for the same compartment.
    Confirmed missing this exact pattern in study `[redacted: study ID]` (2026-09-23):
    a genuine lateral chondral defect finding, with a separately-negated
    "[redacted: report text]" sub-finding right before it in the same
    paragraph, wrongly voted NEGATIVE under the old return-on-first-negated-
    match logic. Same bug class already fixed in make_keyword_proximity_lf
    for a different study (`[redacted: study ID]`, 2026-09-04) -- this LF never got
    the equivalent fix until now.
    """
    compartment_terms = COMPARTMENT_TERMS[compartment]

    def lf(text: str) -> int:
        text_lower = lower_report(text)
        found_negative = False
        for idx, end in _find_oa_matches(text_lower):
            local = _local_window(text_lower, idx)
            inline_match = _has_inline_term(text_lower, idx, compartment_terms)
            declared_match = _declared_compartment_before(text_lower, idx) == compartment
            if not (inline_match or declared_match):
                continue
            if is_negated(text_lower, idx) or is_negated_forward(text_lower, end):
                found_negative = True
                continue
            if any(h in local for h in HEDGE_MARKERS):
                continue  # hedged, not a genuine positive -- keep looking
            return ABSTAIN  # a genuine positive exists for this compartment -- don't vote NEGATIVE
        return NEGATIVE if found_negative else ABSTAIN

    return lf


def make_hedge_flag_lf(compartment: str):
    """Diagnostic only -- NOT a vote-contributing LF (see module docstring).
    Returns True if hedge language was found near a compartment-attributed
    OA-term match, for Step 3's audit of how often hedging occurs at all.
    """
    compartment_terms = COMPARTMENT_TERMS[compartment]

    def flag(text: str) -> bool:
        text_lower = lower_report(text)
        for idx, _end in _find_oa_matches(text_lower):
            local = _local_window(text_lower, idx)
            inline_match = _has_inline_term(text_lower, idx, compartment_terms)
            declared_match = _declared_compartment_before(text_lower, idx) == compartment
            if (inline_match or declared_match) and any(h in local for h in HEDGE_MARKERS):
                return True
        return False

    return flag


def has_compartment_mention(text: str, compartment: str) -> bool:
    """True if ANY OA-term match is attributed to `compartment` -- negated,
    hedged, low-grade, or genuine. Used to decide "the report never talks
    about this compartment" (final_labels.unmentioned_negative_matrix), where
    even a negated or hedged mention means the case isn't a plain omission.
    """
    text_lower = lower_report(text)
    terms = COMPARTMENT_TERMS[compartment]
    return any(
        _has_inline_term(text_lower, idx, terms)
        or _declared_compartment_before(text_lower, idx) == compartment
        for idx, _end in _find_oa_matches(text_lower)
    )


def generic_no_compartment_flag(text: str) -> bool:
    """Diagnostic only, not a vote-contributing LF: True if the report
    mentions an OA/degenerative term with no compartment attribution
    findable nearby (inline or declared) at all -- tracks how often this
    happens, informing whether a "generic OA, unspecified compartment"
    signal is common enough to matter for coverage.
    """
    text_lower = lower_report(text)
    for idx, _end in _find_oa_matches(text_lower):
        local = _local_window(text_lower, idx)
        inline_match = _has_inline_term(text_lower, idx, ALL_COMPARTMENT_TERMS)
        declared_match = _declared_compartment_before(text_lower, idx) is not None
        if not (inline_match or declared_match):
            return True
    return False


# translate_then_match: decided 2026-09-04 (offline, HuggingFace transformers
# + facebook/nllb-200-distilled-600M -- see translate.py). No separate
# matching function needed: an LF here is a pure text -> vote function, so
# feeding make_keyword_proximity_lf/make_negation_lf pre-translated English
# text instead of the original IS the translate-then-match LF -- a genuinely
# independent signal (translation errors differ from direct multilingual
# matching) without duplicating matching logic. Callers must translate
# reports to English first (see translate.py's translate_batch) and pass the
# translated text as `text`; these LFs don't translate internally, since
# translation should happen once per report and be reused/cached, not
# re-run per compartment or per LF.


LF_FACTORIES = {
    "keyword_proximity": make_keyword_proximity_lf,
    "negation": make_negation_lf,
}

# Step 3 audit originally found ("Lateral OA", "keyword_proximity") at 38.5%
# accuracy -- below chance -- and killed it. Revisited when asked to check
# whether the false positives were actually near meniscus/ligament/condyle
# mentions rather than genuine cartilage statements ([redacted: private project notes],
# 2026-09-04): found three real bugs, not that exact pattern, but the same
# family (proximity/negation logic picking up the wrong nearby context):
# 1. Forward negation crossed sentence boundaries ("...chondrosis. No acute
#    fracture..." wrongly negated the chondrosis finding using an unrelated
#    sentence's "No") -- fixed in rule_based.is_negated_forward.
# 2. "chondral" (from chondrosis/chondromalacia/etc.) matched inside
#    "[redacted: report text]", a Fracture-family finding, not OA -- fixed via
#    _OA_EXCLUDE_NEARBY_FRACTURE_TERMS.
# 3. The real logic bug, unrelated to term lists: this LF returned on the
#    FIRST matching finding, so a negated sub-finding ("[redacted: report text]"
#    edema") blocked a later genuine positive ("Osteophytes are present")
#    for the same compartment in the same report -- fixed by scanning for
#    any positive before falling back to a negated result (see
#    make_keyword_proximity_lf's docstring).
# After all three: 58.3% accuracy at 30.0% coverage -- no longer below
# chance, so no longer killed. Still WEAK (down-weight, not exclude) --
# expect Step 5 to learn a modest reliability weight, not treat it as
# trustworthy as Medial/PF's versions.
#
# One more bug in the same family, `[redacted: study ID]` (a PF finding pulling
# "lateral" attribution from the PRECEDING sentence), was deferred here and
# fixed 2026-09-24 by sentence-bounding _local_window -- together with a
# window-edge fix (_has_inline_term) and severity-aware voting
# (_is_low_grade_only). Lateral OA keyword_proximity dev accuracy: 58.3% ->
# 83.3%. See [redacted: private project notes] 2026-09-24.
#
# ("Lateral OA", "negation") scored 75.0% (up from 60.0%, same fixes) on a
# small sample (4 scored) -- kept, not flagged.
DISABLED_LF_COLUMNS = set()


def run_oa_lfs(text: str) -> dict:
    """Run all implemented, vote-contributing LFs against one report.

    Returns {compartment: {lf_name: vote}} for the 3 OA compartments. Does
    NOT include translate_then_match (not implemented) or the diagnostic
    hedge/generic-compartment flags (not votes -- see
    make_hedge_flag_lf/generic_no_compartment_flag for those, used in
    Step 3's audit separately).
    """
    return {
        compartment: {
            lf_name: factory(compartment)(text)
            for lf_name, factory in LF_FACTORIES.items()
        }
        for compartment in COMPARTMENTS
    }
