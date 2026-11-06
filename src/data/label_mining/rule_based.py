"""Rule-based label mining from multilingual radiology report text.

[redacted: detail about the data challenge this code was written for]
`Report` fields, meant for the 4349 training studies that have a report but
no gold label. Every miner here is scoped to what a breadth-first check
against the 58 gold-labeled studies actually supported -- see
[redacted: private project notes]'s 2026-09-03 entries for the full investigation. Two
findings drove the design instead of a plain per-language vocabulary list:

1. Presence-of-term is not the same as the label. For `Fracture`, mentions of
   "[redacted: report text]" (an OCD/cartilage-surface entity) are reliably
   labeled 0 while "[redacted: report text]" (bone-only) is reliably labeled 1,
   confirmed across English, Spanish, Turkish, Croatian, and Bulgarian
   reports. `_FRACTURE_EXCLUDE` encodes only that one well-evidenced
   exclusion -- it deliberately does not attempt general hedge-language
   detection ("[redacted: report text]" vs. "[redacted: report text]" behaved differently across two
   single-example cases each), since one example per side isn't enough to
   build a rule from.
2. Some labels fail structurally, not from missing vocabulary. All three OA
   labels (Medial/Lateral/PF) had zero true positives against the gold 58
   from direct terminology matching ("osteoarthritis"/"artrosis"/etc.)
   because they're driven by compartment-localized cartilage-damage language
   ("chondral ulcer", "[redacted: report text]") rather than any disease-name
   term -- a two-part (finding-type + anatomical-compartment) extraction
   problem, not a vocabulary gap. Left out of this module -- see
   `label_mining.weak_supervision.oa_lfs` for the weak-supervision approach
   being built for these instead (2026-09-04 decision, [redacted: private project notes]).
3. Sometimes the gold labels themselves are genuinely inconsistent, not just
   hard to predict from text. `Effusion` at mild/minimal/trace severity
   split roughly 50/50 in the gold 58 -- including two reports from what
   looks like the same reporting template, near-identical wording, opposite
   labels. `mine_effusion` returns None (no signal) for a mild-severity
   mention rather than guess, and only calls it confidently for
   strong-severity or unqualified mentions (see `_EFFUSION_MILD_TERMS`).

Implemented: Fracture, Baker's, Effusion, Contusion, plus a first pass at ACL,
MCL, Medial Meniscus, Lateral Meniscus, and Synovitis (see the section comment
above `_mine_ligament` for how those differ: they abstain on "never mentioned"
and on hedged / mild-grade / degeneration-only findings). `Contusion` is the least
confident of the first four -- exploratory checks found genuine label
ambiguity around bone vs. soft-tissue contusion language that this miner
does not attempt to resolve (see `mine_contusion`). Synovitis is the weakest
of the new five: it fires on only ~8% of reports and almost only positively,
so it can't supply negatives ([redacted: private project notes], 2026-09-21).

Not implemented (return MinedLabel(value=None, evidence=None) from
`mine_report`): Medial OA, Lateral OA, PF OA (weak-supervision approach, see
above).
"""

import re
from dataclasses import dataclass

from data.text_preprocess import (
    FORWARD_NEGATION_MARKERS,
    FORWARD_NEGATION_WINDOW,
    NEGATION_MARKERS,
    NEGATION_WINDOW,
    is_negated,
    is_negated_forward,
    is_negated_whole_word,
    lower_report,
    split_clauses,
)
from utils.constants import LABEL_COLUMNS

# NEGATION_MARKERS, is_negated, is_negated_forward re-exported here (not
# underscore-prefixed) because label_mining.weak_supervision.oa_lfs imports
# them from this module; the actual definitions moved to text_preprocess.py
# (module 5, 2026-09-22) -- see that module's docstring for why.


@dataclass(frozen=True)
class MinedLabel:
    value: int | None  # 0, 1, or None (no signal / not implemented)
    evidence: str | None  # matched snippet, for audit/debugging


def _find_first_unnegated(text: str, terms, exclude_terms=(), exclude_window: int = 40):
    """Return (start_index, snippet) for the first non-negated match of any
    term in `terms`, skipping matches that have an exclude_term nearby
    (within `exclude_window` chars on either side). None if no valid match.
    """
    text_lower = lower_report(text)
    for term in terms:
        start = 0
        while True:
            idx = text_lower.find(term, start)
            if idx == -1:
                break
            start = idx + 1
            if is_negated(text_lower, idx):
                continue
            # Post-posed "[redacted: report text]" / Turkish "kırık yok" -- was mined
            # POSITIVE ([redacted: study ID], 2026-09-24). Terms are often stems
            # ("fractur"), so check from the end of the WORD, not the stem.
            word_end = idx + len(term)
            while word_end < len(text_lower) and text_lower[word_end].isalpha():
                word_end += 1
            if is_negated_forward(text_lower, word_end):
                continue
            local = text_lower[max(0, idx - exclude_window):idx + exclude_window]
            if any(ex in local for ex in exclude_terms):
                continue
            return idx, text[max(0, idx - 60):idx + 60]
    return None


def _mine_by_terms(text: str, terms, exclude_terms=()) -> MinedLabel:
    match = _find_first_unnegated(text, terms, exclude_terms)
    if match is None:
        return MinedLabel(value=0, evidence=None)
    _, snippet = match
    return MinedLabel(value=1, evidence=snippet)


_FRACTURE_TERMS = [
    "fractur", "kırık", "kırığ", "kirik", "fractuur", "fraktur", "fraktür",
    "bruch", "κάταγ", "фрактур",
]
# Only the OCD/cartilage-surface sense is excluded -- "subchondral"/
# "subarticular"/"insufficiency" fracture language is reliably positive and
# must NOT be excluded (see module docstring, finding 1).
_FRACTURE_EXCLUDE = ["osteochondral", "osteocondral", "остеохондрал"]


def mine_fracture(text: str) -> MinedLabel:
    return _mine_by_terms(text, _FRACTURE_TERMS, _FRACTURE_EXCLUDE)


# Requires the cyst-specific phrase, not a "poplite" stem -- "popliteus"
# (a muscle/tendon) and "popliteal artery/vein" mentions are common and
# unrelated; a bare stem match false-positives on those.
_BAKERS_TERMS = [
    "baker's cyst", "bakers cyst", "baker cyst",
    "popliteal cyst", "quiste poplíteo", "quiste de baker",
    "popliteal kist", "poplitea cyste",
    "poplitealzyste", "baker-zyste",
    "bakerova cist", "poplitealna cist",
    "κύστη baker", "κύστη poplite",
    "киста на бейкър",
    "kyste poplité", "kyste de baker",
    # 2026-09-24, missed on gold dev: nl "[redacted: report text]" (the entry above
    # was a typo), hr "[redacted: report text]", bg "[redacted: report text]"; tr "[redacted: report text]".
    "popliteale cyst", "baker cist", "бекерова киста", "бейкърова киста",
    "baker kist", "bakerzyste",
]


# Small/trace cysts: gold 0 in 3 of 4 gold-dev cases ("[redacted: report text]",
# [redacted: quoted text from a restricted clinical dataset]
# bg "малка ... Бекерова киста", gold 1). Too few and too mixed to vote
# NEGATIVE, so abstain -- the same call mine_effusion makes for mild
# [redacted: quoted text from a restricted clinical dataset]
# cista 22x31x37 mm" is a small EFFUSION then a real cyst, gold 1), and
# word-anchored (hr "mala" is inside "hondromalacija").
_BAKERS_SMALL_RE = re.compile(
    r"\b(?:trace|small|tiny|minimal|pequeñ[oa]|μικρ\w*|мал(?:ка|ък)|mal[aeio]|manj[aei]|klein\w*|küçük)\b"
)
_BAKERS_SMALL_WINDOW = 40


def mine_bakers(text: str) -> MinedLabel:
    match = _find_first_unnegated(text, _BAKERS_TERMS)
    if match is None:
        return MinedLabel(value=0, evidence=None)
    idx, snippet = match
    before = lower_report(text)[max(0, idx - _BAKERS_SMALL_WINDOW):idx]
    before = re.split(r"[.\n]", before)[-1]  # same sentence only
    if _BAKERS_SMALL_RE.search(before):
        return MinedLabel(value=None, evidence=snippet)
    return MinedLabel(value=1, evidence=snippet)


_EFFUSION_TERMS = [
    "effusion", "derrame", "efüzyon", "sıvı art", "sivi art",
    "effusie", "vochtophoping", "erguss", "izljev",
    "συλλογή υγρού", "излив", "épanchement",
]

# A systematic read of all 58 gold-labeled effusion mentions found genuine
# annotation inconsistency at low severity, not a pattern this module failed
# to find: two reports using near-identical template phrasing ("Minimal
# [redacted: quoted text from a restricted clinical dataset]
# before and after) landed on opposite gold labels. Strong-severity mentions
# ("large", "massive", "extensive") were consistently positive (3/3), and
# unqualified mentions (no severity word at all) were mostly positive
# (15/20). So: call it confidently for those two cases, but return
# "no signal" (None) for a mild/minimal/trace mention instead of guessing --
# guessing on that bucket was right only about half the time.
_EFFUSION_MILD_TERMS = [
    "mild", "minimal", "trace", "small", "leve", "mínimo", "minimo",
    "gering", "manja", "manji", "hafif", "az miktarda",
]
_EFFUSION_STRONG_TERMS = [
    "large", "massive", "extensive", "significant", "severe",
    "opsežan", "importante", "abundante",
]
_EFFUSION_SEVERITY_WINDOW = 40  # chars to look back from the match for a severity word


def mine_effusion(text: str) -> MinedLabel:
    match = _find_first_unnegated(text, _EFFUSION_TERMS)
    if match is None:
        return MinedLabel(value=0, evidence=None)
    idx, snippet = match
    text_lower = lower_report(text)
    window = text_lower[max(0, idx - _EFFUSION_SEVERITY_WINDOW):idx + 10]
    is_mild = any(t in window for t in _EFFUSION_MILD_TERMS)
    is_strong = any(t in window for t in _EFFUSION_STRONG_TERMS)
    if is_mild and not is_strong:
        return MinedLabel(value=None, evidence=snippet)
    return MinedLabel(value=1, evidence=snippet)


# Known limitation: exploratory checks found contusion language doesn't
# cleanly separate on bone-vs-soft-tissue lines the way a first look
# [redacted: quoted text from a restricted clinical dataset]
# "[redacted: report text]" explicitly negated in the same sentence; another: a
# hedged "[redacted: report text]" alone was also positive) -- not enough
# evidence to build a bone/soft-tissue rule, so this miner does not attempt
# one. Expect this to be the least reliable of the four when validated
# against the gold 58.
_CONTUSION_TERMS = [
    # "contus" stem (was "contusion"/"contusión"): nl "botcontusie" missed on
    # gold dev ([redacted: study ID]); also covers es "contusivo". hr "kontuzija".
    "contus", "kontüzyon", "kontuzyon", "kontusion", "kontuzij",
    "μώλωπ", "контуз", "bone bruise", "bruise", "botkneuzing",
]


def mine_contusion(text: str) -> MinedLabel:
    return _mine_by_terms(text, _CONTUSION_TERMS)


# --- ACL / MCL / Medial Meniscus / Lateral Meniscus / Synovitis (first pass) ---
#
# These five differ from the miners above in two ways, both driven by the dev
# half of the gold 58 (data/splits/oa_investigation_split.json):
#
# 1. They abstain (None) when a structure is never mentioned, instead of
#    returning 0. Fracture/Baker's/etc. can say "absent from the report ->
#    negative" because radiologists name a finding when it's there; ligament,
#    meniscus, and synovium status is often left to the reader (3 dev reports
#    mention no ligament at all, and 7 Synovitis-positive dev reports never
#    say "synovitis"), so silence is not evidence.
# 2. Per structure they decide clause by clause (one clause = one sentence or
#    one templated line, wrapped lines re-joined) among:
#      POS  -- an unnegated, unhedged injury term (tear/rupture/sprain/...)
#      NEG  -- a negated injury term ("no tear") or a normal/intact marker
#      WEAK -- degeneration/signal-change only, a hedged tear ("suspect",
#              "R/O"), or (ligaments, Synovitis) a mild-tier grade. Gold labels
#              for these split roughly 50/50 in the dev half, so the miner
#              abstains, mirroring mine_effusion's mild-severity handling.
#    and combine clauses as POS > WEAK > NEG > not mentioned (abstain).

# split_clauses and is_negated_whole_word moved to text_preprocess.py (module 5,
# 2026-09-22) -- imported above, behavior verified identical against all 4407
# real reports before the move ([redacted: private project notes]).

_INJURY = re.compile(
    r"\btear\w*|\btorn\b|\brupt\w*|\brüpt\w*|rotura|desgarr\w*|riss\w*|scheur\w*|d[ée]chir\w*|"
    r"yırt\w*|yirt\w*|ρήξ\w*|ρηξ\w*|рупт\w*|скъсван\w*|bucket.?handle|meniscectom\w*|"
    r"sprain\w*|esguince|avulsi\w*|ausriss|injur\w*|distors\w*|insufficien\w*|strain\b|"
    r"lesi[oó]n\w*|l[ée]sion\w*|entorse|d[ée]sinsertion|auffaser\w*"
)
_DEGEN = re.compile(
    r"degenerat\w*|degenerac\w*|dejenerasyon\w*|degeneratie\w*|εκφύλ\w*|дегенерат\w*|"
    r"mucoid|muco\W?i?de|mukoid|laxit\w*|laksite\w*|signal (?:change|increase)\w*|"
    r"increased signal|aumento de señal|hyperintens\w*|hiperintenz\w*|интенз\w*|"
    r"thicken\w*|edema"
)
_HEDGE = re.compile(
    r"suspect\w*|suspicio\w*|possible|possibly|probabl\w*|\bmay\b|\bmight\b|\bcould\b|"
    r"r/o|rule out|suggest\w*|question\w*|\?|moguć\w*|podría|posible|sospech\w*|"
    r"olası|olasi|düşünd\w*|dusund\w*|\bdd\b|\bwohl\b|verdacht|vermutlich|evtl\w*|"
    r"impresiona|se sugiere|compatible|može odgovarati|mo[zž]e odgovarati"
)
_NORMAL = re.compile(
    r"(?<![a-z])normal\w*|unremarkable|intact\w*|preserved|conservad\w*|uredn\w*|"
    r"primjerenih|održan\w*|continu\w*|kontinuitet\w*|odrzan\w*|normaal|normaldir|intakt\w*|φυσιολογικ\w*|"
    r"нормал\w*|запазен\w*|интактн\w*|within normal|dentro de l[ií]mites|"
    r"sin alteraci\w*|αξιόλογα|ohne befund|geen afwijking\w*|sans particularit\w*|"
    r"без особен\w*|б\.о\.|regelrecht\w*|unauffällig\w*|durchgängig\w*|kein anhalt"
)
# Complete tears / high-grade injury; overrides a mild-tier word in the same clause
# [redacted: quoted text from a restricted clinical dataset]
_STRONG_GRADE = re.compile(
    r"complete\w*|full.thickness|high.grade|grade\s*(?:2|ii|3|iii)\b|grado\s*(?:2|ii|3|iii)\b|"
    r"graad\s*[23]\b|completo|volledig\w*|kompletn\w*|tam kat|\btotal\b|пълна|"
    r"\brupt\w*|rotura|\brüpt\w*|ruptur\w*"
)
_MILD_GRADE = re.compile(
    r"interstitial|partial|parsiyel|parcial|parti[eë]l\w*|djelomi\w*|low.grade|"
    r"grade\s*(?:1|i)\b|grado\s*(?:1|i)\b|mild\w*|minimal\w*|m[ií]nim\w*|slight\w*|"
    r"gering\w*|leve\b|sprain\w*"
)

POS, NEG, WEAK = "pos", "neg", "weak"


def _classify_clause(clause: str, use_grades: bool, allow_pos: bool = True):
    """Verdict for one clause about one structure (POS/NEG/WEAK/None),
    assuming the clause is about that structure. use_grades enables the
    mild-tier-grade abstention (ligaments, Synovitis); meniscus tears count
    as positive at any severity in the dev gold. allow_pos=False is for a
    clause that only mentions the structure by a general plural term
    ("[redacted: report text]"): it can never establish an injury.
    """
    injuries = [m for m in _INJURY.finditer(clause)]
    firm = [m for m in injuries if not is_negated_whole_word(clause, m.start(), m.end())]
    if firm and allow_pos:
        if _HEDGE.search(clause):
            return WEAK
        if use_grades and _MILD_GRADE.search(clause) and not _STRONG_GRADE.search(clause):
            return WEAK
        return POS
    if firm:  # unnegated injury, but attributable to a different structure
        return None
    if injuries or _NORMAL.search(clause):
        return NEG
    if _DEGEN.search(clause):
        return WEAK
    return None


def _combine(verdicts, evidence):
    """POS > WEAK > NEG > nothing said (None: abstain)."""
    for wanted, value in ((POS, 1), (WEAK, None), (NEG, 0)):
        if wanted in verdicts:
            return MinedLabel(value=value, evidence=evidence.get(wanted))
    return MinedLabel(value=None, evidence=None)


_ACL_SPECIFIC = re.compile(
    r"\bacl\b|anterior cruciate|\blca\b|cruzado anterior|ön çapraz|on capraz|voorste kruisband|"
    r"\bvkb\b|prednj\w* križn\w*|prednj\w* krizn\w*|предна кръстна|предната кръстна|"
    r"cro[ií]s[ée] ant[ée]rieur|vordere[sn]?\W+(?:und\W+hintere[sn]?\W+)?kreuzband|"
    r"πρόσθιο\w* χιαστ|vordere[sn]? kreuzband"
)
_PCL_SPECIFIC = re.compile(
    r"\bpcl\b|posterior cruciate|\blcp\b|cruzado posterior|arka çapraz|arka capraz|"
    r"achterste kruisband|stražnj\w* križn\w*|strazn\w* krizn\w*|задна кръстна|задната кръстна|"
    r"cro[ií]s[ée] post[ée]rieur|"
    r"οπίσθιο\w* χιαστ|hintere[sn]? kreuzband|\bhkb\b"
)
_CRUCIATE_GENERAL = re.compile(
    r"cruciate ligaments|ligamentos cruzados|çapraz bağlar|capraz baglar|χιαστ\w* σ[υύ]νδ\w*|"
    r"кръстните връзки|kruisbanden|križn\w* ligament\w*|kreuzbänder|kreuzbaender|χιαστ\w*|"
    r"ligaments cro[ií]s[ée]s|kreuzband"
)
_MCL_SPECIFIC = re.compile(
    r"\bmcl\b|medial\W+(?:and\W+|&\W*)?(?:lateral\W+)?collateral|collateral medial|"
    r"colateral medial|colateral interno|colaterales? medial|medial\w* collateral|"
    r"medijaln\w* kolateraln\w*|medyal\w* kollateral|iç yan bağ|"
    r"медиален колатерал|вътрешна странична|innenband|mediale?s? seitenband|"
    r"mediale collaterale|tibial collateral|collat[ée]ral (?:m[ée]dial|interne)|"
    r"ligament lat[ée]ral interne|mediales? kollateralband|mediales? kapselband|innenband"
)
_LCL_ONLY = re.compile(
    r"\blcl\b|lateral collateral|colateral lateral|lateraln\w* kolateraln\w*|"
    r"collat[ée]ral (?:lat[ée]ral|externe)|laterales? kollateralband"
)
_COLLATERAL_GENERAL = re.compile(
    r"collateral ligaments|colaterales|yan bağlar|yan baglar|πλάγι\w* σ[υύ]νδ\w*|"
    r"kolateraln\w* ligament\w*|kollateral ligaman|странични връзки|collaterale ligamenten|"
    r"колатерал\w* лигамент\w*|колатерал\w*|ligaments collat[ée]raux|kollateralb[äa]nder|πλάγι(?:οι|ων)\b"
)


def _mine_ligament(text, specific, other_specific, general) -> MinedLabel:
    verdicts, evidence = set(), {}
    for clause in split_clauses(text):
        if specific.search(clause):
            verdict = _classify_clause(clause, use_grades=True)
        elif general.search(clause) and not other_specific.search(clause):
            verdict = _classify_clause(clause, use_grades=True, allow_pos=False)
        else:
            continue
        if verdict:
            verdicts.add(verdict)
            evidence.setdefault(verdict, clause[:120])
    return _combine(verdicts, evidence)


def mine_acl(text: str) -> MinedLabel:
    return _mine_ligament(text, _ACL_SPECIFIC, _PCL_SPECIFIC, _CRUCIATE_GENERAL)


def mcl_mentioned(text: str) -> bool:
    """True if any clause mentions the MCL specifically or the collateral
    ligaments generally -- the "never mentioned" test for
    final_labels.unmentioned_negative_matrix (mine_mcl's own abstain also
    covers hedged / mild-grade mentions, which must stay masked).
    """
    return any(_MCL_SPECIFIC.search(c) or _COLLATERAL_GENERAL.search(c) for c in split_clauses(text))


def mine_mcl(text: str) -> MinedLabel:
    return _mine_ligament(text, _MCL_SPECIFIC, _LCL_ONLY, _COLLATERAL_GENERAL)


_MENISCUS = re.compile(
    r"\bmenisc(?:us|o|i|os|al|en)?\b|menisk\w*|μηνίσκ\w*|мениск\w*|m[ée]nisque\w*"
)
_MENISCI_PLURAL = re.compile(r"\bmenisci\b|\bmeniscen\b|\bmeniscos\b|menisken\b|менискуси|μηνίσκοι|μηνίσκων|m[ée]nisques\b")
_MEDIAL = re.compile(
    r"(?<![^\W\d_])(?:m[ée]di[aá]l\w*|medij\w*|medyal\w*|intern[oae]\w*|innen\w*|έσω|εσω|"
    r"медиал\w*|вътрешн\w*|mediaal|iç)(?![^\W\d_])|(?<![^\W\d_])(?:medi[aá]l\w*|medij\w*|"
    r"medyal\w*|intern[oae]\w*|innen\w*|έσω|εσω|медиал\w*|вътрешн\w*)"
)
_LATERAL = re.compile(
    r"(?<![^\W\d_])(?:lat[ée]ra\w*|extern\w*|dış|εξω|έξω|латерал\w*|външ\w*|außen\w*|aussen\w*)"
)
_BOTH_CONNECTOR = re.compile(r"^\W*(?:and|&|y|ve|en|und|sowie|et|ou|και|и|i|or|/)?\W*$")
_MENISCUS_BEFORE, _MENISCUS_AFTER = 30, 20


def _meniscus_sides(clause: str):
    """Set of {"Medial", "Lateral"} the clause's meniscus mentions refer to."""
    sides = set()
    for m in _MENISCUS.finditer(clause):
        lo, hi = max(0, m.start() - _MENISCUS_BEFORE), min(len(clause), m.end() + _MENISCUS_AFTER)
        events = sorted(
            [(e.start(), e.end(), "Medial") for e in _MEDIAL.finditer(clause, lo, hi)]
            + [(e.start(), e.end(), "Lateral") for e in _LATERAL.finditer(clause, lo, hi)]
        )
        if not events:
            if _MENISCI_PLURAL.search(clause[m.start():m.end()]):
                sides |= {"Medial", "Lateral"}
            continue
        found = {ev[2] for ev in events}
        if len(found) == 2:
            first, second = events[0], events[-1]
            if first[2] != second[2] and _BOTH_CONNECTOR.match(clause[first[1]:second[0]]):
                sides |= {"Medial", "Lateral"}
                continue
        before = [ev for ev in events if ev[1] <= m.start()]
        chosen = before[-1] if before else events[0]
        sides.add(chosen[2])
    return sides


def _mine_meniscus(text: str, side: str) -> MinedLabel:
    verdicts, evidence = set(), {}
    for clause in split_clauses(text):
        if side not in _meniscus_sides(clause):
            continue
        verdict = _classify_clause(clause, use_grades=False)
        if verdict:
            verdicts.add(verdict)
            evidence.setdefault(verdict, clause[:120])
    return _combine(verdicts, evidence)


def mine_medial_meniscus(text: str) -> MinedLabel:
    return _mine_meniscus(text, "Medial")


def mine_lateral_meniscus(text: str) -> MinedLabel:
    return _mine_meniscus(text, "Lateral")


_SYNOVITIS = re.compile(
    r"synovitis|synovialitis|sinovitis|sinovit\w*|συνοβ[ίι]τιδ\w*|синовит\w*|hoffitis|"
    r"synovi\w* (?:thicken|hypertroph|prolifer)\w*|hypertrophy of the synovium|"
    r"sinovi\w* (?:hipertrof|engrosa)\w*|verdikking\w* (?:van )?(?:het )?synovium"
)


def mine_synovitis(text: str) -> MinedLabel:
    verdicts, evidence = set(), {}
    for clause in split_clauses(text):
        matches = list(_SYNOVITIS.finditer(clause))
        if not matches:
            continue
        if all(is_negated_whole_word(clause, m.start(), m.end()) for m in matches):
            verdict = NEG
        elif _HEDGE.search(clause) or _MILD_GRADE.search(clause):
            verdict = WEAK
        else:
            verdict = POS
        verdicts.add(verdict)
        evidence.setdefault(verdict, clause[:120])
    return _combine(verdicts, evidence)


_MINERS = {
    "Fracture": mine_fracture,
    "Baker's": mine_bakers,
    "Effusion": mine_effusion,
    "Contusion": mine_contusion,
    "ACL": mine_acl,
    "MCL": mine_mcl,
    "Medial Meniscus": mine_medial_meniscus,
    "Lateral Meniscus": mine_lateral_meniscus,
    "Synovitis": mine_synovitis,
}


def mine_report(text: str) -> dict:
    """Mine all 12 labels from one report. Labels without an implemented
    miner return MinedLabel(value=None, evidence=None) -- an explicit
    "no signal" rather than a fabricated guess.
    """
    return {
        label: (_MINERS[label](text) if label in _MINERS else MinedLabel(None, None))
        for label in LABEL_COLUMNS
    }


def mining_agreement(reports, gold, columns=LABEL_COLUMNS):
    """Per-label confusion counts and accuracy against gold labels.

    reports: list[str]. gold: list[dict[str, int]], same length, one dict of
    {column: 0_or_1} per report. Only implemented labels are scored --
    n_scored counts how many rows had a non-None mined value for that
    column, so an unimplemented label reports n_scored=0 rather than a
    fabricated accuracy.
    """
    stats = {
        col: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n_scored": 0} for col in columns
    }
    for text, gold_row in zip(reports, gold):
        mined = mine_report(text)
        for col in columns:
            pred = mined[col].value
            if pred is None or col not in gold_row:
                continue
            truth = gold_row[col]
            s = stats[col]
            s["n_scored"] += 1
            if pred == 1 and truth == 1:
                s["tp"] += 1
            elif pred == 0 and truth == 0:
                s["tn"] += 1
            elif pred == 1 and truth == 0:
                s["fp"] += 1
            else:
                s["fn"] += 1
    for col, s in stats.items():
        s["accuracy"] = (s["tp"] + s["tn"]) / s["n_scored"] if s["n_scored"] else None
    return stats
