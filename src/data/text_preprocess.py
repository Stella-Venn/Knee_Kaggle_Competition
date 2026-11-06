"""Text preprocessing module (gameplan module 5).

Per the gameplan's own framing, this is "likely mostly built already as part
of [label mining], so this may just be a refinement pass" -- and that's what
this module is: the generic, label-independent text utilities that had been
living inside `label_mining/rule_based.py` (clause splitting, negation
detection, language detection) are consolidated here so they have one home
instead of being duplicated or reached into across modules. Nothing here is
new logic; behavior is unchanged from where each piece came from.

- `split_clauses`, `is_negated`, `is_negated_forward`, `is_negated_whole_word`
  moved from `rule_based.py` (still re-exported there for anything importing
  the old names -- `weak_supervision/oa_lfs.py` imports `NEGATION_MARKERS`,
  `is_negated`, `is_negated_forward` from `rule_based`, and that keeps working).
- `detect_language` consolidates a langdetect-with-fallback block that was
  duplicated between `scripts/translate_reports.py` and
  `scripts/audit_translate_then_match.py`.

Two negation strategies live here side by side because they solve different
problems, not because one supersedes the other -- see `is_negated` and
`is_negated_whole_word`'s docstrings for when each applies.
"""

import re

MICRO_SIGN = "µ"  # some Greek reports use the micro sign instead of mu

# Not underscore-prefixed: reused by label_mining.weak_supervision.oa_lfs
# (imported there via label_mining.rule_based, which re-exports this).
NEGATION_MARKERS = [
    "no ", "not ", "without", "negative for",
    "sin ", "no hay",
    "yok", "değil", "degil",
    "geen", "niet",
    "kein", "keine", "ohne", "nicht",
    "nema", "bez ",
    "δεν", "χωρίς",
    "без", "няма",
    "pas de", "sans", "aucun",
]

NEGATION_WINDOW = 30  # characters to look back from a term match


def is_negated(text_lower: str, match_start: int, window: int = NEGATION_WINDOW) -> bool:
    """Substring-based negation check (not whole-word -- see
    `is_negated_whole_word` for that). Doesn't cross a sentence boundary
    (period) -- a negation word in a PRECEDING, unrelated sentence can
    otherwise fall inside the window and wrongly negate this match. Found
    during OA weak-supervision work (2026-09-04): "[redacted: report text]"
    [redacted: quoted text from a restricted clinical dataset]
    present" using the previous sentence's "no". All of rule_based.py's
    term-list miners (Fracture/Baker's/Effusion/Contusion) negate within the
    same sentence in their validated test cases, so this tightening doesn't
    regress them -- re-verified against the full gold 58 after this change
    (see [redacted: private project notes]).
    """
    start = max(0, match_start - window)
    period_before = text_lower.rfind(".", start, match_start)
    if period_before != -1:
        start = period_before + 1
    context = text_lower[start:match_start]
    return any(neg in context for neg in NEGATION_MARKERS)


# Structured templates often name a finding category, then negate it on the
# [redacted: quoted text from a restricted clinical dataset]
# negation comes AFTER the term, which is_negated (backward-only) can't see.
# Found in oa_lfs.py development (study [redacted: study ID]) but this convention isn't
# OA-specific, so it lives here for any miner that needs it.
FORWARD_NEGATION_MARKERS = [
    "none", "no ", "nil", "negative", "not seen", "unremarkable",
    "ninguno", "ninguna", "yok", "geen", "keine",
]
FORWARD_NEGATION_WINDOW = 20


def is_negated_forward(text_lower: str, match_end: int, window: int = FORWARD_NEGATION_WINDOW) -> bool:
    """Post-posed negation, for the "finding: None." template convention --
    see the module note above FORWARD_NEGATION_MARKERS.

    Strips a trailing plural/punctuation tail too -- "[redacted: report text]"
    leaves an "s" right after the matched term, before the colon, which a
    plain " :\\n\\t" strip doesn't reach past (confirmed missing study
    [redacted: study ID]'s "Degenerative changes:\\nNone." during Step 1 gap-fixing).

    Deliberately does NOT strip "." -- a period ends the sentence, and text
    after it is a new, unrelated statement, not a negation of this match.
    Stripping past periods caused a real false negative during Step 3
    [redacted: quoted text from a restricted clinical dataset]
    the chondrosis finding using an unrelated fracture sentence's "No".
    """
    context = text_lower[match_end:match_end + window].lstrip("s :\n\t")
    return any(context.startswith(m) for m in FORWARD_NEGATION_MARKERS)


def lower_report(text: str) -> str:
    """Lowercase report text for substring matching, with the micro sign
    folded to Greek mu first. 305 of the 321 Greek reports write mu as
    U+00B5 (MICRO SIGN) in some words, so any Greek term containing mu
    ("μώλωπ", "μηνίσκ", "διαμέρισμα") silently failed to match there
    (2026-09-24). Length-preserving, so match indices stay valid.
    """
    return text.replace(MICRO_SIGN, "μ").lower()


def split_clauses(text: str):
    """Lowercased clauses. Newlines inside a sentence (wrapped report lines)
    are joined; a newline after . ; ! ? or before a bullet is a boundary. A
    line ending in ':' is a header and joins the line after it. Splitting on
    [.;!?] is allowed with no following space ("...izlenmiştir.Lateral") but
    not before a digit ("3.2 cm").
    """
    text = text.replace(MICRO_SIGN, "μ")  # micro sign -> Greek mu
    text = re.sub(r"(?<![.;!?])\n(?!\s*[*>•\-–]|\n)", " ", text)
    parts = re.split(r"[.;!?]+(?=\s|$|[A-ZÀ-ÝΑ-ΩА-ЯÇĞİÖŞÜ])|\n+", text)
    return [p.lower().strip() for p in parts if p.strip()]


# Whole-word negation, for clause-scoped miners that need to avoid the
# false match a substring check like `is_negated` risks -- e.g. Spanish
# "[redacted: report text]" contains "no " as a substring of "interno", which
# `is_negated` would wrongly read as a negation marker.
_NEGATION_WORDS = re.compile(
    r"(?<![^\W\d_])(?:no|not|without|negative for|sin|no hay|yok\w*|değil|degil|geen|niet|"
    r"kein\w*|ohne|nicht|nema|bez|δεν|χωρίς|без|няма|pas de|sans|aucun\w*)(?![^\W\d_])"
)
_NEGATION_TAIL = ("etmiyor", "yok", "izlenmedi", "izlenmemiş", "saptanmadı", "not seen", "none")
_NEGATION_LOOKBACK = 35


def is_negated_whole_word(clause: str, start: int, end: int) -> bool:
    """Whether the match clause[start:end] is negated, using whole-word
    negation markers (unlike `is_negated`, a substring check) and looking
    back only to the last comma/colon (a negation belongs to its own
    phrase). Also checks a short forward tail for post-posed negation
    (Turkish verb-final negation, "yırtık ... izlenmemiştir").
    """
    context = clause[max(0, start - _NEGATION_LOOKBACK):start]
    context = re.split(r"[,:]", context)[-1]
    if _NEGATION_WORDS.search(context):
        return True
    tail = clause[end:end + 20].lstrip("s :\t")
    return tail.startswith(_NEGATION_TAIL)


# Languages actually found in this dataset's reports (report_translations.json,
# built by scripts/translate_reports.py): en 39%, es 16%, tr 12%, hr 9%, el 7%,
# de 6%, bg 5%, nl 4%, fr 2%. detect_language falls back to "en" on any
# detection failure (e.g. a report too short/ambiguous for langdetect) rather
# than raising, since a translation pipeline can't stop for one row.
def detect_language(text: str) -> str:
    """2-letter langdetect code for `text`, "en" on any detection failure.
    Seeds langdetect's DetectorFactory for reproducibility (it's otherwise
    randomized) -- safe to call repeatedly, langdetect no-ops a reseed.
    """
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    try:
        return detect(text)
    except Exception:
        return "en"
