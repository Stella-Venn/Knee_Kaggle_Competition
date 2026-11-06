"""Step 3 of the OA weak-supervision pipeline: audit LF quality against the
gold dev subset ([redacted: private project notes], 2026-09-04 decision entry).

Coverage and accuracy-when-fired per LF, per compartment. Deliberately
separate from vote_matrix.py (which needs no labels) -- this is exactly the
point where gold labels get consulted, and per the holdout discipline this
must only ever run against the dev subset in
data/splits/oa_investigation_split.json, never the locked 18.
"""

from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS, LF_FACTORIES
from data.label_mining.weak_supervision.vote_matrix import TRANSLATE_PREFIX


def audit_lf(compartment, lf_name, reports, gold_values, translated_reports=None):
    """reports: list[str]. gold_values: list[int] (0 or 1), same length --
    the gold label for `compartment` on each report. Returns a dict with
    coverage (fraction not ABSTAIN), accuracy-when-fired, and the raw
    confusion counts.

    lf_name may be "translate_"-prefixed (e.g. "translate_keyword_proximity")
    -- the base LF (keyword_proximity) is applied to `translated_reports`
    (required in that case) instead of `reports`, matching how
    vote_matrix.build_vote_matrix handles the same distinction.
    """
    base_name = lf_name
    texts = reports
    if lf_name.startswith(TRANSLATE_PREFIX):
        base_name = lf_name[len(TRANSLATE_PREFIX):]
        if translated_reports is None:
            raise ValueError(f"{lf_name} needs translated_reports")
        texts = translated_reports

    lf = LF_FACTORIES[base_name](compartment)
    tp = tn = fp = fn = 0
    for text, gold in zip(texts, gold_values):
        vote = lf(text)
        if vote == 0:
            continue
        if vote == 1 and gold == 1:
            tp += 1
        elif vote == 1 and gold == 0:
            fp += 1
        elif vote == -1 and gold == 0:
            tn += 1
        elif vote == -1 and gold == 1:
            fn += 1
    n_scored = tp + tn + fp + fn
    n_total = len(texts)
    return {
        "compartment": compartment,
        "lf_name": lf_name,
        "n_total": n_total,
        "n_scored": n_scored,
        "coverage": n_scored / n_total if n_total else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n_scored if n_scored else None,
    }


def audit_all(reports, gold_by_compartment, translated_reports=None):
    """gold_by_compartment: dict[compartment] -> list[int], same order and
    length as `reports`. Returns one audit_lf() result per (compartment,
    lf_name) combination across COMPARTMENTS x (LF_FACTORIES plus their
    translate_-prefixed variants). translated_reports required if any
    translate_ column is to be scored (it will be, by default) -- omit only
    if you don't need those columns audited.
    """
    lf_names = list(LF_FACTORIES) + [TRANSLATE_PREFIX + n for n in LF_FACTORIES]
    return [
        audit_lf(compartment, lf_name, reports, gold_by_compartment[compartment], translated_reports)
        for compartment in COMPARTMENTS
        for lf_name in lf_names
    ]
