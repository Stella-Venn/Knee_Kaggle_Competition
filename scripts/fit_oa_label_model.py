"""Step 5 of the OA weak-supervision pipeline: fit the hand-rolled label
model over the full vote matrix and sanity-check it against the gold dev
subset (a quick gut-check, not the formal Step 7 validation -- that's its
own pass with per-language breakdowns per the original plan).

Merges known-redundant LF pairs first (Step 4's finding, [redacted: private project notes]
2026-09-09: keyword_proximity/translate_keyword_proximity and negation/
translate_negation agree 99-100% of the time they both fire -- same logic,
different input language, not independent corroboration) so the aggregator
sees one opinion per underlying check, not two correlated copies of it.

**Lateral OA investigation (2026-09-23) -- one real fix shipped, alpha
overrides NOT applied in the end, documented here so the next session
doesn't retread this.**

Fixed a real bug in `make_negation_lf` (it voted NEGATIVE off a negated
sub-finding without first checking whether a genuine, un-negated positive
existed elsewhere in the same report for that compartment -- study
`[redacted: study ID]`, same bug class already fixed in `make_keyword_proximity_lf` back
on 2026-09-04). That fix is unambiguous and correct on its own merits
(negation's per-case accuracy genuinely improves) and ships regardless of
anything below.

Then tried overriding LF alphas with `fit_label_model`'s new `fixed_alphas`
mechanism, using real gold-dev accuracy in place of EM's self-fit estimate,
and found it doesn't help -- three configurations were run end to end
through `build_training_labels.py` and compared:

1. **No override (EM only, current shipped state)**: EM converges BOTH
   `keyword_proximity` and `negation` to the alpha_clip ceiling (0.98) --
   this turns out true even in the pre-existing, unmodified code (verified
   via `git stash`), so the negation-LF bug fix alone changes nothing about
   Lateral OA's aggregate mined-label numbers. Result: 480 positive / 334
   negative training rows, 53.8% accuracy (n=13) against gold dev.
2. **Override only `negation`'s alpha to its gold-measured value (1.0,
   clipped to 0.98)**: byte-identical result to (1) -- EM was already
   estimating 0.98 for negation unsupervised, so this override is a no-op
   in practice, not a fix.
3. **Override both `keyword_proximity` (0.538, its real gold-dev accuracy)
   AND `negation`**: catastrophic, not an improvement -- with `negation`
   never voting POSITIVE by design, `keyword_proximity` is the ONLY signal
   source that can push Lateral OA's soft label toward positive, and
   correctly down-weighting it to ~54% makes it structurally too weak to
   ever cross the 0.85 threshold alone. Result: ZERO weak-supervision-
   derived Lateral OA positives (every positive in the resulting table
   traced back to the 58 gold rows, which are excluded from training folds
   entirely) -- not a smaller, cleaner positive signal, no positive signal
   at all for the compartment the model actually trains on.

**Conclusion: ship the negation-LF bug fix (real, unambiguous), do NOT
apply any alpha override for Lateral OA (both configurations tried are
no-ops or harmful)**, and treat Lateral OA's ~54% mined-label accuracy as an
unsolved problem, not a solved one. The `fixed_alphas`/`alpha_overrides`
plumbing stays in `label_model.py` (tested, may be useful for a genuinely
different mechanism later) but nothing calls it with real values right now
-- `alpha_overrides` here always ends up `{}`. Root cause is structural, not
a labeling-function bug: the 0.85/0.15 hard-threshold scheme needs a
compartment with only one imperfect positive-capable LF to either get a
better/additional positive-detecting LF, or move to soft-label/PU-learning
training for this specific compartment (module 8) instead of a hard
threshold -- flagged for a deliberate follow-up decision, not solved here.

**Update 2026-09-24 -- solved at the LF level instead, no override needed.**
Fixing cross-sentence attribution bleed, a window-edge miss, and adding
severity-aware voting (low-grade-only findings vote NEGATIVE) in `oa_lfs.py`
raised the merged Lateral OA signal from 53.8% to 76.9% on gold dev while
keeping 329 weak-supervision positives. The structural problem above was
real, but the fix was a more accurate LF, not a lower alpha.

Usage: python scripts/fit_oa_label_model.py
Rebuilds the vote matrix if data/processed/oa_vote_matrix.npz is missing.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.weak_supervision.label_model import fit_oa_label_models
from data.label_mining.weak_supervision.redundancy import merge_translate_duplicates
from data.label_mining.weak_supervision.vote_matrix import VOTE_COLUMNS, build_vote_matrix
from translate_reports import load_or_build_translations


def main():
    cache_path = ROOT / "data" / "processed" / "oa_vote_matrix.npz"
    train_csv = ROOT / "data" / "raw" / "train.csv"

    with open(train_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    label_cols = header[2:]
    study_ids = [row[0] for row in rows]
    reports = [row[1] for row in rows]

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["study_ids"]) == study_ids and cached["matrix"].shape[1] == len(VOTE_COLUMNS):
            matrix = cached["matrix"].astype(int)
            print(f"Loaded cached vote matrix {matrix.shape}")
        else:
            print("Cache doesn't match current train.csv/VOTE_COLUMNS, rebuilding...")
            translated_reports = load_or_build_translations(study_ids, reports)
            matrix, _ = build_vote_matrix(reports, translated_reports)
    else:
        print("No cached vote matrix, building...")
        translated_reports = load_or_build_translations(study_ids, reports)
        matrix, _ = build_vote_matrix(reports, translated_reports)

    merged_matrix, merged_columns = merge_translate_duplicates(matrix, VOTE_COLUMNS)
    print(f"Merged {matrix.shape[1]} columns -> {merged_matrix.shape[1]} "
          f"(collapsed known-redundant translate_* pairs)")

    # Gold-dev accuracy for Lateral OA's two LFs -- diagnostic only, printed
    # for visibility, NOT applied as an alpha override. See module docstring
    # ("Lateral OA investigation") for why: both override configurations
    # tried (negation-only, both) turned out to be either a no-op or
    # actively harmful, so `alpha_overrides` stays empty and EM's own
    # unsupervised fit is what actually ships.
    with open(ROOT / "data" / "splits" / "oa_investigation_split.json") as f:
        dev_ids_for_override = set(json.load(f)["dev"])
    row_by_id = {row[0]: row for row in rows}
    lateral_oa_col = label_cols.index("Lateral OA")
    for lf_name in ("keyword_proximity", "negation"):
        idx = merged_columns.index(("Lateral OA", lf_name))
        tp = tn = fp = fn = 0
        for i, sid in enumerate(study_ids):
            if sid not in dev_ids_for_override:
                continue
            gold_str = row_by_id[sid][2 + lateral_oa_col]
            if gold_str == "":
                continue
            gold, vote = int(gold_str), merged_matrix[i, idx]
            if vote == 0:
                continue
            tp += vote == 1 and gold == 1
            fp += vote == 1 and gold == 0
            tn += vote == -1 and gold == 0
            fn += vote == -1 and gold == 1
        n_scored = tp + tn + fp + fn
        accuracy = (tp + tn) / n_scored if n_scored else None
        acc_str = f"{accuracy:.1%}" if accuracy is not None else "n/a"
        print(f"Lateral OA {lf_name} gold-dev accuracy: {acc_str} (n_scored={n_scored}) "
              f"-- diagnostic only, not applied as an override")

    alpha_overrides = {}  # deliberately empty -- see module docstring
    results = fit_oa_label_models(merged_matrix, merged_columns, alpha_overrides=alpha_overrides)

    print("\nLearned per-LF reliability (pure unsupervised EM, no gold labels used to fit "
          "this -- see module docstring for why Lateral OA has no alpha override):")
    for compartment, r in results.items():
        print(f"  {compartment}: {r['lf_reliability']}")

    # Sanity check against gold dev subset -- not the formal Step 7 pass.
    with open(ROOT / "data" / "splits" / "oa_investigation_split.json") as f:
        split = json.load(f)
    dev_ids = set(split["dev"])
    study_to_row = {row[0]: row for row in rows}
    dev_rows = [study_to_row[sid] for sid in study_ids if sid in dev_ids and study_to_row[sid][2] != ""]
    dev_idx = [i for i, sid in enumerate(study_ids) if sid in dev_ids and study_to_row[sid][2] != ""]

    print(f"\nSanity check against {len(dev_rows)} gold dev-subset rows "
          f"(threshold soft label at 0.5, not the formal Step 7 pass):")
    for compartment in results:
        col = label_cols.index(compartment)
        soft = results[compartment]["soft_labels"][dev_idx]
        gold = np.array([int(dev_rows[i][2 + col]) for i in range(len(dev_rows))])
        confident = (soft > 0.5) | (soft < 0.5)
        no_signal = (soft == 0.5).sum()
        pred = (soft > 0.5).astype(int)
        scored = soft != 0.5
        acc = (pred[scored] == gold[scored]).mean() if scored.any() else None
        acc_str = f"{acc:.1%}" if acc is not None else "n/a"
        print(f"  {compartment:<12} n_scored={scored.sum():>3}/{len(dev_rows)} "
              f"no_signal={no_signal:>3} accuracy={acc_str}")

    out_path = ROOT / "data" / "processed" / "oa_soft_labels.npz"
    np.savez(
        out_path,
        study_ids=np.array(study_ids, dtype=object),
        **{f"{c}_soft": r["soft_labels"] for c, r in results.items()},
    )
    print(f"\nSaved soft labels to {out_path}")


if __name__ == "__main__":
    main()
