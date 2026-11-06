"""Idea 5 ([redacted: private project notes], 2026-09-25): blend our mined labels with the public LLM
report labels into one table in the LLM table's own format, so the DINOv2 fork reads it
unchanged (upstream's find_label_table() takes any report_labels*.csv with the
StudyInstanceUID + <target> + <target>__conf columns).

Upstream uses <target> as the soft target and 0.25 + 0.75 * <target>__conf as the loss
weight. The LLM table's values: NO 0.08 (conf 0.85), UNK 0.28 (conf 0.05), YES 0.68-0.94
graded (conf 0.95).

One uniform rule for every label -- deliberately NOT tuned per label on the 40 gold dev
studies, where our rule-based labels were themselves tuned and would look better than
they are:
  LLM YES/NO + ours agree      -> LLM value, conf max(LLM conf, 0.95 if YES else 0.85)
  LLM YES/NO + ours disagree   -> midpoint of the two values, conf 0.35 (contested)
  LLM UNK + ours from a rule   -> ours (pos 0.80 / neg 0.08), conf 0.70
  LLM UNK + ours "never mentioned" negative (OA/MCL only) -> 0.15, conf 0.40
  ours masked (or Synovitis's positive-unlabeled fill) -> LLM unchanged
Synovitis's unmentioned fill is treated as no information: a third of those studies are
really positive (gold dev).

Output: data/processed/report_labels_blend.csv (+ a gold-dev comparison printed).
Usage: python scripts/build_blended_labels.py --llm <report_labels_v2.csv>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.final_labels import (  # noqa: E402
    MASKED, SOURCE_UNMENTIONED, assemble_training_labels, rule_label_matrix,
    threshold_soft_labels, unmentioned_negative_matrix,
)
from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS  # noqa: E402
from utils.constants import LABEL_COLUMNS  # noqa: E402

OURS_POS, OURS_NEG, UNMENTIONED_NEG = 0.80, 0.08, 0.15
CONF_AGREE_POS, CONF_AGREE_NEG, CONF_CONTESTED, CONF_OURS_RULE, CONF_UNMENTIONED = 0.95, 0.85, 0.35, 0.70, 0.40


def blend_cell(llm_v, llm_conf, llm_verdict, ours, ours_source, column):
    """(value, conf) for one (study, label). `ours` in {1, 0, MASKED}."""
    unmentioned = ours_source == SOURCE_UNMENTIONED
    if ours == MASKED or (unmentioned and column == "Synovitis"):
        return llm_v, llm_conf
    if llm_verdict == "UNK":
        if unmentioned:
            return UNMENTIONED_NEG, CONF_UNMENTIONED
        return (OURS_POS if ours == 1 else OURS_NEG), CONF_OURS_RULE
    llm_pos = llm_verdict == "YES"
    if llm_pos == (ours == 1):
        return llm_v, max(llm_conf, CONF_AGREE_POS if llm_pos else CONF_AGREE_NEG)
    ours_v = OURS_POS if ours == 1 else (UNMENTIONED_NEG if unmentioned else OURS_NEG)
    return (llm_v + ours_v) / 2, CONF_CONTESTED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="path to pilkwang's report_labels_v2.csv")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "report_labels_blend.csv"))
    args = ap.parse_args()

    llm = pd.read_csv(args.llm).set_index("StudyInstanceUID")
    # Our MINED labels for every study, with no gold override -- the same assembly as
    # build_training_labels.py minus the gold layer, so gold studies carry what the
    # miners say about them (needed for an honest-as-possible gold dev comparison; the
    # table's gold rows are never read in training, upstream substitutes gold there).
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        tr_rows = list(csv.reader(f))[1:]
    ids = [r[0] for r in tr_rows]
    reports = [r[1] for r in tr_rows]
    rule = rule_label_matrix(reports)
    soft = np.load(ROOT / "data" / "processed" / "oa_soft_labels.npz", allow_pickle=True)
    assert list(soft["study_ids"]) == ids, "oa_soft_labels.npz out of sync with train.csv"
    weak = np.full(rule.shape, MASKED, dtype=np.int8)
    for c in COMPARTMENTS:
        weak[:, LABEL_COLUMNS.index(c)] = threshold_soft_labels(soft[f"{c}_soft"], 0.85, 0.15)
    no_gold = np.full(rule.shape, MASKED, dtype=np.int8)
    labels, source = assemble_training_labels(no_gold, rule, weak, unmentioned_negative_matrix(reports))

    rows, n_changed, n_contested = [], 0, 0
    for i, sid in enumerate(ids):
        row = {"StudyInstanceUID": sid}
        for j, c in enumerate(LABEL_COLUMNS):
            if sid in llm.index:
                lv, lc, lverd = float(llm.at[sid, c]), float(llm.at[sid, c + "__conf"]), llm.at[sid, c + "__verdict"]
            else:
                lv, lc, lverd = 0.28, 0.05, "UNK"
            ours = labels[i, j]
            v, cf = blend_cell(lv, lc, lverd, ours, source[i, j], c)
            n_changed += (v, cf) != (lv, lc)
            n_contested += cf == CONF_CONTESTED
            row[c], row[c + "__conf"] = round(v, 4), round(cf, 4)
            row[c + "__verdict"] = "YES" if v >= 0.5 else ("NO" if cf >= 0.5 and v < 0.2 else "UNK")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    n = len(rows) * len(LABEL_COLUMNS)
    print(f"wrote {args.out}: {len(rows)} studies; {n_changed}/{n} cells differ from the LLM table "
          f"({n_contested} contested)")

    # gold dev check (label quality as a ranking score vs gold) -- optimistic for ours
    from sklearn.metrics import roc_auc_score
    tr = pd.read_csv(ROOT / "data" / "raw" / "train.csv").set_index("StudyInstanceUID")
    dev = set(json.load(open(ROOT / "data" / "splits" / "oa_investigation_split.json"))["dev"])
    dev = [s for s in dev if s in tr.index and not pd.isna(tr.at[s, "ACL"])]
    b = out.set_index("StudyInstanceUID")
    print(f"gold dev (n={len(dev)}), label table as a ranking score vs gold:")
    print(f"{'label':<17}{'LLM':>7}{'blend':>7}")
    a_l, a_b = [], []
    for c in LABEL_COLUMNS:
        g = tr.loc[dev, c].astype(int).to_numpy()
        pl = np.array([llm.at[s, c] if s in llm.index else 0.28 for s in dev], float)
        # the blend for a gold study uses our NON-gold mined labels (see above)
        pb = b.loc[dev, c].to_numpy(float)
        a_l.append(roc_auc_score(g, pl)); a_b.append(roc_auc_score(g, pb))
        print(f"{c:<17}{a_l[-1]:>7.3f}{a_b[-1]:>7.3f}")
    print(f"{'MACRO':<17}{np.mean(a_l):>7.3f}{np.mean(a_b):>7.3f}   "
          f"(ours was tuned on these studies: treat the blend's number as optimistic)")


if __name__ == "__main__":
    main()
