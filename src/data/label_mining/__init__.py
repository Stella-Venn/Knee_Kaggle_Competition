"""Label mining: rule-based miners (`rule_based`) plus, for labels that
don't yield to a single hand-curated rule, a weak-supervision labeling
function ensemble (`weak_supervision`) -- see [redacted: private project notes]'s 2026-09-04
decision entry for why OA compartment labels moved to the latter.
"""

from data.label_mining.rule_based import (
    LABEL_COLUMNS,
    MinedLabel,
    mine_bakers,
    mine_contusion,
    mine_effusion,
    mine_fracture,
    mine_report,
    mining_agreement,
)

__all__ = [
    "LABEL_COLUMNS",
    "MinedLabel",
    "mine_bakers",
    "mine_contusion",
    "mine_effusion",
    "mine_fracture",
    "mine_report",
    "mining_agreement",
]
