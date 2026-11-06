"""Unit tests for src/data/label_mining/weak_supervision/oa_lfs.py.

All report text here is SYNTHETIC. It was written to mirror the structure of real
radiology reports (section layout, wrapped lines, distances between a compartment
name and a finding), because proximity-based rules are sensitive to exactly that.
Real report text is part of a restricted dataset and is never committed.

Several assertions below document ABSTAIN results that are correct but for an
incomplete reason (a term-list or window-width gap, not principled handling) --
flagged explicitly so a future reader doesn't mistake "abstains here" for
"understands this case."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.rule_based import is_negated_forward
from data.label_mining.weak_supervision.oa_lfs import (
    ABSTAIN,
    NEGATIVE,
    POSITIVE,
    generic_no_compartment_flag,
    make_hedge_flag_lf,
    make_keyword_proximity_lf,
    make_negation_lf,
    run_oa_lfs,
)

# Templated English report: severe medial and PF findings, mild lateral only.
_REPORT_A = (
    "Study: MRI of the right knee, no contrast.\nReason: chronic pain on stairs.\n"
    "Prior imaging: none available.\n\nMETHOD:\nThree-plane sequences at 1.5 T "
    "with a dedicated coil and the routine non-contrast protocol.\n\nFINDINGS:\n"
    "BONES:\nNormal alignment. Negative for acute fracture or marrow contusion. "
    "Marrow signal unremarkable. No destructive lesion. Marginal osteophytes in all "
    "compartments\n\nJOINT:\nSmall effusion. No loose bodies. Negative for "
    "Baker cyst seen.\n\nMEDIAL COMPARTMENT:\nMedial meniscus: Complete "
    "body and root tear, marked extrusion.\nMedial compartment "
    "cartilage: Full thickness cartilage loss over the medial femoral condyle and "
    "the medial tibial plateau with subchondral cysts.\n\nLATERAL COMPARTMENT:\n"
    "Lateral meniscus: Signal change from degeneration, no surface "
    "tear.\nLateral cartilage: Mild cartilage irregularity with "
    "fissures.\n\nPATELLOFEMORAL COMPARTMENT:\nPatellofemoral surfaces: "
    "Severe cartilage loss on the medial patellar facet. Severe cartilage "
    "loss on the lateral trochlea."
)

# Paragraph-style English report with no cartilage findings anywhere.
_REPORT_B = (
    " In the medial compartment, the meniscus looks intact on all sequences and "
    "there is no focal chondrosis or chondral damage. \n  \n The lateral compartment, "
    "a complex tear with radial and vertical parts involves the rear horn of the "
    "lateral meniscus. Again no focal chondrosis or chondral damage. \n  \n The "
    "patellofemoral joint, patella and trochlea look smooth without "
    "chondral lesion. The MPFL looks fine."
)

# [redacted: quoted text from a restricted clinical dataset]
_REPORT_C = (
    "FINDINGS:\n\nFractures:\nNondisplaced avulsion fracture of the tibial spine at "
    "the ACL insertion, reaching the front root of the lateral meniscus and the rear "
    "roots of both menisci. The fracture crosses the central medial tibial plateau "
    "with 2 mm of articular depression.\nSmall avulsion of the lateral tibial rim "
    "[Segond fracture].\nMinor osteochondral impaction at the weightbearing part "
    "of lateral femoral condyle.\nBone marrow edema accompanies these fractures.\n\n"
    "Joint alignment:\nSlight widening of the medial femorotibial joint space. "
    "Alignment otherwise preserved.\n\nDegenerative changes:\nNone."
)


def test_medial_keyword_proximity_positive():
    lf = make_keyword_proximity_lf("Medial OA")
    assert lf(_REPORT_A) == POSITIVE


def test_pf_keyword_proximity_positive():
    lf = make_keyword_proximity_lf("PF OA")
    assert lf(_REPORT_A) == POSITIVE


def test_lateral_meniscus_degeneration_alone_does_not_false_positive():
    # [redacted: quoted text from a restricted clinical dataset]
    # meniscus finding, not cartilage/OA -- tested in isolation (just that
    # sentence) to confirm the meniscus/ligament exclusion still holds
    # without the report's later genuine lateral cartilage finding (see
    # test_lateral_cartilage_finding_now_detected_but_disagrees_with_gold)
    # pulling in a POSITIVE vote via the declaration-scope mechanism.
    text = (
        "LATERAL COMPARTMENT:\nLateral meniscus: Signal change from "
        "degeneration, no surface tear."
    )
    lf = make_keyword_proximity_lf("Lateral OA")
    assert lf(text) == ABSTAIN


def test_lateral_mild_only_finding_votes_negative_matching_gold():
    # [redacted: quoted text from a restricted clinical dataset]
    # fissuring." -- a real finding (found via _AMBIGUOUS_OA_TERMS' cartilage-
    # companion check + declaration scope), which used to vote POSITIVE
    # against gold Lateral OA=0 and was documented as an unexplained
    # "correctly parsed, disagrees with gold" case. Explained 2026-09-24:
    # the gold annotators don't call OA on low-grade-only chondral change
    # (same report: full-thickness medial and high-grade PF findings are
    # gold 1). Low-grade-only now votes LOW_GRADE_ONLY_VOTE (NEGATIVE).
    lf = make_keyword_proximity_lf("Lateral OA")
    assert lf(_REPORT_A) == NEGATIVE


def test_pf_negation_close_proximity():
    # [redacted: quoted text from a restricted clinical dataset]
    # are close together in one clause, well within the proximity window.
    lf = make_negation_lf("PF OA")
    assert lf(_REPORT_B) == NEGATIVE


def test_lateral_negation_within_window():
    lf = make_negation_lf("Lateral OA")
    assert lf(_REPORT_B) == NEGATIVE


def test_medial_negation_reaches_paragraph_style_attribution():
    # [redacted: quoted text from a restricted clinical dataset]
    # ~90 characters later -- outside the old MATCH_WINDOW=60, so this used
    # to ABSTAIN despite a clear negated finding. Fixed via
    # _declared_compartment_before: a compartment declaration's scope now
    # extends up to _DECLARATION_MAX_SCOPE chars, not just the tight window.
    lf = make_negation_lf("Medial OA")
    assert lf(_REPORT_B) == NEGATIVE


def test_osteochondral_is_excluded_not_a_genuine_oa_finding():
    # [redacted: quoted text from a restricted clinical dataset]
    # the OCD/cartilage-surface entity excluded from Fracture for the same
    # reason (rule_based.py's _FRACTURE_EXCLUDE). Correctly does not fire
    # POSITIVE for Lateral OA.
    lf = make_keyword_proximity_lf("Lateral OA")
    assert lf(_REPORT_C) == ABSTAIN


def test_forward_negation_catches_category_header_none_pattern():
    # [redacted: quoted text from a restricted clinical dataset]
    # AFTER the term, a structured-template convention is_negated (backward-
    # only) can't see. Fixed via rule_based.is_negated_forward, reusable
    # beyond OA (the same convention appears for other categories too, e.g.
    # study [redacted: study ID] also has "[redacted: report text]").
    text = "Findings: Degenerative changes:\nNone. Otherwise unremarkable."
    assert is_negated_forward(text.lower(), text.lower().index("degenerative change") + len("degenerative change"))


def test_degenerative_changes_none_still_abstains_with_no_compartment_context():
    # Confirms the forward-negation fix didn't just move the gap: this case
    # [redacted: quoted text from a restricted clinical dataset]
    # no compartment attribution nearby at all (the preceding sentence's
    # [redacted: quoted text from a restricted clinical dataset]
    # "[redacted: report text]", not this one) -- not because negation-after-header
    # goes undetected (see test_forward_negation_catches_category_header_
    # none_pattern for that being fixed).
    lf = make_keyword_proximity_lf("Medial OA")
    assert lf(_REPORT_C) == ABSTAIN


def test_run_oa_lfs_returns_all_three_compartments():
    # Lateral OA is NEGATIVE here (mild-only finding) -- see
    # test_lateral_mild_only_finding_votes_negative_matching_gold.
    result = run_oa_lfs(_REPORT_A)
    assert set(result.keys()) == {"Medial OA", "Lateral OA", "PF OA"}
    assert result["Medial OA"]["keyword_proximity"] == POSITIVE
    assert result["Lateral OA"]["keyword_proximity"] == NEGATIVE
    assert result["PF OA"]["keyword_proximity"] == POSITIVE


def test_hedge_flag_is_diagnostic_not_a_vote():
    # make_hedge_flag_lf returns a bool, not a vote -- confirms it's wired
    # as the diagnostic-only function the module docstring describes.
    flag = make_hedge_flag_lf("Medial OA")
    result = flag(_REPORT_A)
    assert isinstance(result, bool)


def test_generic_no_compartment_flag_is_bool():
    result = generic_no_compartment_flag(_REPORT_A)
    assert isinstance(result, bool)


# Lateral paragraph with a real finding after a negated one. Real bug
# found while investigating whether Lateral OA's false positives were near
# meniscus/ligament/condyle mentions (they mostly weren't, but this
# [redacted: quoted text from a restricted clinical dataset]
# BEFORE "[redacted: report text]" (genuine positive) for the same
# compartment -- the old logic returned NEGATIVE on the first match found
# and never reached the real positive finding.
_REPORT_D_LATERAL_NEGATED_THEN_POSITIVE = (
    "The lateral compartment, a small horizontal tear involves the front horn of "
    "the lateral meniscus. There is a focal high grade chondral defect at the "
    "central lateral tibial plateau (about 1 x 1 cm). "
    "There is no subchondral edema. Osteophytes noted."
)


def test_genuine_positive_wins_over_earlier_negated_subfinding():
    lf = make_keyword_proximity_lf("Lateral OA")
    assert lf(_REPORT_D_LATERAL_NEGATED_THEN_POSITIVE) == POSITIVE


def test_subchondral_fracture_not_read_as_oa_finding():
    # "[redacted: report text]" (a Fracture-family finding) must not trigger
    # via the bare "chondral" stem -- confirmed against study [redacted: study ID].
    text = (
        "Findings favor a subchondral fracture of the lateral femoral condyle. "
        "Contusion of the distal femur and proximal tibia."
    )
    lf = make_keyword_proximity_lf("Lateral OA")
    assert lf(text) == ABSTAIN


def test_forward_negation_does_not_cross_sentence_boundary():
    # Real bug found while fixing the above: the original forward-negation
    # fix stripped past periods too, so an unrelated "No X" in the NEXT
    # sentence wrongly negated the current finding. Confirmed against study
    # [redacted: study ID].
    text = "due to overlying chondrosis. No new fracture or bone bruise."
    end = text.lower().index("chondrosis") + len("chondrosis")
    assert is_negated_forward(text.lower(), end) is False


# A genuine, un-negated lateral finding sits right next to a negated sub-finding
# for the SAME compartment. make_negation_lf used to vote NEGATIVE off the negated
# sub-finding without checking whether a genuine positive existed nearby -- the same
# "return on first match" bug class already fixed in make_keyword_proximity_lf.
_REPORT_E = (
    " In the medial compartment, a radial tear involves the rear root of the medial "
    "meniscus, with a displaced fragment in the gutter. There is mixed low and high "
    "grade chondrosis (over half of cartilage thickness) along the medial femoral "
    "condyle and tibial plateau, with subchondral edema and osteophytes. "
    "\n  \n The lateral compartment, a small horizontal tear involves the "
    "front horn of the lateral meniscus. There is a focal high grade chondral defect "
    "at the central lateral tibial plateau (about 1 x 1 cm). There is no subchondral "
    "edema. Osteophytes noted.  \n  \n The patellofemoral joint, there is mixed low "
    "and high grade chondrosis (over half of cartilage thickness) along both trochleas "
    "and both patellar facets, with subchondral edema and osteophytes. The "
    "MPFL looks fine.  \n  \n In the notch, the "
    "cruciate ligaments look intact. \n  \n The collateral ligaments look intact. "
    "\n  \n The extensor mechanism is normal.  \n A moderate effusion is seen. A 2 cm "
    "baker's cyst sits behind the knee. Osteoarthritis in all three compartments on "
    "the basis of chondrosis, edema and osteophytes. \n  Radial tear at the rear root "
    "of the medial meniscus. \n  Possible small horizontal tear at the front horn of "
    "the lateral meniscus. "
)


def test_negation_does_not_override_a_genuine_positive_elsewhere():
    # The bug: "[redacted: report text]" (negated) sits right before a genuine,
    # un-negated "[redacted: report text]" finding for the same compartment. The old
    # logic returned NEGATIVE on the first negated match found; the fix
    # requires checking the WHOLE text for a genuine positive first.
    lf = make_negation_lf("Lateral OA")
    assert lf(_REPORT_E) == ABSTAIN
    # keyword_proximity (already correct) should still catch the genuine positive.
    kp = make_keyword_proximity_lf("Lateral OA")
    assert kp(_REPORT_E) == POSITIVE


def test_negation_still_fires_when_no_genuine_positive_exists():
    # Regression guard: the fix must not make this LF stop firing NEGATIVE
    # entirely -- a compartment that's ONLY described as negated (no genuine
    # positive anywhere) should still vote NEGATIVE, same as before the fix.
    text = "In the medial compartment, there is no chondral defect and no osteophytes."
    lf = make_negation_lf("Medial OA")
    assert lf(text) == NEGATIVE


# [redacted: quoted text from a restricted clinical dataset]
# Exercises all three 2026-09-24 fixes at once -- see the tests below.
_REPORT_F = (
    'Historia clínica:\nDolor crónico.\nResultados:\nLigamentos sin alteraciones.\n'
    'Menisco medial íntegro.\nDesgarro complejo del cuerpo y raíz posterior del\n'
    'menisco lateral con extrusión. Edema óseo en el platillo tibial lateral.\n'
    'Pérdida condral focal en la zona de carga del compartimento\nfemorotibial lateral. '
    'Cambios óseos mínimos. Condropatía leve superficial del compartimento\n'
    'femorotibial medial y de la tróclea.\nDefecto condral parcial de la faceta lateral '
    'de la rótula.\nDerrame articular moderado.\nConclusión:\nDesgarro complejo del '
    'menisco lateral.\nCondropatía con lesiones grado 3 y 4 de compartimento femorotibial '
    'lateral.\nCondropatía leve superficial de compartimento femorotibial medial. Condropatía\n'
    'con lesiones grado dos y tres de la rótula.\nDerrame articular moderado.'
)


from data.label_mining.weak_supervision.oa_lfs import (  # noqa: E402
    _has_inline_term,
    _is_low_grade_only,
    _local_window,
)


def test_local_window_stops_at_sentence_boundary():
    # [redacted: study ID]'s bug in miniature: a compartment term in the PREVIOUS
    # sentence must not attribute an unrelated finding in this one.
    text = "ulceras del compartimento femorotibial lateral. condropatía de rótula leve."
    idx = text.index("condropatía")
    assert "lateral" not in _local_window(text, idx)


def test_local_window_does_not_split_on_newline_or_ordinal_grade():
    # Croatian hard-wraps lines mid-sentence and writes grades as
    # "[redacted: report text]" -- neither is a sentence end.
    text = "hondromalacija ii. stupnja\nlateralnog tibijalnog dijela."
    assert "lateralnog tibijalnog dijela" in _local_window(text, 0)


def test_leading_space_term_anchors_in_its_own_sentence():
    # " oa " starts with the space inside ". " -- must not reach back into
    # [redacted: quoted text from a restricted clinical dataset]
    text = "hrskavice medijalnog kondila. oa nalaz obostrano."
    idx = text.index(" oa ")
    assert "medijalnog" not in _local_window(text, idx)


def test_long_compartment_term_straddling_window_edge_still_attributes():
    text = "condropatía con lesiones grado 3 y 4 de compartimento femorotibial lateral."
    assert _has_inline_term(text, 0, ["compartimento femorotibial lateral"])


def test_low_grade_only_classification():
    assert _is_low_grade_only("mild cartilage irregularity with fissures")
    assert _is_low_grade_only("condropatía leve superficial de compartimento")
    assert _is_low_grade_only("hondromalacija ii° zone lateralnog")
    assert not _is_low_grade_only("condropatía con grado 2 y 3 de la zona")
    assert not _is_low_grade_only("condropatía con grado dos y tres de la rótula")
    assert not _is_low_grade_only("mixed low grade and high grade chondrosis")
    assert not _is_low_grade_only("mild chondrosis. osteophytes noted")
    assert not _is_low_grade_only("chondrosis looks new or worse")  # unqualified


def test_low_grade_markers_are_word_anchored():
    # "leve" (es: mild) inside "level", "early" inside "nearly" -- must not
    # count as low-grade qualifiers.
    assert not _is_low_grade_only("chondrosis at the level of the joint line")
    assert not _is_low_grade_only("nearly complete thickness cartilage loss")
    # "sever" inside "several" must not count as at-least-moderate
    assert _is_low_grade_only("several foci of mild chondrosis")


def test_window_edge_severity_and_bleed_fixes_together():
    # [redacted: quoted text from a restricted clinical dataset]
    # femorotibial lateral" -- only reachable via the window-edge fix
    # (previously caught only by accident through a cross-sentence bleed).
    assert make_keyword_proximity_lf("Lateral OA")(_REPORT_F) == POSITIVE
    # Medial: only "[redacted: report text]" -> low-grade only -> NEGATIVE (gold 0).
    assert make_keyword_proximity_lf("Medial OA")(_REPORT_F) == NEGATIVE
    # [redacted: quoted text from a restricted clinical dataset]
    assert make_keyword_proximity_lf("PF OA")(_REPORT_F) == POSITIVE


def test_compartment_term_hard_wrapped_across_newline_still_attributes():
    text = "condropatía grado 3 en la región de carga del cóndilo femoral\nmedial."
    assert _has_inline_term(text, 0, ["cóndilo femoral medial"])


def test_spanish_condral_plural_is_an_oa_term_but_osteocondral_is_not():
    lf = make_keyword_proximity_lf("Lateral OA")
    assert lf("úlceras condrales completas en el platillo tibial lateral.") == POSITIVE
    assert lf("lesión osteocondral en el platillo tibial lateral.") == ABSTAIN


def test_standalone_header_line_is_a_sentence_boundary():
    # [redacted: study ID]: a PF finding under "Impresión:" borrowed "lateral"[redacted: report text]"
    # line above the header (no period there to stop it).
    text = "la banda y el cóndilo femoral lateral\nimpresión:\ncondropatía leve de la rótula."
    assert "lateral" not in _local_window(text, text.index("condropatía"))


def test_joint_space_narrowing_in_any_language_is_not_low_grade():
    # [redacted: quoted text from a restricted clinical dataset]
    # which -- like English "[redacted: report text]" -- defines OA.
    assert not _is_low_grade_only("geringe veränderungen und medial gelenkspaltverschmälerung")


def test_croatian_meniscus_degeneration_is_not_an_oa_finding():
    lf = make_keyword_proximity_lf("Medial OA")
    text = "blage degenerativne promjene roga medijalnog meniska."
    assert lf(text) == ABSTAIN


def test_numeral_period_newline_is_a_boundary_unless_ordinal_wrapped():
    text = "condropatía con lesiones grado 2.\nmoderado derrame en rodilla."
    assert "moderado" not in _local_window(text, 0)
    wrapped = "hondromalacija iv.\nstupnja lateralnog dijela."
    assert "lateralnog dijela" in _local_window(wrapped, 0)


def test_icrs_subgrade_bulgarian_degree_and_gradig_compounds():
    assert _is_low_grade_only("(hondromalacija 1.b stupnja po icrs\nljestvici).")
    assert _is_low_grade_only("хондромалация – ii степен.")
    assert not _is_low_grade_only("гонартроза – ii-iii степен.")
    assert not _is_low_grade_only("хондромалация – 4 степен.")
    assert not _is_low_grade_only("auch höhergradige chondrale läsionen")
    assert _is_low_grade_only("zweitgradige chondropathie")
    assert _is_low_grade_only("lichtgradige kraakbeenlijden")


def test_greek_micro_sign_compartment_attribution():
    # "[redacted: report text]" written with MICRO SIGN mu, as in 305/321 Greek reports
    text = "Στο έσω διαµέρισµα παρατηρείται χονδρομαλακία βαθμού iv."
    assert make_keyword_proximity_lf("Medial OA")(text) == POSITIVE


def test_greek_grade_word():
    assert _is_low_grade_only("χονδρομαλακία βαθμού ii στον έσω κόνδυλο")
    assert not _is_low_grade_only("χονδρομαλακία βαθμού iv στον έσω κόνδυλο")


def test_arthrosis_forms_are_oa_terms_but_hemarthrosis_is_not():
    from data.label_mining.weak_supervision.oa_lfs import _find_oa_matches
    for t in ("gonartroz.", "osteoarthrosis.", "gonarthrosis.", "medial gonartroz"):
        assert list(_find_oa_matches(t)), t
    for t in ("hemarthrosis.", "lipohemarthrosis.", "hämarthros.", "hemartroz."):
        assert not list(_find_oa_matches(t)), t


def test_early_oa_diagnosis_is_not_downgraded_but_mild_chondral_lesion_is():
    lf = make_keyword_proximity_lf("Medial OA")
    assert lf("Early osteoarthritis in the medial compartment.") == POSITIVE
    assert lf("Mild chondromalacia on the medial femoral condyle.") == NEGATIVE
