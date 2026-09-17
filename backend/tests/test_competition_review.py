from copy import deepcopy

import pytest

from app.competition_review import CRITERIA, compile_review


def sample():
    images = [dict(image_id="a", image_path="images/a.png", sheet_path="sheets/s.png",
                   source_study_key="s", pixel_sha256="pixels")]
    studies = [dict(source_study_key="s", excel_row=3, regions={"spine": {
        "quality_class": 0, "criteria": dict(spine_coverage=0, spine_axis=0, spine_artifact=0)}})]
    row = {**{k: "" for k in ("quality_class", *CRITERIA, "notes")}, "image_id": "a",
           "image_path": "images/a.png", "sheet_path": "sheets/s.png", "excel_row": "3",
           "anatomical_region": "spine", "review_status": "mapped", "reviewer": "doctor-1"}
    return images, studies, [row]


def test_pending_never_becomes_a_label_and_mapping_preserves_original():
    images, studies, rows = sample()
    summary, labels, issues = compile_review(images, studies, rows)
    assert labels[0]["quality_class"] == 0
    assert labels[0]["label_origin"] == "organizer_labels_after_region_mapping"
    assert summary["clinical_labels_independently_reviewed"] == 0
    rows[0].update(review_status="pending", reviewer="")
    _, labels, issues = compile_review(images, studies, rows)
    assert labels == [] and issues[0]["code"] == "pending_review"


def test_unknown_expert_fields_not_backfilled_and_corrections_require_reason():
    images, studies, rows = sample()
    rows[0].update(review_status="reviewed", quality_class="1", spine_axis="1")
    with pytest.raises(ValueError, match="notes"):
        compile_review(images, studies, rows)
    rows[0]["notes"] = "Axis violation; other criteria not assessed"
    _, labels, _ = compile_review(images, studies, rows)
    assert labels[0]["criteria"]["spine_coverage"] is None
    assert labels[0]["original_labels"]["quality_class"] == 0


def test_ambiguous_region_and_contradictory_original_withheld():
    images, studies, rows = sample()
    images.append({**images[0], "image_id": "b", "image_path": "images/b.png", "pixel_sha256": "other"})
    rows.append({**rows[0], "image_id": "b", "image_path": "images/b.png"})
    summary, labels, issues = compile_review(images, studies, rows)
    assert labels == []
    assert summary["issue_counts"]["multiple_distinct_images_for_one_study_region"] == 2
    studies[0]["regions"]["spine"]["quality_class"] = 1
    _, labels, issues = compile_review(images[:1], studies, rows[:1])
    assert not labels and issues[0]["code"].startswith("original_overall")


def test_swapped_identity_missing_rows_and_irrelevant_labels_rejected():
    images, studies, rows = sample()
    bad = deepcopy(rows)
    bad[0]["excel_row"] = "99"
    with pytest.raises(ValueError, match="Immutable"):
        compile_review(images, studies, bad)
    with pytest.raises(ValueError, match="exactly"):
        compile_review(images, studies, [])
    rows[0].update(review_status="reviewed", hip_roi_coverage="0")
    with pytest.raises(ValueError, match="Non-applicable"):
        compile_review(images, studies, rows)
