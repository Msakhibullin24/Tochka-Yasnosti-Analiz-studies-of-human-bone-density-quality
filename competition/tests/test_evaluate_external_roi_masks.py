import gzip

import numpy as np
import pytest

from evaluate_external_roi_masks import bbox, overlap, patient_bootstrap, read_seg_nrrd


def test_nrrd_layer_and_xy_order(tmp_path):
    raster = np.zeros((2, 3, 4, 1), dtype=np.uint8)
    raster[1, 2, 1, 0] = 1
    header = ("NRRD0005\n"
              "type: uint8\n"
              "dimension: 4\n"
              "sizes: 2 3 4 1\n"
              "kinds: list domain domain domain\n"
              "encoding: gzip\n"
              "Segment0_ID:=Femoral_Total_bone_area\n"
              "Segment0_LabelValue:=1\n"
              "Segment0_Layer:=1\n\n")
    path = tmp_path / "mask.seg.nrrd"
    path.write_bytes(header.encode() + gzip.compress(raster.tobytes(order="F")))
    masks = read_seg_nrrd(path, (4, 3))
    assert masks["Femoral_Total_bone_area"].shape == (4, 3)
    assert masks["Femoral_Total_bone_area"][1, 2]
    assert bbox(masks["Femoral_Total_bone_area"], padding=0) == (2, 1, 3, 2)
    assert overlap(masks["Femoral_Total_bone_area"], masks["Femoral_Total_bone_area"])["dice"] == 1


def test_nrrd_rejects_wrong_image_shape(tmp_path):
    path = tmp_path / "mask.seg.nrrd"
    path.write_bytes(("NRRD0005\ntype: uint8\ndimension: 4\nsizes: 1 3 4 1\n"
                      "kinds: list domain domain domain\nencoding: gzip\n"
                      "Segment0_ID:=x\nSegment0_LabelValue:=1\nSegment0_Layer:=0\n\n").encode()
                     + gzip.compress(bytes(12)))
    with pytest.raises(ValueError, match="does not match"):
        read_seg_nrrd(path, (3, 4))


def test_patient_bootstrap_zero_delta_for_identical_masks():
    cases = [{"patient_sha256": "a", "lite_medsam": {"dice": .8},
              "filled_box_baseline": {"dice": .8}},
             {"patient_sha256": "b", "lite_medsam": {"dice": .6},
              "filled_box_baseline": {"dice": .6}}]
    result = patient_bootstrap(cases, repeats=100)
    assert result["delta_dice_vs_filled_box_95"] == [0, 0]
