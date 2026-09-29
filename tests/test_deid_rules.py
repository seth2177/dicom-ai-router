from pathlib import Path

import pytest

from airouter import deid
from airouter.config import load_config
from airouter.rules import match_rule
from airouter.tools.modality_sim import make_study

CFG = Path(__file__).resolve().parents[1] / "config" / "router.yaml"
SALT_A, SALT_B = "a" * 32, "b" * 32


@pytest.fixture
def chest():
    slices, truth = make_study(1, "chest", None)
    return slices[0]


def test_no_phi_survives(chest):
    anon, pairs = deid.deidentify(chest, SALT_A)
    assert deid.find_phi(anon, chest) == []
    assert "InstitutionName" not in anon and "StationName" not in anon
    assert anon.PatientIdentityRemoved == "YES"
    assert str(anon.PatientName).startswith("ANON^")


def test_uids_deterministic_and_reversible(chest):
    a1, p1 = deid.deidentify(chest, SALT_A)
    a2, _ = deid.deidentify(chest, SALT_A)
    a3, _ = deid.deidentify(chest, SALT_B)
    assert a1.StudyInstanceUID == a2.StudyInstanceUID != a3.StudyInstanceUID
    crosswalk = {anon: orig for anon, orig, _ in p1}
    assert crosswalk[a1.SOPInstanceUID] == chest.SOPInstanceUID
    assert crosswalk[a1.StudyInstanceUID] == chest.StudyInstanceUID


def test_burned_in_annotation_rejected(chest):
    chest.BurnedInAnnotation = "YES"
    with pytest.raises(deid.BurnedInAnnotationError):
        deid.deidentify(chest, SALT_A)


def test_rules(tmp_path):
    cfg = load_config(CFG, {"router": {"workdir": str(tmp_path)}, "deid": {"salt": SALT_A}})
    chest, _ = make_study(2, "chest", None)
    head, _ = make_study(3, "head", None)
    assert match_rule(chest[0], cfg.rules).name == "ct-chest-by-bodypart"
    assert match_rule(head[0], cfg.rules) is None
    # Blank BodyPartExamined is common in the field -> falls back to the description rule
    chest[0].BodyPartExamined = ""
    assert match_rule(chest[0], cfg.rules).name == "ct-chest-by-description"
