"""Regression tests for security and correctness bugs found in testing. Each test fails on the old code."""
import secrets
import socket
from pathlib import Path

import numpy as np
import pytest
from pydicom import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
from pynetdicom import AE

from airouter import deid, phi_audit, results, site_scrub
from airouter.config import load_config
from airouter.pipeline import select_instances
from airouter.router import Router
from airouter.rules import match_rule
from airouter.tools.modality_sim import make_study
from tests import synthetic_qa

CFG = Path(__file__).resolve().parents[1] / "config" / "router.yaml"
SALT = secrets.token_hex(32)


# ---- salt ----------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "change-me-per-site", "salt", "short"])
def test_weak_or_placeholder_salt_is_refused(bad, tmp_path, monkeypatch):
    monkeypatch.delenv("AIROUTER_DEID_SALT", raising=False)
    with pytest.raises(deid.WeakSaltError):
        deid.deidentify(make_study(1)[0][0], bad)
    with pytest.raises(deid.WeakSaltError):
        load_config(CFG, {"router": {"workdir": str(tmp_path)}, "deid": {"salt": bad}})


def test_salt_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AIROUTER_DEID_SALT", SALT)
    assert load_config(CFG, {"router": {"workdir": str(tmp_path)}}).deid_salt == SALT


# ---- receiver ------------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def router(tmp_path):
    port = _free_port()
    cfg = load_config(CFG, {"router": {"workdir": str(tmp_path / "w"), "port": port, "bind_address": "127.0.0.1",
                                       "study_quiet_seconds": 60, "allowed_calling_aes": ["CT_SIM01"]},
                            "deid": {"salt": SALT}})
    r = Router(cfg).start()
    yield r, tmp_path, port
    r.stop()


def _send(ds, port, called="AIROUTER", calling="CT_SIM01"):
    ae = AE(ae_title=calling)
    ae.add_requested_context(CTImageStorage, ExplicitVRLittleEndian)
    assoc = ae.associate("127.0.0.1", port, ae_title=called)
    if not assoc.is_established:
        return None
    status = assoc.send_c_store(ds)
    assoc.release()
    return status.Status


@pytest.mark.filterwarnings("ignore:Invalid value for VR UI")
def test_path_traversal_uid_is_rejected(router):
    r, tmp, port = router
    ds = make_study(2)[0][0]
    ds.SOPInstanceUID = "../../../escaped"
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    assert _send(ds, port) == 0xC000
    assert not list(tmp.rglob("escaped*"))
    assert not list(tmp.parent.rglob("escaped*"))


def test_wrong_called_ae_is_rejected(router):
    assert _send(make_study(3)[0][0], router[2], called="NOT_THE_ROUTER") is None


def test_calling_ae_allow_list(router):
    assert _send(make_study(4)[0][0], router[2], calling="STRANGER") is None
    assert _send(make_study(4)[0][0], router[2], calling="CT_SIM01") == 0x0000


# ---- de-identification ---------------------------------------------------------------------
def test_router_deid_passes_the_independent_audit(tmp_path):
    """The patient profile alone (no site tokens) must already clear dates, UIDs, comments,
    overlays, private tags, file-meta AE and preamble -- as the docs claim."""
    orig = synthetic_qa._ds("head", "Daily QA", "Daily QA 7/29/2026 1:48 AM", "QA Phantom Scan: Head", True)
    clean, _ = deid.deidentify(orig, SALT)
    clean.save_as(tmp_path / "0001.dcm", enforce_file_format=True)
    uids = {str(e.value) for e in orig.iterall() if e.VR == "UI"} | {str(orig.file_meta.MediaStorageSOPInstanceUID)}
    uids = {u for u in uids if not u.startswith("1.2.840.10008.")}
    r = phi_audit.audit(tmp_path, set(), uids, originals=[orig])
    assert r["passed"], r["counts"]
    assert "IrradiationEventUID" in clean and str(clean.IrradiationEventUID).startswith("2.25.")


def test_multivalued_names_are_harvested_and_caught(tmp_path):
    orig = synthetic_qa._ds("head", "Daily QA", "QA", "QA Phantom Scan: Head", True)
    orig.OperatorsName = ["WHITAKER^JUNE", "ABERNATHY^TOM"]
    orig.ProtocolName = "WHITAKER head protocol"          # free text the patient profile keeps
    tokens = site_scrub.harvest_tokens([orig])
    assert {"WHITAKER", "ABERNATHY", "JUNE"} <= tokens
    assert not any("[" in t for t in tokens)
    # Proper scrub: gone.
    clean, _ = site_scrub.scrub(orig, SALT, tokens)
    assert "WHITAKER" not in str(clean.get("ProtocolName", "")).upper()
    # Simulate a harvester bug (no tokens): the audit's OWN harvest must still catch it.
    leaky, _ = site_scrub.scrub(orig, SALT, set())
    leaky.save_as(tmp_path / "0001.dcm", enforce_file_format=True)
    r = phi_audit.audit(tmp_path, set(), set(), originals=[orig])
    assert not r["passed"] and r["counts"]["token"] > 0


# ---- routing -------------------------------------------------------------------------------
def test_qa_rule_does_not_capture_patients(tmp_path):
    cfg = load_config(CFG, {"router": {"workdir": str(tmp_path)}, "deid": {"salt": SALT}})
    ds = make_study(5, "chest")[0][0]
    ds.PatientName = "DAILY^JOHN"
    ds.StudyDescription = "CT CHEST WITH WATER ORAL CONTRAST"
    assert match_rule(ds, cfg.rules).model == "lung-nodule"
    ds.PatientName = "WATERS^ANN"
    assert match_rule(ds, cfg.rules).model == "lung-nodule"
    ds.PatientID = "QC-00123"                                   # an MRN that starts with QC
    ds.StudyDescription = "CT CHEST QA FOLLOWUP"
    assert match_rule(ds, cfg.rules).model == "lung-nodule"


def test_localizer_and_dose_screen_do_not_break_a_study():
    slices, _ = make_study(6, "chest")
    loc = make_study(6, "chest")[0][0]
    loc.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    loc.SeriesInstanceUID = "1.2.3.4.5"
    dose = make_study(6, "chest")[0][1]
    dose.BurnedInAnnotation = "YES"
    dose.SeriesInstanceUID = "1.2.3.4.6"
    thin = make_study(6, "chest")[0][:3]
    for d in thin:
        d.SeriesInstanceUID = "1.2.3.4.7"
    used, dropped = select_instances(slices + [loc, dose] + thin, "largest")
    assert len(used) == len(slices)
    assert dropped == {"localizer": 1, "burned-in annotation": 1, "not the main series": 3}


def test_model_server_handles_mixed_matrix_sizes():
    """A second recon at a different matrix size must not crash the volume stack."""
    from fastapi.testclient import TestClient

    from airouter.mock_ai.app import app
    a, _ = make_study(8, "chest")
    b = make_study(8, "chest")[0][:3]
    for d in b:
        d.SeriesInstanceUID, d.Rows, d.Columns = "1.2.3.4.8", 64, 64
        d.PixelData = np.zeros((64, 64), np.uint16).tobytes()
    files = [("files", (f"{i}.dcm", _bytes(d), "application/dicom")) for i, d in enumerate(a + b)]
    r = TestClient(app).post("/infer/lung-nodule", files=files)
    assert r.status_code == 200 and r.json()["n_images"] == len(a)


def test_dose_sr_first_does_not_hijack_routing(tmp_path):
    """Real CT studies often start with a dose SR at InstanceNumber 1."""
    from airouter.audit import Audit
    from airouter.pipeline import inbox_folder, process_study
    from airouter.store import Store
    cfg = load_config(CFG, {"router": {"workdir": str(tmp_path)}, "deid": {"salt": SALT},
                            "ai_endpoints": {"lung-nodule": {"url": "http://127.0.0.1:9/none", "retries": 1}}})
    slices, _ = make_study(9, "chest")
    sr = Dataset()
    sr.SOPClassUID, sr.SOPInstanceUID = "1.2.840.10008.5.1.4.1.1.88.67", "1.2.3.4.5.6.7"   # X-Ray Radiation Dose SR
    sr.StudyInstanceUID, sr.SeriesInstanceUID, sr.Modality, sr.InstanceNumber = slices[0].StudyInstanceUID, "1.2.3.9", "SR", 0
    folder = inbox_folder(cfg, slices[0].StudyInstanceUID)
    folder.mkdir(parents=True)
    for d in slices:
        d.save_as(folder / f"{d.SOPInstanceUID}.dcm", enforce_file_format=True)
    sr.save_as(folder / "dose.dcm", enforce_file_format=True, implicit_vr=True, little_endian=True)
    store = Store(cfg.db_path)
    store.instance_received(slices[0].StudyInstanceUID, "CT")
    process_study(slices[0].StudyInstanceUID, cfg, store, Audit(cfg.audit_path))
    row = store.studies()[0]
    assert row["state"] == "FAILED" and "AI endpoint" in row["error"]      # it was ROUTED (then the fake model is down)
    audit = (tmp_path / "audit.jsonl").read_text()
    assert '"rule": "ct-chest-by-bodypart"' in audit and '"not a CT image": 1' in audit


def test_result_uids_are_deterministic():
    """A re-run must re-send the same objects, not add duplicates to PACS."""
    slices, _ = make_study(10, "chest")
    ai = {"model": "lung-nodule", "model_version": "t", "summary_lines": []}
    one = (results.build_sr(slices[0], slices, ai, None), results.build_key_image(slices[5], ai, False))
    two = (results.build_sr(slices[0], slices, ai, None), results.build_key_image(slices[5], ai, False))
    assert [o.SOPInstanceUID for o in one] == [o.SOPInstanceUID for o in two]
    assert [o.SeriesInstanceUID for o in one] == [o.SeriesInstanceUID for o in two]
    assert one[0].SOPInstanceUID != one[1].SOPInstanceUID
    # Different findings (e.g. after late images): new instance, same series (DICOM: changed content, new UID)
    revised = dict(ai, result="POSITIVE", summary_lines=[["Finding 1", "pulmonary nodule, left lung"]])
    three = results.build_sr(slices[0], slices, revised, None)
    assert three.SOPInstanceUID != one[0].SOPInstanceUID and three.SeriesInstanceUID == one[0].SeriesInstanceUID


def test_audit_cli_fails_closed_on_bad_originals(tmp_path):
    from airouter.phi_audit import main as audit_main
    (tmp_path / "tokens.json").write_text("[]")
    with pytest.raises(SystemExit) as e:
        audit_main([str(tmp_path), "--tokens", str(tmp_path / "tokens.json"), "--originals", str(tmp_path / "nope")])
    assert "no readable DICOM" in str(e.value)


def test_names_at_any_depth_never_reach_the_model():
    ds = make_study(11)[0][0]
    ds.PhysiciansOfRecord = "HOUSE^GREGORY"
    ds.PatientMotherBirthName = "JONES"
    ds.ReasonForTheRequestedProcedure = "smoker, MRN 4471823"
    sps = Dataset()
    sps.ScheduledPerformingPhysicianName, sps.ScheduledStationAETitle = "CUDDY^LISA", "CTROOM2"
    ds.ScheduledProcedureStepSequence = Sequence([sps])
    grp = Dataset()
    grp.PatientName, grp.PatientID = "SMITH^JOHN", "MRN12345"
    ds.SourcePatientGroupIdentificationSequence = Sequence([grp])
    ref = Dataset()
    ref.NameOfPhysiciansReadingStudy = "WILSON^JAMES"
    ds.ReferencedPerformedProcedureStepSequence = Sequence([ref])
    clean, _ = deid.deidentify(ds, SALT)
    text = str(clean).upper()
    for leaked in ("HOUSE", "JONES", "4471823", "CUDDY", "CTROOM2", "SMITH", "MRN12345", "WILSON"):
        assert leaked not in text, leaked
    assert clean.SOPClassUID == ds.SOPClassUID                  # class UIDs are not "pseudonymised"


def test_rerun_command(tmp_path, monkeypatch):
    from airouter.__main__ import main
    from airouter.store import Store
    monkeypatch.setenv("AIROUTER_DEID_SALT", SALT)
    cfgfile = tmp_path / "router.yaml"
    cfgfile.write_text(CFG.read_text().replace("workdir: ./data", f"workdir: {(tmp_path / 'w').as_posix()}"))
    with pytest.raises(SystemExit):
        main(["rerun", "--config", str(cfgfile)])               # nothing to rerun -> clear message
    store = Store(tmp_path / "w" / "router.sqlite")
    store.instance_received("1.2.3", "CT")
    store.set_state("1.2.3", "FAILED", error="x")
    main(["rerun", "--failed", "--config", str(cfgfile)])       # no inbox copy -> skipped, no crash


def _bytes(ds) -> bytes:
    import io

    from pydicom import dcmwrite
    buf = io.BytesIO()
    dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


# ---- result objects ------------------------------------------------------------------------
def test_sr_groups_evidence_by_series_and_has_type2():
    a, _ = make_study(7, "chest")
    b = make_study(7, "chest")[0][:4]
    for d in b:
        d.SeriesInstanceUID = "1.2.3.4.99"
        d.StudyInstanceUID = a[0].StudyInstanceUID
    for d in a:
        del d.AccessionNumber                                   # absent in source -> still written empty
    ai = {"model": "ct-qa", "model_version": "t", "summary_lines": [["Series 1 (120 kV, 10 mm)", "x"],
                                                                     ["Series 1 (120 kV, 10 mm) note", "y"]]}
    sr = results.build_sr(a[0], a + b, ai, None)
    series = sr.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence
    assert len(series) == 2
    assert sorted(len(s.ReferencedSOPSequence) for s in series) == [4, len(a)]
    for kw in ("PerformedProcedureCodeSequence", "ReferencedPerformedProcedureStepSequence", "AccessionNumber",
               "PatientBirthDate", "ReferringPhysicianName", "StudyID"):
        assert kw in sr, kw
    codes = [i.ConceptNameCodeSequence[0].CodeValue for i in sr.ContentSequence]
    assert len(codes) == len(set(codes))                        # no two labels share a code
    assert sr.CodingSchemeIdentificationSequence[0].CodingSchemeDesignator == "99AIROUTER"


def test_audit_negative_control_covers_every_check(tmp_path):
    ds = synthetic_qa._ds("head", "Daily QA", "Daily QA 7/29/2026 1:48 AM", "QA Phantom Scan: Head", True)
    ds.BurnedInAnnotation = "YES"
    (tmp_path / "LAKESIDE").mkdir()
    ds.save_as(tmp_path / "LAKESIDE" / "0001.dcm", enforce_file_format=True)
    (tmp_path / "manifest.json").write_text('{"site": "LAKESIDE VALLEY"}')
    r = phi_audit.audit(tmp_path, {"LAKESIDE"}, {str(ds.SOPInstanceUID)}, originals=[ds])
    missing = [c for c, n in r["counts"].items() if n == 0]
    assert missing == [], missing
