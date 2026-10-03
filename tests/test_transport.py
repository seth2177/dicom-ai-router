"""Model transport: multipart/form-data (default) and DICOMweb STOW-RS, same contract and retry rules."""
import io
import json
import secrets
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydicom import dcmread

import run_demo
from airouter import ai_client, deid
from airouter.config import EndpointCfg, load_config
from airouter.mock_ai.app import app, stow_parts
from airouter.tools.modality_sim import make_study

CFG = Path(__file__).resolve().parents[1] / "config" / "router.yaml"
SALT = secrets.token_hex(32)
URL = "http://model/infer/lung-nodule"


@pytest.fixture
def wire(monkeypatch):
    """Route ai_client's HTTP calls into the mock model in-process, recording every request.
    state["fail"] = how many of the next calls get a 503 instead."""
    client, sent, state = TestClient(app), [], {"fail": 0}

    def post(url, timeout=None, **kw):
        sent.append(kw)
        if state["fail"] > 0:
            state["fail"] -= 1
            return httpx.Response(503, text="injected")
        return client.post(url.replace("http://model", ""), **kw)
    monkeypatch.setattr(ai_client.httpx, "post", post)
    monkeypatch.setattr(ai_client.time, "sleep", lambda s: None)
    return sent, state


def _study(seed=21):
    slices, _ = make_study(seed, "chest")
    anon = [deid.deidentify(d, SALT)[0] for d in slices]
    return slices, anon


def test_stow_rs_request_is_ps3_18_shaped_and_carries_only_deidentified_data(wire):
    sent, _ = wire
    slices, anon = _study()
    out = ai_client.infer(EndpointCfg(URL, transport="stow-rs"), anon)
    assert out["model"] == "lung-nodule" and out["n_images"] == len(slices) and out["_attempts"] == 1
    req = sent[0]
    ctype = req["headers"]["Content-Type"]
    assert ctype.startswith("multipart/related") and 'type="application/dicom"' in ctype and "files" not in req
    parts = stow_parts(ctype, req["content"])
    assert len(parts) == len(slices)
    body = req["content"]
    for orig in slices:                                         # nothing identifying, in any form
        for value in (str(orig.PatientName), str(orig.PatientID), str(orig.StudyInstanceUID),
                      str(orig.SeriesInstanceUID), str(orig.SOPInstanceUID), str(orig.get("InstitutionName", ""))):
            assert not value or value.encode() not in body, value
    for part, orig in zip(parts, slices, strict=True):
        ds = dcmread(io.BytesIO(part))
        assert deid.find_phi(ds, orig) == []
        assert str(ds.PatientName).startswith("ANON^") and ds.PatientIdentityRemoved == "YES"
    assert out["study_instance_uid"] == anon[0].StudyInstanceUID != slices[0].StudyInstanceUID


def test_both_transports_get_the_same_answer(wire):
    _, anon = _study(22)
    a = ai_client.infer(EndpointCfg(URL), anon)
    b = ai_client.infer(EndpointCfg(URL, transport="stow-rs"), anon)
    for k in ("result", "summary_lines", "key_sop_instance_uid", "overlays"):
        assert a[k] == b[k], k


def test_stow_rs_retries_5xx_and_never_4xx(wire, monkeypatch):
    sent, state = wire
    _, anon = _study(23)
    ep = EndpointCfg(URL, retries=3, transport="stow-rs")
    state["fail"] = 2
    assert ai_client.infer(ep, anon)["_attempts"] == 3
    state["fail"] = 3
    with pytest.raises(ai_client.AIError, match="after 3 attempts"):
        ai_client.infer(ep, anon)
    sent.clear()
    # A request the model cannot accept (wrong media type) is a 415: reported at once, never retried
    monkeypatch.setattr(ai_client, "stow_body", lambda blobs: (b"--x--\r\n", 'multipart/related; type="application/json"; boundary=x'))
    with pytest.raises(ai_client.AIError, match="HTTP 415"):
        ai_client.infer(ep, anon)
    assert len(sent) == 1


def test_mock_rejects_non_dicom_parts():
    body = b'--b\r\nContent-Type: application/json\r\n\r\n{}\r\n--b--\r\n'
    r = TestClient(app).post("/infer/ct-qa", content=body,
                             headers={"Content-Type": 'multipart/related; type="application/dicom"; boundary=b'})
    assert r.status_code == 415


def test_unknown_transport_is_refused(tmp_path):
    with pytest.raises(ValueError, match="transport"):
        load_config(CFG, {"router": {"workdir": str(tmp_path)}, "deid": {"salt": SALT},
                          "ai_endpoints": {"ct-qa": {"url": URL, "transport": "dimse"}}})


def test_pipeline_end_to_end_over_stow_rs(tmp_path):
    rows = run_demo.main(["--count", "4", "--seed", "7", "--workdir", str(tmp_path), "--base-port", "14112",
                          "--transport", "stow-rs"])
    assert "FAILED" not in [r["state"] for r in rows] and "DONE" in [r["state"] for r in rows]
    for f in (tmp_path / "results").glob("*.json"):
        assert json.loads(f.read_text())["ai"]["n_images"] > 0
