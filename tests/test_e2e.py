"""Full pipeline over real DICOM networking on localhost."""
import json

import run_demo


def test_pipeline_end_to_end(tmp_path):
    rows = run_demo.main(["--count", "10", "--seed", "7", "--workdir", str(tmp_path), "--base-port", "12112"])
    truths = {r["study_uid"]: json.loads((tmp_path / "truth" / f"{r['study_uid']}.json").read_text()) for r in rows}
    heads = sum(t["kind"] == "head" for t in truths.values())
    states = [r["state"] for r in rows]
    assert "FAILED" not in states
    assert states.count("IGNORED") == heads      # head CTs never leave the router
    assert states.count("DONE") == len(rows) - heads

    pacs = tmp_path / "pacs"
    for r in rows:
        truth = truths[r["study_uid"]]
        folder = pacs / r["study_uid"]
        if truth["kind"] == "qa":
            # Phantoms go to ct-qa, never to the clinical model, and the verdict matches the injected fault
            result = json.loads((tmp_path / "results" / f"{r['study_uid']}.json").read_text())
            assert result["ai"]["model"] == "ct-qa" and r["rule"] == "ct-qa-phantom"
            assert result["ai"]["result"] == truth["qa_expected"]["result"]
            assert len(list(folder.glob("SR_*.dcm"))) == 1
            continue
        if truth["kind"] == "head":
            assert not folder.exists()           # nothing left the router
            continue
        # Results came back INTO the original study, as SR + key image
        assert len(list(folder.glob("SR_*.dcm"))) == 1
        assert len(list(folder.glob("OT_*.dcm"))) == 1
        # Solid nodules >= 6 mm are found, on the correct side
        n = truth["nodule"]
        result = json.loads((tmp_path / "results" / f"{r['study_uid']}.json").read_text())
        assert result["ai"]["model"] == "lung-nodule"
        found = [f for f in result["ai"]["findings"] if f["present"]]
        if n and n["density"] == "solid" and n["diameter_mm"] >= 6:
            assert found and found[0]["laterality"] == n["laterality"]
        if not n:
            assert not found                     # no false positives on clean studies


def test_retries_survive_flaky_ai(tmp_path):
    """Deterministic: every study's first 2 model calls fail with 503; the 3rd succeeds."""
    rows = run_demo.main(["--count", "4", "--seed", "11", "--workdir", str(tmp_path),
                          "--base-port", "13112", "--fail-first", "2"])
    assert all(r["state"] in ("DONE", "IGNORED") for r in rows)
    steps = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    inference = [s for s in steps if s["step"] == "ai_inference"]
    assert inference and all(s["ok"] and s["attempts"] == 3 for s in inference)
