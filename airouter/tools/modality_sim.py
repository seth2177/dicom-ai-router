"""Synthetic CT scanner: builds fake chest (and head) CT studies and sends them by C-STORE.

Every patient, MRN and image here is synthetic. The pixel data is geometry:
a body ellipse, two air-filled lungs and, optionally, a spherical nodule of
known size and side. Because we placed the nodule, we know the ground truth.
That truth is saved to data/truth/<StudyInstanceUID>.json -- it is what the
llm-eval-radiology stage scores the AI and the LLM report against.

  python -m airouter.tools.modality_sim --count 6 --seed 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np
from pydicom import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROWS = COLS = 128
SPACING_MM = 2.5          # 320 mm field of view
SLICE_MM = 5.0
N_SLICES = 24

FIRST = ["JANE", "JOHN", "MARIA", "LUIS", "ANA", "ROBERT", "LINDA", "CARLOS", "SUSAN", "DAVID"]
LAST = ["DOE", "ROE", "SYNTHETIC", "EXAMPLE", "SAMPLE", "TESTCASE"]


def make_volume(rng: np.random.Generator, nodule: dict | None) -> np.ndarray:
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    cy, cx = ROWS / 2, COLS / 2
    vol = np.full((N_SLICES, ROWS, COLS), -1000.0, dtype=np.float32)
    body = ((xx - cx) / 58) ** 2 + ((yy - cy) / 44) ** 2 <= 1
    r_lung = ((xx - (cx - 26)) / 20) ** 2 + ((yy - cy) / 32) ** 2 <= 1   # image-left = patient RIGHT
    l_lung = ((xx - (cx + 26)) / 20) ** 2 + ((yy - cy) / 32) ** 2 <= 1
    for z in range(N_SLICES):
        vol[z][body] = 40
        if 2 <= z <= N_SLICES - 3:
            vol[z][r_lung | l_lung] = -850
    if nodule:
        side_cx = cx - 26 if nodule["laterality"] == "right" else cx + 26
        ncx, ncy, ncz = side_cx + nodule["dx"], cy + nodule["dy"], nodule["slice"]
        r_px, r_z = nodule["diameter_mm"] / 2 / SPACING_MM, nodule["diameter_mm"] / 2 / SLICE_MM
        for z in range(N_SLICES):
            dz = (z - ncz) / max(r_z, 0.5)
            if abs(dz) > 1:
                continue
            rr = r_px * np.sqrt(max(0.0, 1 - dz ** 2))
            vol[z][(xx - ncx) ** 2 + (yy - ncy) ** 2 <= rr ** 2] = nodule["hu"]
    vol += rng.normal(0, 15, vol.shape).astype(np.float32)
    return vol


PHANTOM_DIAMETER_MM = 200.0


def make_phantom_volume(rng: np.random.Generator, qa: dict) -> np.ndarray:
    """Water phantom in air. qa = {bias_hu, cup_hu, noise_hu}: known faults we inject.

    bias  -> whole-phantom CT-number drift (what a failed air/water calibration looks like)
    cup   -> edge brighter than centre by `cup` HU at the rim (beam-hardening cupping)
    """
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    R = PHANTOM_DIAMETER_MM / 2 / SPACING_MM
    rr = np.sqrt((xx - COLS / 2) ** 2 + (yy - ROWS / 2) ** 2)
    vol = np.full((N_SLICES, ROWS, COLS), -1000.0, dtype=np.float32)
    inside = rr <= R
    water = qa["bias_hu"] + qa["cup_hu"] * (rr / R) ** 2
    for z in range(N_SLICES):
        vol[z][inside] = water[inside]
    vol += rng.normal(0, qa["noise_hu"], vol.shape).astype(np.float32)
    return vol


def expected_qa(qa: dict) -> dict:
    """Analytic ground truth for what an ideal ROI measurement should read."""
    from ..mock_ai.qa import grade
    R = PHANTOM_DIAMETER_MM / 2 / SPACING_MM
    roi_r = 0.1 * R
    edge_d = R - 10.0 / SPACING_MM - roi_r
    # mean of cup*(r/R)^2 over an ROI of radius a centred at distance d is cup*(d^2 + a^2/2)/R^2,
    # so edge ROI minus centre ROI = cup * edge_d^2 / R^2
    uniformity = qa["cup_hu"] * edge_d ** 2 / R ** 2
    water = qa["bias_hu"] + qa["cup_hu"] * (roi_r ** 2 / 2) / R ** 2
    status, _ = grade(water, abs(uniformity))
    return {"ct_hu": round(water, 2), "uniformity_hu": round(abs(uniformity), 2),
            "noise_hu": qa["noise_hu"], "result": status}


def make_study(seed: int, kind: str = "chest", nodule: dict | None = None) -> tuple[list[Dataset], dict]:
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    now = dt.datetime.now()
    patient = {"name": f"{r.choice(LAST)}^{r.choice(FIRST)}", "id": f"SYN{r.randint(100000, 999999)}",
               "birth": f"{r.randint(1940, 1995)}{r.randint(1, 12):02d}{r.randint(1, 28):02d}",
               "sex": r.choice("MF"), "accession": f"ACC{r.randint(1000000, 9999999)}"}
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    if kind == "qa":
        patient = {"name": "QA^DAILY", "id": "QA", "birth": "", "sex": "O", "accession": ""}
        desc, body_part, series_desc = "DAILY QA", "", "WATER 120KV"
        vol = make_phantom_volume(rng, nodule)
    else:
        desc, body_part = ("CT CHEST W/O CONTRAST", "CHEST") if kind == "chest" else ("CT HEAD W/O CONTRAST", "HEAD")
        series_desc = "AXIAL 5MM"
        vol = make_volume(rng, nodule if kind == "chest" else None)
    stored = np.clip(vol + 1024, 0, 4095).astype(np.uint16)

    slices = []
    for z in range(N_SLICES):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.PatientName, ds.PatientID = patient["name"], patient["id"]
        ds.PatientBirthDate, ds.PatientSex = patient["birth"], patient["sex"]
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
        ds.AccessionNumber = patient["accession"]
        ds.ReferringPhysicianName = "SYNTHETIC^REFERRER"
        ds.InstitutionName = "SYNTHETIC IMAGING CENTER"
        ds.StationName = "CT_SIM01"
        ds.StudyID = "1"
        ds.StudyDescription = desc
        ds.SeriesDescription = series_desc
        ds.KVP = 120
        ds.BodyPartExamined = body_part
        ds.Modality = "CT"
        ds.Manufacturer = "SyntheticCT"
        ds.SeriesNumber, ds.InstanceNumber = 2, z + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImagePositionPatient = [-160.0, -160.0, -z * SLICE_MM]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceThickness = SLICE_MM
        ds.PixelSpacing = [SPACING_MM, SPACING_MM]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = ROWS, COLS
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = -1024, 1, "HU"
        ds.WindowCenter, ds.WindowWidth = -600, 1500
        ds.BurnedInAnnotation = "NO"
        ds.PixelData = stored[z].tobytes()
        slices.append(ds)

    if kind == "qa":
        return slices, {"study_uid": study_uid, "kind": "qa", "patient_name": patient["name"],
                        "nodule_present": False, "nodule": None, "qa_injected": nodule, "qa_expected": expected_qa(nodule)}
    truth = {"study_uid": study_uid, "kind": kind, "patient_name": patient["name"],
             "nodule_present": bool(nodule and kind == "chest"),
             "nodule": ({**{k: nodule[k] for k in ("laterality", "diameter_mm", "slice")},
                         "density": "ground-glass" if nodule["hu"] < -200 else "solid"}
                        if nodule and kind == "chest" else None)}
    return slices, truth


def random_case(r: random.Random) -> tuple[str, dict | None]:
    """A realistic mix: negatives, clear nodules, small/faint nodules the model may miss,
    head CTs (no rule), and daily-QA water phantoms (some drifted or cupped, i.e. failing)."""
    roll = r.random()
    if roll < 0.12:
        return "head", None
    if roll < 0.27:
        return "qa", {"bias_hu": r.choice([0.0, 1.5, -2.0, 3.0, 9.0]), "cup_hu": r.choice([0.0, 2.0, 4.0, 10.0]),
                      "noise_hu": r.choice([4.0, 5.0, 6.0])}
    if roll < 0.45:
        return "chest", None
    diameter = r.choice([3.0, 5.0, 6.0, 8.0, 10.0, 14.0, 18.0])
    return "chest", {"laterality": r.choice(["right", "left"]), "diameter_mm": diameter,
                     "slice": r.randint(6, N_SLICES - 7), "dx": r.randint(-6, 6), "dy": r.randint(-14, 14),
                     "hu": r.choice([40, 40, 40, -350])}  # -350 HU = faint ground-glass nodule


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11112)
    ap.add_argument("--called-ae", default="AIROUTER")
    ap.add_argument("--truth-dir", default="data/truth")
    a = ap.parse_args(argv)
    send_studies(a.count, a.seed, a.host, a.port, a.called_ae, Path(a.truth_dir))


def send_studies(count, seed, host, port, called_ae, truth_dir: Path) -> list[dict]:
    from airouter.sender import c_store

    truth_dir.mkdir(parents=True, exist_ok=True)
    r = random.Random(seed)
    truths = []
    for i in range(count):
        kind, nodule = random_case(r)
        slices, truth = make_study(seed * 1000 + i, kind, nodule)
        (truth_dir / f"{truth['study_uid']}.json").write_text(json.dumps(truth, indent=2))
        c_store(slices, host, port, called_ae, calling_ae="CT_SIM01")
        if kind == "qa":
            e = truth["qa_expected"]
            desc = f"QA phantom, expected {e['result']} (CT {e['ct_hu']:+.1f} HU, uniformity {e['uniformity_hu']:.1f} HU)"
        elif kind == "head":
            desc = "head CT (no rule should match)"
        else:
            desc = "no nodule" if not truth["nodule"] else f"{truth['nodule']['diameter_mm']}mm {truth['nodule']['laterality']}"
        print(f"  sent {kind:5s} study {i + 1}/{count}: {truth['patient_name']:18s} {len(slices)} images  truth: {desc}")
        truths.append(truth)
    return truths


def send_folder(folder: Path, host: str, port: int, called_ae: str = "AIROUTER") -> int:
    """C-STORE every DICOM file under `folder`, one association per study (e.g. scrubbed real phantoms)."""
    from collections import defaultdict

    from pydicom import dcmread

    from airouter.sender import c_store
    studies = defaultdict(list)
    for p in sorted(Path(folder).rglob("*.dcm")):
        ds = dcmread(p)
        studies[str(ds.StudyInstanceUID)].append(ds)
    for ss in studies.values():
        c_store(ss, host, port, called_ae, calling_ae="REAL_QA")
    return len(studies)


if __name__ == "__main__":
    main()
