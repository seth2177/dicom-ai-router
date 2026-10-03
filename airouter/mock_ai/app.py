"""Model server (FastAPI). Stand-in for vendor model endpoints.

POST /infer/lung-nodule   DICOM Part-10 instances of one study, as either
                            multipart/form-data, field 'files', or
                            STOW-RS: multipart/related; type="application/dicom" (PS3.18)
POST /infer/ct-qa         same, for water-phantom QA studies
GET  /health

Every endpoint answers with the same contract, so the router never needs
model-specific code:

  result              short verdict (POSITIVE / NEGATIVE / PASS / FAIL / MEASURED / NOT_A_PHANTOM)
  summary_lines       [[label, value], ...]  -> SR text items and key-image caption
  key_sop_instance_uid  the image the result is about (anonymised UID)
  overlays            [{cx, cy, r, label}]  circles to draw on the key image
  display_window      [level, width] for the key image
  findings / metrics  model-specific detail, kept for evaluation

Fault injection, to see the router's retries work:
  MOCK_AI_FAIL_RATE=0.3   fraction of requests answered with HTTP 503 (random; for demos)
  MOCK_AI_FAIL_FIRST=2    every study's first N attempts get HTTP 503 (deterministic; for tests)
  MOCK_AI_DELAY_S=2       added latency per request
"""
from __future__ import annotations

import io
import os
import random
import threading
import time
from collections import defaultdict

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from pydicom import dcmread

from .detector import detect
from .qa import analyse_series

VERSIONS = {"lung-nodule": "0.2.0-demo", "ct-qa": "0.1.0"}
app = FastAPI(title="dicom-ai-router model server")


@app.get("/health")
def health():
    return {"ok": True, "models": VERSIONS}


_attempts: dict[str, int] = defaultdict(int)
_attempts_lock = threading.Lock()


async def dicom_parts(request: Request) -> list[bytes]:
    """The request's DICOM instances, whichever transport the router used."""
    ctype = request.headers.get("content-type", "")
    if ctype.lower().startswith("multipart/related"):
        return stow_parts(ctype, await request.body())
    if ctype.lower().startswith("multipart/form-data"):
        form = await request.form()
        return [await f.read() for f in form.getlist("files") if hasattr(f, "read")]
    raise HTTPException(415, "expected multipart/form-data or multipart/related; type=\"application/dicom\"")


def stow_parts(ctype: str, body: bytes) -> list[bytes]:
    """Split a STOW-RS (PS3.18) multipart/related body into its application/dicom parts."""
    params = {k.strip().lower(): v.strip().strip('"') for k, _, v in (p.partition("=") for p in ctype.split(";")[1:])}
    if params.get("type", "").lower() != "application/dicom" or not params.get("boundary"):
        raise HTTPException(415, 'STOW-RS needs type="application/dicom" and a boundary')
    delim = b"--" + params["boundary"].encode()
    parts = []
    for chunk in body.split(delim)[1:]:
        if chunk.startswith(b"--"):                  # closing delimiter
            break
        head, sep, content = chunk.partition(b"\r\n\r\n")
        if not sep or b"content-type: application/dicom" not in head.lower():
            raise HTTPException(415, "every STOW-RS part must be application/dicom")
        parts.append(content[:-2] if content.endswith(b"\r\n") else content)
    return parts


def _load(parts: list[bytes]) -> list:
    # Endpoints are plain `def`: FastAPI runs them in a worker thread pool, so a slow
    # (CPU-bound) inference does not block other requests the way it would in `async def`.
    if not parts:
        raise HTTPException(422, "no DICOM instances in the request")
    slices = [dcmread(io.BytesIO(b)) for b in parts]
    study = str(slices[0].get("StudyInstanceUID", "")) if slices else ""
    with _attempts_lock:
        if len(_attempts) > 10_000:                # bounded: this is test instrumentation
            _attempts.clear()
        _attempts[study] += 1
        attempt = _attempts[study]
    if attempt <= int(os.environ.get("MOCK_AI_FAIL_FIRST", 0)):
        raise HTTPException(503, f"injected failure (attempt {attempt})")
    if random.random() < float(os.environ.get("MOCK_AI_FAIL_RATE", 0)):
        raise HTTPException(503, "injected failure")
    time.sleep(float(os.environ.get("MOCK_AI_DELAY_S", 0)))
    slices = [s for s in slices if s.get("Modality") == "CT" and "PixelData" in s
              and "LOCALIZER" not in [str(v).upper() for v in s.get("ImageType", [])]]
    if not slices:
        raise HTTPException(422, "no axial CT images with pixel data")
    return slices


def _largest_consistent(slices: list) -> list:
    """Defensive: a volume needs one series of one matrix size."""
    groups: dict = defaultdict(list)
    for s in slices:
        groups[(str(s.SeriesInstanceUID), s.Rows, s.Columns)].append(s)
    return max(groups.values(), key=len)


def _volume(slices) -> np.ndarray:
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if "ImagePositionPatient" in s else int(s.InstanceNumber))
    return np.stack([s.pixel_array.astype(np.float32) * float(s.get("RescaleSlope", 1)) + float(s.get("RescaleIntercept", 0))
                     for s in slices])


def _envelope(model, slices, t0, **body):
    return {"model": model, "model_version": VERSIONS[model],
            "study_instance_uid": str(slices[0].StudyInstanceUID),   # the ANONYMISED UID -- models never see the real one
            "n_images": len(slices), **body, "processing_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/infer/lung-nodule")
def lung_nodule(files: list[bytes] = Depends(dicom_parts)):
    t0 = time.perf_counter()
    slices = _largest_consistent(_load(files))
    vol = _volume(slices)
    spacing = float(slices[0].PixelSpacing[0])
    findings = detect(vol, spacing)
    overlays, lines, key = [], [], None
    for i, f in enumerate(findings, 1):
        f["key_sop_instance_uid"] = key = str(slices[f["slice_index"]].SOPInstanceUID)
        overlays.append({"cx": f["center_px"][0], "cy": f["center_px"][1],
                         "r": f["diameter_mm"] / spacing / 2 + 3, "label": ""})
        lines += [(f"Finding {i}", f"pulmonary nodule, {f['laterality']} lung"),
                  (f"Finding {i} size", f"{f['diameter_mm']:.1f} mm"),
                  (f"Finding {i} confidence", f"{f['confidence']:.2f}")]
    if not findings:
        lines = [("Result", "No finding detected by model")]
    return _envelope("lung-nodule", slices, t0, result="POSITIVE" if findings else "NEGATIVE",
                     summary_lines=lines, key_sop_instance_uid=key, overlays=overlays,
                     display_window=[-600, 1500], findings=findings)


@app.post("/infer/ct-qa")
def ct_qa(files: list[bytes] = Depends(dicom_parts)):
    t0 = time.perf_counter()
    slices = _load(files)
    by_series = defaultdict(list)
    for s in slices:
        by_series[(str(s.SeriesInstanceUID), s.Rows, s.Columns)].append(s)

    rank = {"FAIL": 0, "NOT_A_PHANTOM": 1, "MEASURED": 2, "PASS": 3}
    series_results = []
    for (uid, _, _), ss in by_series.items():
        vol = _volume(ss)
        r = analyse_series(vol, float(ss[0].PixelSpacing[0]))
        r.update(series_instance_uid=uid, series_description=str(ss[0].get("SeriesDescription", "")),
                 kvp=ss[0].get("KVP"), slice_thickness=ss[0].get("SliceThickness"),
                 key_sop_instance_uid=str(ss[r["slice_index"]].SOPInstanceUID))
        series_results.append(r)
    series_results.sort(key=lambda r: rank[r["result"]])
    worst = series_results[0]

    lines = [("Result", worst["result"])]
    for i, r in enumerate(series_results, 1):
        tag = f"Series {i}" + (f" ({r['kvp']} kV, {r['slice_thickness']} mm)" if r.get("kvp") else "")
        if r["result"] == "NOT_A_PHANTOM":
            lines.append((tag, f"not graded: {r['reason']}"))
            continue
        lines.append((tag, f"CT {r['ct_hu']:+.1f} HU ({r['material']}), noise {r['noise_hu']:.1f} HU, "
                           f"uniformity {r['uniformity_hu']:.1f} HU"))
        lines += [(f"{tag} note", n) for n in r["notes"]]
    overlays = [{"cx": o["cx"], "cy": o["cy"], "r": o["r"], "label": f"{o['label']}:{o['mean']:+.0f}"}
                for o in worst.get("rois", [])]
    return _envelope("ct-qa", slices, t0, result=worst["result"], summary_lines=lines,
                     key_sop_instance_uid=worst["key_sop_instance_uid"], overlays=overlays,
                     display_window=[round(worst.get("ct_hu", 0)), 100], metrics=series_results)
