"""Send de-identified images to an AI model over HTTP; get findings back as JSON.

Transport: an HTTP multipart/form-data upload of DICOM Part-10 files. This is a
simple stand-in for a vendor API. A standards-based deployment would send a
DICOMweb STOW-RS request instead (multipart/related; type="application/dicom"),
which is a transport change in this one module.
"""
from __future__ import annotations

import io
import logging
import time

import httpx
from pydicom import Dataset, dcmwrite

from .config import EndpointCfg

log = logging.getLogger("airouter.ai")


class AIError(RuntimeError):
    pass


def infer(endpoint: EndpointCfg, datasets: list[Dataset]) -> dict:
    files = [("files", (f"{i:04d}.dcm", _to_bytes(ds), "application/dicom")) for i, ds in enumerate(datasets)]
    last = "no attempt"
    for attempt in range(1, endpoint.retries + 1):
        try:
            r = httpx.post(endpoint.url, files=files, timeout=endpoint.timeout_s)
            if r.status_code == 200:
                out = r.json()
                out["_attempts"] = attempt
                return out
            # 4xx = our request is wrong; retrying will not help.
            if 400 <= r.status_code < 500:
                raise AIError(f"model rejected request: HTTP {r.status_code} {r.text[:200]}")
            last = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}"
        log.warning("AI call failed (attempt %d/%d): %s", attempt, endpoint.retries, last)
        if attempt < endpoint.retries:
            time.sleep(0.5 * 2 ** (attempt - 1))
    raise AIError(f"AI endpoint unavailable after {endpoint.retries} attempts: {last}")


def _to_bytes(ds: Dataset) -> bytes:
    buf = io.BytesIO()
    dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()
