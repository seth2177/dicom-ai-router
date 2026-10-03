"""Send de-identified images to an AI model over HTTP; get findings back as JSON.

Two transports, chosen per endpoint (`transport:` in router.yaml):

  multipart  HTTP multipart/form-data upload of DICOM Part-10 files, field 'files'.
             A simple stand-in for a vendor API. The default.
  stow-rs    DICOMweb STOW-RS request (PS3.18): POST multipart/related; type="application/dicom",
             one Part-10 instance per part. The model still answers with the findings JSON below,
             not a Store Instances Response.

Same retry rules for both: 5xx and network errors are retried with backoff, 4xx never.
"""
from __future__ import annotations

import io
import logging
import time
import uuid

import httpx
from pydicom import Dataset, dcmwrite

from .config import EndpointCfg

log = logging.getLogger("airouter.ai")


class AIError(RuntimeError):
    pass


def infer(endpoint: EndpointCfg, datasets: list[Dataset]) -> dict:
    blobs = [_to_bytes(ds) for ds in datasets]
    if endpoint.transport == "stow-rs":
        body, content_type = stow_body(blobs)
        request = {"content": body, "headers": {"Content-Type": content_type, "Accept": "application/json"}}
    else:
        request = {"files": [("files", (f"{i:04d}.dcm", b, "application/dicom")) for i, b in enumerate(blobs)]}
    last = "no attempt"
    for attempt in range(1, endpoint.retries + 1):
        try:
            r = httpx.post(endpoint.url, timeout=endpoint.timeout_s, **request)
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


def stow_body(blobs: list[bytes]) -> tuple[bytes, str]:
    """PS3.18 STOW-RS request body: multipart/related, each part one application/dicom instance."""
    boundary = uuid.uuid4().hex
    while any(boundary.encode() in b for b in blobs):          # the boundary must not occur in the content
        boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for b in blobs:
        out.write(f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode())
        out.write(b)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f'multipart/related; type="application/dicom"; boundary={boundary}'


def _to_bytes(ds: Dataset) -> bytes:
    buf = io.BytesIO()
    dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()
