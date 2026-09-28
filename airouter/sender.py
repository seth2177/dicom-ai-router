"""DICOM C-STORE client (SCU): push datasets to PACS, with retry and backoff."""
from __future__ import annotations

import logging
import time

from pydicom import Dataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE

log = logging.getLogger("airouter.sender")


class SendError(RuntimeError):
    pass


def c_store(datasets: list[Dataset], host: str, port: int, called_ae: str,
            calling_ae: str = "AIROUTER", retries: int = 3, backoff_s: float = 1.0) -> int:
    """Send all datasets on one association. Returns count sent. Raises SendError."""
    last_err = "no attempt"
    for attempt in range(1, retries + 1):
        try:
            return _send_once(datasets, host, port, called_ae, calling_ae)
        except SendError as e:
            last_err = str(e)
            log.warning("C-STORE to %s@%s:%s failed (attempt %d/%d): %s",
                        called_ae, host, port, attempt, retries, e)
            if attempt < retries:
                time.sleep(backoff_s * 2 ** (attempt - 1))
    raise SendError(f"gave up after {retries} attempts: {last_err}")


def _send_once(datasets, host, port, called_ae, calling_ae) -> int:
    ae = AE(ae_title=calling_ae)
    # Propose exactly the SOP classes we are about to send -- one presentation
    # context each. Proposing every storage class is a common reason
    # associations get rejected by picky PACS.
    for sop_class in sorted({str(ds.SOPClassUID) for ds in datasets}):
        ae.add_requested_context(sop_class, [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
    assoc = ae.associate(host, port, ae_title=called_ae)
    if not assoc.is_established:
        raise SendError(f"association rejected/aborted by {called_ae}@{host}:{port}")
    sent = 0
    try:
        for ds in datasets:
            status = assoc.send_c_store(ds)
            # 0x0000 success; 0xB000 / 0xB006 / 0xB007 are Warning (stored, with coercion) -- also stored.
            if not status or status.Status not in (0x0000, 0xB000, 0xB006, 0xB007):
                code = hex(status.Status) if status else "no response"
                raise SendError(f"C-STORE status {code} for SOP {ds.SOPInstanceUID}")
            sent += 1
    finally:
        assoc.release()
    return sent
