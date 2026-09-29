"""Tiny stand-in PACS: a C-STORE receiver that files whatever it gets.

Stores to data/pacs/<StudyInstanceUID>/<Modality>_<SOPInstanceUID>.dcm and
writes a PNG of every Secondary Capture so you can open the AI key image
without a DICOM viewer. Use Orthanc (docker-compose.yml) for a real PACS UI.

  python -m airouter.tools.mini_pacs --port 11113
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE, AllStoragePresentationContexts, evt

from airouter.router import UID_RX

log = logging.getLogger("mini_pacs")


class MiniPacs:
    def __init__(self, root: Path, port: int = 11113, ae_title: str = "PACS", bind: str = "127.0.0.1"):
        self.root, self.port, self.ae_title, self.bind = root, port, ae_title, bind
        self.received: list[tuple[str, str]] = []   # (study_uid, modality)
        self._server = None

    def _on_store(self, event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        uids = [str(ds.get(k, "")) for k in ("StudyInstanceUID", "SOPInstanceUID")]
        if not all(len(u) <= 64 and UID_RX.match(u) for u in uids):
            return 0xC000                        # never build a path from an invalid UID
        folder = self.root / str(ds.StudyInstanceUID)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{ds.Modality}_{ds.SOPInstanceUID}.dcm"
        ds.save_as(path, enforce_file_format=True)
        if ds.Modality == "OT":
            Image.fromarray(ds.pixel_array).save(path.with_suffix(".png"))
        self.received.append((str(ds.StudyInstanceUID), str(ds.Modality)))
        log.info("PACS stored %s for %s", ds.Modality, ds.get("PatientName"))
        return 0x0000

    def start(self):
        ae = AE(ae_title=self.ae_title)
        ae.require_called_aet = True
        for cx in AllStoragePresentationContexts:
            ae.add_supported_context(cx.abstract_syntax, [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
        self._server = ae.start_server((self.bind, self.port), block=False,
                                       evt_handlers=[(evt.EVT_C_STORE, self._on_store)])
        return self

    def stop(self):
        if self._server:
            self._server.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11113)
    ap.add_argument("--root", default="data/pacs")
    ap.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to accept senders on other machines")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    MiniPacs(Path(a.root), a.port, bind=a.bind).start()
    print(f"mini PACS listening as PACS on {a.port}; Ctrl+C to stop")
    import time
    while True:
        time.sleep(3600)
