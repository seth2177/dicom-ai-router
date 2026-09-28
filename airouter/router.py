"""The router service: DICOM listener + study-completion timer + worker pool.

Design rule #1: never make the modality wait. The C-STORE handler only
writes the file to disk and answers "Success". All real work happens later,
on a worker thread. A slow receiver backs up the scanner's send queue --
that is the most common "images are slow to arrive" call in the field.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE, AllStoragePresentationContexts, evt

from .audit import Audit
from .config import Config
from .pipeline import inbox_folder, process_study
from .store import Store

log = logging.getLogger("airouter")

# A DICOM UID is digits and dots, at most 64 characters. Anything else arriving over the
# network is rejected before it can be used in a file path ("../../escape" is not a UID).
UID_RX = re.compile(r"^[0-9]+(\.[0-9]+)*$")
STATUS_CANNOT_UNDERSTAND = 0xC000
STATUS_OUT_OF_RESOURCES = 0xA700


class Router:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.db_path)
        self.audit = Audit(cfg.audit_path)
        self._last_seen: dict[str, float] = {}
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="study")
        self._server = None

    # ---- DICOM receive (SCP) -------------------------------------------------
    def _on_c_store(self, event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        study, sop = str(ds.get("StudyInstanceUID", "")), str(ds.get("SOPInstanceUID", ""))
        if not all(len(u) <= 64 and UID_RX.match(u) for u in (study, sop)):
            log.warning("rejected C-STORE with invalid UID(s) from %s", event.assoc.requestor.ae_title)
            return STATUS_CANNOT_UNDERSTAND
        folder = inbox_folder(self.cfg, study)
        folder.mkdir(parents=True, exist_ok=True)
        # Write to a temp name, then rename: a worker never reads a half-written file.
        tmp = folder / f".{sop}.part"
        ds.save_as(tmp, enforce_file_format=True)
        for attempt in range(5):
            try:
                os.replace(tmp, folder / f"{sop}.dcm")
                break
            except PermissionError:           # Windows: a worker has the old copy open (a re-send)
                time.sleep(0.2 * (attempt + 1))
        else:
            log.error("could not store re-sent instance %s", sop)
            return STATUS_OUT_OF_RESOURCES
        calling = event.assoc.requestor.ae_title
        self.store.instance_received(study, calling)
        with self._lock:
            if study not in self._last_seen:
                self.audit.log(study, "receive_start", calling_ae=calling,
                               modality=ds.get("Modality"), sop_class=str(ds.SOPClassUID))
            self._last_seen[study] = time.monotonic()
        return 0x0000  # Success -- immediately

    # ---- Study completion ----------------------------------------------------
    def _watch(self):
        while not self._stop.wait(0.25):
            now = time.monotonic()
            with self._lock:
                done = [s for s, t in self._last_seen.items() if now - t >= self.cfg.study_quiet_seconds]
                for s in done:
                    del self._last_seen[s]
            for s in done:
                self._submit(s)

    def _submit(self, study: str) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            if study in self._inflight:
                # Images arrived while this study was being processed: run it again after.
                self._last_seen[study] = time.monotonic()
                return
            self._inflight.add(study)
        self.store.set_state(study, "QUEUED")
        self.audit.log(study, "study_complete", quiet_s=self.cfg.study_quiet_seconds)
        self._pool.submit(self._run, study)

    def _run(self, study: str) -> None:
        try:
            process_study(study, self.cfg, self.store, self.audit)
        finally:
            with self._lock:
                self._inflight.discard(study)

    def requeue_unfinished(self) -> int:
        """After a crash or restart: pick up studies that never reached a terminal state."""
        n = 0
        for row in self.store.studies():
            if row["state"] in ("RECEIVING", "QUEUED", "PROCESSING") and inbox_folder(self.cfg, row["study_uid"]).exists():
                with self._lock:
                    self._last_seen[row["study_uid"]] = 0.0          # due immediately
                n += 1
        return n

    # ---- Lifecycle -----------------------------------------------------------
    def start(self):
        ae = AE(ae_title=self.cfg.ae_title)
        ae.require_called_aet = True                # wrong called AE title -> association rejected
        if self.cfg.allowed_calling_aes:
            ae.require_calling_aet = list(self.cfg.allowed_calling_aes)
        # Accept every storage SOP class, but only the two uncompressed transfer
        # syntaxes. Negotiation is where many integrations silently fail: if the
        # modality only offers JPEG 2000, this is the line you change.
        for cx in AllStoragePresentationContexts:
            ae.add_supported_context(cx.abstract_syntax, [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
        ae.maximum_pdu_size = 0
        self._server = ae.start_server((self.cfg.bind_address, self.cfg.port), block=False,
                                       evt_handlers=[(evt.EVT_C_STORE, self._on_c_store)])
        requeued = self.requeue_unfinished()
        if requeued:
            log.info("requeued %d unfinished studies from a previous run", requeued)
        self._watch_thread = threading.Thread(target=self._watch, daemon=True, name="study-watch")
        self._watch_thread.start()
        log.info("AI router listening as %s on port %d", self.cfg.ae_title, self.cfg.port)
        return self

    def stop(self):
        """Stop accepting, finish in-flight work. Studies still waiting on their quiet timer
        stay RECEIVING in the database and are requeued by the next start()."""
        if self._server:
            self._server.shutdown()
        self._stop.set()
        if getattr(self, "_watch_thread", None):
            self._watch_thread.join(timeout=2)
        self._pool.shutdown(wait=True)

    def wait_idle(self, expected: int, timeout: float = 60) -> list[dict]:
        """Block until `expected` studies reached a terminal state (used by demo and tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rows = self.store.studies()
            with self._lock:
                busy = bool(self._last_seen or self._inflight)
            if not busy and len(rows) >= expected and all(r["state"] in ("DONE", "IGNORED", "FAILED") for r in rows):
                return rows
            time.sleep(0.25)
        raise TimeoutError(f"router not idle after {timeout}s: {self.store.studies()}")
