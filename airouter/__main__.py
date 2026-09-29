"""Run the router as a long-lived service, inspect its state, re-run studies, or run the demo.

Installed from PyPI, `dicom-ai-router` is the same command as `python -m airouter`.

  python -m airouter demo   [--count 12] [--fail-rate 0.4] ...   the whole pipeline, synthetic data
  python -m airouter serve  [--config config/router.yaml]
  python -m airouter status [--config config/router.yaml]
  python -m airouter rerun  --failed | <StudyInstanceUID> [...]   [--config config/router.yaml]
"""
from __future__ import annotations

import argparse
import logging
import socket
import sys
import time

from .audit import Audit
from .config import load_config
from .deid import WeakSaltError
from .pipeline import inbox_folder, process_study
from .router import Router
from .store import Store


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["demo"]:
        from .demo import main as demo
        demo(argv[1:])
        return
    ap = argparse.ArgumentParser(prog="airouter")
    ap.add_argument("command", choices=["demo", "serve", "status", "rerun"])
    ap.add_argument("studies", nargs="*", help="rerun: StudyInstanceUIDs to re-run")
    ap.add_argument("--failed", action="store_true", help="rerun: every study currently FAILED")
    ap.add_argument("--force", action="store_true", help="rerun: run even if the service seems to be up")
    ap.add_argument("--config", default="config/router.yaml")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config, require_salt=(a.command != "status"))
    except WeakSaltError as e:
        raise SystemExit(f"airouter: {e}") from None

    if a.command == "status":
        rows = Store(cfg.db_path).studies()
        if not rows:
            print("no studies yet")
        for r in rows:
            detail = r["error"] or r["summary"] or ""
            print(f"{r['state']:10s} {r['n_instances']:4d} img  {r['calling_ae'] or '':10s} {r['study_uid'][-24:]}  {detail}")
        return

    if a.command == "rerun":
        # Refuse while the service is listening: both would process the same study at once.
        if _port_in_use(cfg.bind_address, cfg.port) and not a.force:
            raise SystemExit(f"airouter rerun: something is listening on port {cfg.port} (the router service?). "
                             "Stop it first, or pass --force if you are sure it is not processing these studies.")
        store, audit = Store(cfg.db_path), Audit(cfg.audit_path)
        targets = list(a.studies)
        if a.failed:
            targets += [r["study_uid"] for r in store.studies() if r["state"] == "FAILED"]
        if not targets:
            raise SystemExit("airouter rerun: give StudyInstanceUIDs or --failed")
        for uid in targets:
            if not inbox_folder(cfg, uid).exists():
                print(f"skip {uid}: no inbox copy")
                continue
            audit.log(uid, "manual_rerun")
            print(f"{process_study(uid, cfg, store, audit):8s} {uid}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        router = Router(cfg).start()
    except OSError as e:
        raise SystemExit(f"airouter: cannot listen on {cfg.bind_address}:{cfg.port} ({e.strerror or e}). "
                         "Is another router or PACS already using that port?") from None
    print(f"AIROUTER up: DICOM {cfg.ae_title}@{cfg.bind_address}:{cfg.port} -> AI -> "
          f"{cfg.pacs.ae_title}@{cfg.pacs.host}:{cfg.pacs.port}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        router.stop()


if __name__ == "__main__":
    main()
