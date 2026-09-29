"""One command, whole pipeline: mini-PACS + mock AI + router + synthetic CT scanner.

  dicom-ai-router demo                 # 12 studies (from a checkout: python run_demo.py)
  dicom-ai-router demo --count 20 --seed 3
  dicom-ai-router demo --fail-rate 0.4 # watch the router retry a flaky AI endpoint

Then look in ./data:
  pacs/<study>/OT_*.png   the AI key images, as a radiologist would see them
  audit.jsonl             every hop, with timings
  results/*.json          AI output per study  } inputs to the llm-eval-radiology stage
  truth/*.json            what was really there }
QA truth "QA FAIL (+9/2)" = injected CT-number drift +9 HU / cupping 2 HU.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import threading
import time
from pathlib import Path

import uvicorn

from .config import load_config
from .router import Router
from .tools.mini_pacs import MiniPacs
from .tools.modality_sim import send_folder, send_studies

# The demo config: config/router.yaml in a checkout, a copy inside the installed wheel.
_CONFIG_CANDIDATES = (Path(__file__).resolve().parent / "_data" / "router.yaml",
                      Path(__file__).resolve().parent.parent / "config" / "router.yaml")


def demo_config_path() -> Path:
    for p in _CONFIG_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit("dicom-ai-router: demo config router.yaml not found (broken install?)")


def start_mock_ai(port: int, timeout: float = 20) -> uvicorn.Server:
    from .mock_ai.app import app
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while not server.started:
        if not thread.is_alive() or time.time() > deadline:
            raise SystemExit(f"model server did not start on port {port} (port in use?). "
                             f"Try again with --base-port 21112")
        time.sleep(0.05)
    return server


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(prog="dicom-ai-router demo")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workdir", default="data")
    ap.add_argument("--fail-rate", type=float, default=0.0, help="fraction of AI calls that return HTTP 503 (random)")
    ap.add_argument("--fail-first", type=int, default=0, help="every study's first N AI calls return HTTP 503")
    ap.add_argument("--keep", action="store_true", help="do not wipe the workdir first")
    ap.add_argument("--base-port", type=int, default=11112)
    ap.add_argument("--real", type=Path, help="also send every DICOM file under this folder (e.g. real_world/samples)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    work = Path(a.workdir)
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    saved_env = {k: os.environ.get(k) for k in ("MOCK_AI_FAIL_RATE", "MOCK_AI_FAIL_FIRST")}
    os.environ["MOCK_AI_FAIL_RATE"] = str(a.fail_rate)
    os.environ["MOCK_AI_FAIL_FIRST"] = str(a.fail_first)
    try:
        return _run(a, work)
    finally:
        for k, v in saved_env.items():               # do not leak fault injection into the caller
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(a, work: Path) -> list[dict]:

    router_port, pacs_port, ai_port = a.base_port, a.base_port + 1, a.base_port + 388
    # A throwaway secret salt for this demo run (the real router takes it from AIROUTER_DEID_SALT).
    cfg = load_config(demo_config_path(), {
        "router": {"workdir": str(work), "port": router_port, "study_quiet_seconds": 1.5,
                   "bind_address": "127.0.0.1"},
        "deid": {"salt": os.environ.get("AIROUTER_DEID_SALT") or secrets.token_hex(32)},
        "pacs": {"port": pacs_port},
        "ai_endpoints": {m: {"url": f"http://127.0.0.1:{ai_port}/infer/{m}", "retries": 5}
                         for m in ("lung-nodule", "ct-qa")},
    })

    print("\n[1/4] starting mini-PACS, mock AI model and router")
    pacs = MiniPacs(work / "pacs", pacs_port).start()
    ai = start_mock_ai(ai_port)
    try:
        router = Router(cfg).start()
    except OSError as e:
        pacs.stop()
        ai.should_exit = True
        raise SystemExit(f"router could not listen on port {router_port} ({e.strerror or e}). "
                         f"Try again with --base-port 21112") from None

    print(f"[2/4] synthetic CT scanner sending {a.count} studies to AIROUTER:{router_port}")
    truths = {t["study_uid"]: t for t in send_studies(a.count, a.seed, "127.0.0.1", router_port, "AIROUTER", work / "truth")}
    n_real = 0
    if a.real:
        n_real = send_folder(a.real, "127.0.0.1", router_port)
        print(f"  sent {n_real} real (scrubbed) studies from {a.real}")

    print("[3/4] router working: study-complete timer -> route -> de-id -> AI -> SR + key image -> PACS")
    rows = router.wait_idle(a.count + n_real, timeout=180)

    print("[4/4] results\n")
    print(f"  {'patient':18s} {'truth':20s} {'router':8s} {'AI said':50s} {'PACS got'}")
    for r in rows:
        t = truths.get(r["study_uid"], {})
        if t.get("kind") == "qa":
            e = t["qa_expected"]
            truth = f"QA {e['result']} ({t['qa_injected']['bias_hu']:+.0f}/{t['qa_injected']['cup_hu']:.0f})"
        elif t.get("kind") == "head":
            truth = "head CT"
        elif t.get("nodule"):
            n = t["nodule"]
            truth = f"{n['diameter_mm']:.0f}mm {n['laterality']}" + (" GGO" if n.get("density") == "ground-glass" else "")
        else:
            truth = "no nodule" if t else "(real, scrubbed)"
        got = sorted(m for s, m in pacs.received if s == r["study_uid"])
        said = r["error"] if r["state"] == "FAILED" else (r["summary"] or "")
        said = said.replace("Finding 1: ", "").replace("Finding 1 size: ", "").replace("Finding 1 confidence: ", "conf ")
        said = said.replace("No finding detected by model", "no finding")
        name = t.get("patient_name") or ("real phantom" if r["calling_ae"] == "REAL_QA" else "?")
        print(f"  {name:18s} {truth:20s} {r['state']:8s} {said[:50]:50s} {','.join(got) or '-'}")

    steps = [json.loads(line) for line in (work / "audit.jsonl").read_text().splitlines()]
    ai_ms = [s["ms"] for s in steps if s["step"] == "ai_inference" and s.get("ok")]
    if ai_ms:
        print(f"\n  AI round-trip: median {sorted(ai_ms)[len(ai_ms) // 2]:.0f} ms over {len(ai_ms)} studies")
    print(f"  audit trail: {work / 'audit.jsonl'}   key images: {work / 'pacs'}/*/OT_*.png\n")

    router.stop()
    pacs.stop()
    ai.should_exit = True
    return rows


if __name__ == "__main__":
    main()
