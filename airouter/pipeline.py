"""What happens to one complete study, step by step.

  load -> route -> de-identify -> AI inference -> re-identify -> build SR + key image -> C-STORE to PACS

Every step is audited with its timing. Any failure marks the study FAILED
with the reason and leaves the inbox copy in place, so nothing is lost:
`python -m airouter rerun --failed` re-runs it once the fault is fixed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydicom import dcmread

from . import ai_client, deid, results, sender
from .audit import Audit
from .config import Config
from .rules import match_rule
from .store import Store

log = logging.getLogger("airouter.pipeline")


def process_study(study_uid: str, cfg: Config, store: Store, audit: Audit) -> str:
    folder = cfg.inbox / _safe(study_uid)
    store.set_state(study_uid, "PROCESSING")
    try:
        # 1. Load everything that arrived, in slice order.
        with audit.step(study_uid, "load") as x:
            originals = sorted((dcmread(p) for p in folder.glob("*.dcm")), key=lambda d: int(d.get("InstanceNumber", 0)))
            x["instances"] = len(originals)
        if not originals:
            raise RuntimeError("no instances in inbox")
        # Route on the first real image. Studies often lead with a dose SR or a localizer
        # (InstanceNumber 1), whose header says little about the clinical study.
        images = [o for o in originals if str(o.SOPClassUID) in IMAGE_SOP_CLASSES
                  and "LOCALIZER" not in [str(v).upper() for v in o.get("ImageType", [])]]
        first = images[0] if images else originals[0]

        # 2. Route on the header. No rule -> nothing leaves the router.
        rule = match_rule(first, cfg.rules)
        audit.log(study_uid, "route", rule=rule.name if rule else None, model=rule.model if rule else None,
                  modality=first.get("Modality"), body_part=first.get("BodyPartExamined"),
                  description=first.get("StudyDescription"))
        if rule is None:
            store.set_state(study_uid, "IGNORED", summary="no routing rule matched")
            return "IGNORED"
        if rule.model is None:
            store.set_state(study_uid, "IGNORED", rule=rule.name, summary=f"held by rule {rule.name}")
            return "IGNORED"

        # 3. Choose which images the model gets. Real studies carry localizers, dose screens and
        #    other captures; they are dropped per image (and audited), never allowed to fail the study.
        with audit.step(study_uid, "select_images", series_policy=rule.series) as x:
            used, dropped = select_instances(originals, rule.series, cfg.reject_burned_in)
            x["used"], x["dropped"] = len(used), dropped
        if not used:
            store.set_state(study_uid, "IGNORED", rule=rule.name, summary="no eligible images for the model")
            return "IGNORED"

        # 4. De-identify. The crosswalk stays local.
        with audit.step(study_uid, "deidentify") as x:
            anon, pairs = [], []
            for ds in used:
                a, p = deid.deidentify(ds, cfg.deid_salt, cfg.reject_burned_in)
                anon.append(a)
                pairs.extend(p)
            store.put_crosswalk(pairs)
            x["anon_study"] = anon[0].StudyInstanceUID

        # 5. Inference.
        with audit.step(study_uid, "ai_inference", model=rule.model) as x:
            ai = ai_client.infer(cfg.endpoints[rule.model], anon)
            x["result"] = ai.get("result")
            x["model_ms"] = ai.get("processing_ms")
            x["attempts"] = ai.get("_attempts", 1)

        # 6. Re-identify: map the model's key image back to the real one.
        key_orig_uid = None
        if ai.get("key_sop_instance_uid"):
            key_orig_uid = store.orig_uid(ai["key_sop_instance_uid"])
            if key_orig_uid is None:
                raise RuntimeError(f"AI referenced unknown SOP {ai['key_sop_instance_uid']} (crosswalk miss)")

        # 7. Build result objects inside the ORIGINAL study.
        with audit.step(study_uid, "build_results") as x:
            key_ds = next((o for o in used if o.SOPInstanceUID == key_orig_uid), used[len(used) // 2])
            model, digest = str(ai.get("model")), results.content_digest(ai)
            predecessor = predecessor_for(store, study_uid, model, digest)
            sr = results.build_sr(used[0], used, ai, key_orig_uid, predecessor)
            out = [sr, results.build_key_image(key_ds, ai, key_orig_uid is not None)]
            x["objects"] = [o.Modality for o in out]
            if predecessor:
                x["revises"] = predecessor["sop_uid"]

        # 8. Return to PACS.
        with audit.step(study_uid, "c_store_pacs", pacs=f"{cfg.pacs.ae_title}@{cfg.pacs.host}:{cfg.pacs.port}") as x:
            x["sent"] = sender.c_store(out, cfg.pacs.host, cfg.pacs.port, cfg.pacs.ae_title,
                                       calling_ae=cfg.ae_title, retries=cfg.pacs.retries)
        store.record_sr(study_uid, model, sr, digest, predecessor["sop_uid"] if predecessor else None)

        summary = "; ".join(f"{v}" for k, v in results.finding_lines(ai)[1:-1])
        _write_result(cfg, study_uid, rule.name, ai, key_orig_uid)
        store.set_state(study_uid, "DONE", rule=rule.name, model=rule.model, summary=summary)
        return "DONE"

    except Exception as e:  # noqa: BLE001 -- every failure must land in the DB, not kill the worker
        log.exception("study %s failed", study_uid)
        store.set_state(study_uid, "FAILED", error=f"{type(e).__name__}: {e}")
        return "FAILED"


def predecessor_for(store: Store, study_uid: str, model: str, digest: str) -> dict | None:
    """The SR a new result revises. Same findings as the last one sent: re-send that SR unchanged (with
    whatever it revised). Different findings: the last one sent is the predecessor. Nothing sent yet: None."""
    last = store.latest_sr(study_uid, model)
    if last is None:
        return None
    if last["digest"] == digest:
        return store.sent_sr(last["predecessor"]) if last["predecessor"] else None
    return last


IMAGE_SOP_CLASSES = {"1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1"}   # CT, Enhanced CT


def select_instances(originals: list, series_policy: str = "all", reject_burned_in: bool = True):
    """Return (images for the model, {reason: count} dropped)."""
    dropped: dict[str, int] = {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    keep = []
    for ds in originals:
        image_type = [str(v).upper() for v in ds.get("ImageType", [])]
        if str(ds.SOPClassUID) not in IMAGE_SOP_CLASSES or "PixelData" not in ds:
            drop("not a CT image")
        elif "LOCALIZER" in image_type:
            drop("localizer")
        elif reject_burned_in and str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
            drop("burned-in annotation")
        else:
            keep.append(ds)
    if series_policy == "largest" and keep:
        by_series: dict[str, list] = {}
        for ds in keep:
            by_series.setdefault(str(ds.SeriesInstanceUID), []).append(ds)
        main = max(by_series.values(), key=len)
        for s in by_series.values():
            if s is not main:
                for _ in s:
                    drop("not the main series")
        keep = main
    return keep, dropped


def _write_result(cfg: Config, study_uid: str, rule: str, ai: dict, key_orig_uid: str | None) -> None:
    """Machine-readable record per study -- the input to the llm-eval-radiology stage."""
    record = {"study_uid": study_uid, "rule": rule, "key_sop_instance_uid": key_orig_uid,
              "ai": {k: v for k, v in ai.items() if not k.startswith("_")}}
    (cfg.results_dir / f"{_safe(study_uid)}.json").write_text(json.dumps(record, indent=2))


def _safe(uid: str) -> str:
    return "".join(c for c in uid if c.isdigit() or c == ".")


def inbox_folder(cfg: Config, study_uid: str) -> Path:
    return cfg.inbox / _safe(study_uid)
