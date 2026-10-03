"""Turn the AI's JSON answer into DICOM objects a PACS and a radiologist can use.

Two objects go back per study, both inside the ORIGINAL study (same
StudyInstanceUID, real patient demographics) as new series:

  1. Basic Text SR  -- structured findings; RIS / reporting tools can parse it.
  2. Secondary Capture key image -- the slice with the finding circled and a
     "NOT FOR DIAGNOSIS" banner, so a radiologist sees it in any viewer.

Both are marked VerificationFlag / labelling as unverified AI output.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw
from pydicom import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

BASIC_TEXT_SR = "1.2.840.10008.5.1.4.1.1.88.11"
SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"

# Patient / General Study attributes copied from the source. Type 2 ones are always written
# (empty if the source lacks them) because receivers are entitled to expect them present.
PATIENT_STUDY_TYPE2 = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                       "StudyDate", "StudyTime", "AccessionNumber", "ReferringPhysicianName", "StudyID")
PATIENT_STUDY_OPTIONAL = ("StudyDescription",)
PRIVATE_SCHEME = "99AIROUTER"

BANNER = "AI RESULT - NOT FOR DIAGNOSIS"


def result_uid(orig: Dataset, model_key: str, part: str, content: str = "") -> str:
    """Deterministic UIDs for result objects.

    Series UID: study + model + object type, so every version of a result lives in ONE series.
    Instance UID: additionally keyed on a digest of the result content. DICOM requires changed
    content to carry a new SOP Instance UID, so:
      * a re-run with the same findings re-sends the identical object (no duplicate in PACS);
      * a re-run with different findings (late images) adds a new instance in the same series,
        which a PACS will not silently discard as a duplicate.
    """
    return generate_uid(entropy_srcs=[str(orig.StudyInstanceUID), model_key, part, content])


def content_digest(ai: dict) -> str:
    """What the result says, independent of timing fields."""
    keep = {k: ai.get(k) for k in ("model", "model_version", "result", "summary_lines", "key_sop_instance_uid", "overlays")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def _base(orig: Dataset, sop_class: str, modality: str, series_number: int, series_desc: str,
          model_key: str = "", content: str = "") -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    for kw in PATIENT_STUDY_TYPE2:
        setattr(ds, kw, orig.data_element(kw).value if kw in orig else "")
    for kw in PATIENT_STUDY_OPTIONAL:
        if kw in orig:
            setattr(ds, kw, orig.data_element(kw).value)
    ds.StudyInstanceUID = orig.StudyInstanceUID
    now = dt.datetime.now()
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = result_uid(orig, model_key, f"{modality}-instance", content)
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesInstanceUID = result_uid(orig, model_key, f"{modality}-series")
    ds.Modality = modality
    ds.SeriesNumber = series_number
    ds.SeriesDescription = series_desc
    ds.InstanceNumber = 1
    ds.Manufacturer = "dicom-ai-router (demo)"
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.SpecificCharacterSet = "ISO_IR 100"
    return ds


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    c = Dataset()
    c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
    return c


def _text_item(index: int, name: str, text: str) -> Dataset:
    """TEXT content item. Each gets its own private code (AIR0001, AIR0002, ...) whose meaning is the
    label, so two labels can never collide into one code value."""
    it = Dataset()
    it.RelationshipType = "CONTAINS"
    it.ValueType = "TEXT"
    it.ConceptNameCodeSequence = Sequence([_code(f"AIR{index:04d}", PRIVATE_SCHEME, name[:64])])
    it.TextValue = text
    return it


def _scheme_declaration() -> Sequence:
    s = Dataset()
    s.CodingSchemeDesignator = PRIVATE_SCHEME
    s.CodingSchemeName = "dicom-ai-router result labels"
    s.CodingSchemeResponsibleOrganization = "dicom-ai-router"
    return Sequence([s])


def finding_lines(ai: dict) -> list[tuple[str, str]]:
    """Human-readable (label, value) pairs -- shared by the SR and the key image.

    Comes straight from the model's `summary_lines`, so a new model needs no
    router code: it just answers with the same contract.
    """
    lines = [("Model", f"{ai.get('model')} v{ai.get('model_version')}")]
    lines += [(str(a), str(b)) for a, b in ai.get("summary_lines", [])]
    lines.append(("Status", "Unverified AI output - requires review"))
    return lines


def build_sr(orig_first: Dataset, originals: list[Dataset], ai: dict, key_sop_uid: str | None,
             predecessor: dict | None = None) -> Dataset:
    """predecessor: the SR this one revises ({sop_uid, series_uid, sop_class}, from Store.sent_sr), or None.
    It is part of the instance UID: the same findings revising a different SR is different content."""
    content = content_digest(ai) + (f"|{predecessor['sop_uid']}" if predecessor else "")
    ds = _base(orig_first, BASIC_TEXT_SR, "SR", 9901, f"AI Results - {ai.get('model')} (NOT FOR DIAGNOSIS)",
               model_key=str(ai.get("model")), content=content)
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([_code("18748-4", "LN", "Diagnostic imaging report")])
    ds.CodingSchemeIdentificationSequence = _scheme_declaration()
    # SR Document General module, Type 2: present, empty (no procedure-step linkage here)
    ds.PerformedProcedureCodeSequence = Sequence([])
    ds.ReferencedPerformedProcedureStepSequence = Sequence([])
    if predecessor:
        # SR Document General module, Type 1C: required when this document replaces an earlier one.
        # Hierarchical SOP Instance Reference: study > series > instance.
        ref = Dataset()
        ref.ReferencedSOPClassUID = predecessor["sop_class"]
        ref.ReferencedSOPInstanceUID = predecessor["sop_uid"]
        series = Dataset()
        series.SeriesInstanceUID = predecessor["series_uid"]
        series.ReferencedSOPSequence = Sequence([ref])
        study = Dataset()
        study.StudyInstanceUID = orig_first.StudyInstanceUID
        study.ReferencedSeriesSequence = Sequence([series])
        ds.PredecessorDocumentsSequence = Sequence([study])

    items = [_text_item(i, label, value) for i, (label, value) in enumerate(finding_lines(ai), 1)]
    if key_sop_uid:
        key = next(o for o in originals if o.SOPInstanceUID == key_sop_uid)
        img = Dataset()
        img.RelationshipType = "CONTAINS"
        img.ValueType = "IMAGE"
        img.ConceptNameCodeSequence = Sequence([_code("121112", "DCM", "Source of Measurement")])
        ref = Dataset()
        ref.ReferencedSOPClassUID = key.SOPClassUID
        ref.ReferencedSOPInstanceUID = key.SOPInstanceUID
        img.ReferencedSOPSequence = Sequence([ref])
        items.append(img)
    ds.ContentSequence = Sequence(items)

    # Evidence: the images this report is about, grouped by their own series.
    by_series: dict[str, list[Dataset]] = {}
    for o in originals:
        by_series.setdefault(str(o.SeriesInstanceUID), []).append(o)
    series_items = []
    for series_uid, members in by_series.items():
        series = Dataset()
        series.SeriesInstanceUID = series_uid
        refs = []
        for o in members:
            r = Dataset()
            r.ReferencedSOPClassUID = o.SOPClassUID
            r.ReferencedSOPInstanceUID = o.SOPInstanceUID
            refs.append(r)
        series.ReferencedSOPSequence = Sequence(refs)
        series_items.append(series)
    study = Dataset()
    study.StudyInstanceUID = orig_first.StudyInstanceUID
    study.ReferencedSeriesSequence = Sequence(series_items)
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([study])
    return ds


def build_key_image(key: Dataset, ai: dict, is_key: bool = True) -> Dataset:
    """Secondary Capture: windowed slice, overlays drawn, banner burned in.
    is_key=False (no finding): a representative slice, labelled as such rather than as a key image."""
    hu = key.pixel_array.astype(np.float32) * float(key.get("RescaleSlope", 1)) + float(key.get("RescaleIntercept", 0))
    level, width = ai.get("display_window") or (40, 400)
    img8 = np.clip((hu - (level - width / 2)) / width * 255, 0, 255).astype(np.uint8)

    scale = max(1, 512 // max(img8.shape))               # upscale small images so text is legible
    im = Image.fromarray(img8).resize((img8.shape[1] * scale, img8.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    for o in ai.get("overlays", []):
        cx, cy, r = o["cx"] * scale, o["cy"] * scale, o["r"] * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=2)
        if o.get("label"):
            draw.rectangle([cx - r, cy - r - 13, cx - r + 7 * len(o["label"]), cy - r - 1], fill=0)
            draw.text((cx - r + 1, cy - r - 13), o["label"], fill=255)
    draw.rectangle([0, 0, im.width, 16], fill=0)
    draw.text((4, 3), BANNER, fill=255)
    caption = finding_lines(ai)
    y = im.height - 14 * len(caption) - 4
    draw.rectangle([0, y - 2, im.width, im.height], fill=0)
    for label, value in caption:
        draw.text((4, y), f"{label}: {value}"[: im.width // 6], fill=255)
        y += 14

    arr = np.asarray(im, dtype=np.uint8)
    kind = "Key Image" if is_key else "Result Image"
    ds = _base(key, SECONDARY_CAPTURE, "OT", 9902, f"AI {kind} - {ai.get('model')} (NOT FOR DIAGNOSIS)",
               model_key=str(ai.get("model")), content=content_digest(ai))
    ds.ConversionType = "WSD"
    ds.PatientOrientation = ""                    # General Image module, Type 2
    ds.BurnedInAnnotation = "YES"
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    src = Dataset()
    src.ReferencedSOPClassUID = key.SOPClassUID
    src.ReferencedSOPInstanceUID = key.SOPInstanceUID
    ds.SourceImageSequence = Sequence([src])
    return ds
