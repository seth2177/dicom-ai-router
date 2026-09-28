"""Site-level scrub: remove every trace of WHERE an image came from.

`deid.deidentify` already removes identity and site attributes, every date,
every UID and pattern-matched text. Publishing images from a customer's
scanner needs one more thing: the site's own vocabulary. An institution name,
a city, a station or AE title, a serial or an operator's name can turn up in
*any* free-text field (a comment, a protocol name, a vendor private tag).

So this module:
  1. harvests site tokens from the ORIGINAL headers and the package names
     (`harvest_tokens`, `tokens_from_filename`), plus any added by hand, and
  2. runs the patient profile with those tokens, so any text value containing
     one is redacted.

Tokens are secrets: they are kept in a *.local.json file that never goes to
git, and are never printed in the audit report.
"""
from __future__ import annotations

import re

from pydicom import Dataset

from . import deid

SITE_KEYWORDS = (
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "StationName",
    "DeviceSerialNumber", "OperatorsName", "PerformingPhysicianName", "ReferringPhysicianName",
    "NameOfPhysiciansReadingStudy", "RequestingPhysician", "ScheduledPerformingPhysicianName",
    "RetrieveAETitle", "StationAETitle", "ScheduledStationAETitle", "PerformedStationAETitle",
    "PerformedLocation", "ScheduledProcedureStepLocation", "AccessionNumber",
)
META_AE = ("SourceApplicationEntityTitle", "SendingApplicationEntityTitle", "ReceivingApplicationEntityTitle")

# Words too generic to be a site identifier on their own. A site called "X Scan Center" must not
# cause every "QA Phantom Scan" description to be redacted.
GENERIC = {
    "MEDICAL", "CENTER", "CENTRE", "HOSPITAL", "IMAGING", "REGIONAL", "HEALTH", "HEALTHCARE", "RADIOLOGY",
    "CLINIC", "DEPT", "DEPARTMENT", "SYSTEM", "SYSTEMS", "GENERAL", "COUNTY", "MEMORIAL", "UNIVERSITY",
    "PHILIPS", "SIEMENS", "TOSHIBA", "CANON", "HEALTHINEERS", "BRILLIANCE", "INGENUITY", "INCISIVE",
    "DAILY", "WATER", "PHANTOM", "TEST", "SERVICE", "SERVICES", "ROOM", "SCANNER", "DEID",
    "ARCHIVE", "FULL", "NONE", "UNKNOWN", "ANONYMOUS", "OPERATOR", "ADMIN",
    "SCAN", "SCANS", "CHECK", "BODY", "HEAD", "CHEST", "BRAIN", "NECK", "SPINE", "PELVIS", "ABDOMEN",
    "AXIAL", "SPIRAL", "HELICAL", "HEART", "CARDIAC", "PHYSICS", "PROCEDURE", "QUALITY", "CONTROL",
    "IMAGE", "IMAGES", "DIAGNOSTIC", "DIAGNOSTICS", "OUTPATIENT", "INPATIENT", "EMERGENCY",
}
# Words that appear in service-package names as error text, not names ("XRAY_SYSTEM_CANNOT_COMPLY").
ERROR_WORDS = {
    "ALIGNMENT", "AUDIT", "CANNOT", "CHANGE", "COLLECTION", "COMPLY", "CRITICAL", "ENOUGH",
    "ERROR", "ERRORS", "FAILING", "FAILURE", "FRAME", "FROZE", "GANTRY", "HISTORICAL", "HOST", "INITIALIZE",
    "ISSUES", "MACHINE", "MANY", "MEMORY", "MULTIPLE", "POST", "PROBLEM", "RECONS", "REPLY", "RETRY",
    "ROTATION", "ROTATING", "SHUT", "SHUTDOWN", "STOPPED", "STOPS", "TABLE", "TUBE", "TWICE", "XRAY", "SAME",
    "WITH", "WORK", "WORKING", "DURING", "UNABLE", "PLEASE", "THERE", "SEEMS", "WOULD", "LIKE", "CREATE",
    "REPORT", "APPLICATION", "SHORT", "CONDITIONING", "CALCIUM", "ONLY",
}
MONTHS = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC"}


def _values(elem) -> list[str]:
    """All values of an element as strings. pydicom's MultiValue is not a list, so check VM."""
    if elem.VR == "SQ" or elem.value in (None, ""):
        return []
    return [str(v) for v in (elem.value if elem.VM > 1 else [elem.value])]


def _split(value: str) -> set[str]:
    out = set()
    value = value.replace("^", " ").strip()
    if len(value) >= 3:
        out.add(value.upper())
    for word in re.split(r"[\s,._/\-]+", value):
        if len(word) >= 4 and not word.isdigit() and word.upper() not in GENERIC | ERROR_WORDS:
            out.add(word.upper())
        elif word.isdigit() and len(word) >= 5:           # serial numbers
            out.add(word)
    return out


def harvest_tokens(datasets: list[Dataset], extra: list[str] = ()) -> set[str]:
    """Site identifiers from the originals: whole values plus their distinctive words."""
    tokens: set[str] = set()
    for t in extra:
        if t:
            tokens |= _split(t)
    for ds in datasets:
        def visit(_, elem):
            if elem.keyword in SITE_KEYWORDS:
                for v in _values(elem):
                    tokens.update(_split(v))
        ds.walk(visit)                                     # nested sequences too
        meta = getattr(ds, "file_meta", None)
        for kw in META_AE:
            if meta is not None and kw in meta:
                tokens.update(_split(str(meta.data_element(kw).value)))
    return tokens


def tokens_from_filename(name: str) -> set[str]:
    """Package names often carry the site: DEID_2026_07_29_01_52_10_Lakeside_PM_Jul_26.zip"""
    words = re.split(r"[\s,._/\-]+", re.sub(r"\.(zip|7z|tar|gz)$", "", name, flags=re.I))
    return {w.upper() for w in words if w.isalpha() and len(w) >= 4 and w.upper() not in GENERIC | MONTHS | ERROR_WORDS}


def scrub(ds: Dataset, salt: str, tokens: set[str]) -> tuple[Dataset, list[tuple[str, str, str]]]:
    """Patient profile + site tokens. Returns (clean copy, UID crosswalk pairs)."""
    out, pairs = deid.deidentify(ds, salt, reject_burned_in=True, tokens=tokens)
    for kw in SITE_KEYWORDS:
        if kw in out and kw not in deid.EMPTY:          # Type 2 ones stay present, emptied by deid
            del out[kw]
    out.DeidentificationMethod = "dicom-ai-router site scrub v2 (PS3.15 subset+site)"   # LO: max 64
    return out, pairs
