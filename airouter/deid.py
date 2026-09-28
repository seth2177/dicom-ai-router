"""De-identification before anything leaves the building.

Implements a working subset of the DICOM PS3.15 Annex E Basic Application
Confidentiality Profile. It is a subset, not a certified implementation:

  * PatientName / PatientID, at any depth -> keyed pseudonyms (HMAC-SHA256 with a secret salt)
  * every other person name (PN) and AE title, at any depth -> emptied
  * listed identity, staff and site attributes -> removed or emptied
  * EVERY DA / DT / TM value, anywhere (nested sequences included)  -> emptied
  * EVERY UID, anywhere, except DICOM registry UIDs (1.2.840.10008.*) -> deterministic
    pseudonymous 2.25 UID; references inside sequences stay consistent
  * free text containing HOST-nnnn names, IPv4 / e-mail addresses, or dates / times -> "REDACTED"
  * private tags, overlays (60xx), curves (50xx), file-meta AE titles, the 128-byte preamble -> removed
  * images flagged BurnedInAnnotation=YES -> rejected (pixel data is never altered)

Study / series / protocol descriptions are RETAINED (like the standard's Retain Description
Option) because they carry the clinical meaning a model may need; they are still pattern- and
token-scrubbed like every other text value.

The real->anon UID pairs are returned so the router can keep them in its
local crosswalk and re-identify the model's answer on the way back.

The salt is a secret. Pseudonyms are only as strong as the salt: with a known
salt, an MRN or a date can be brute-forced back from its pseudonym. So a
missing, placeholder or short salt is refused (see `check_salt`).
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import re

from pydicom import Dataset
from pydicom.dataelem import DataElement

PSEUDONYMISE = ("PatientName", "PatientID")

# Type-2 attributes of the Patient / General Study modules: always present, zero-length.
# (Type 2 means "must be present, may be empty" -- deleting one makes the object non-conformant.)
EMPTY = ("PatientBirthDate", "PatientSex", "AccessionNumber", "ReferringPhysicianName", "StudyID")

REMOVE = (
    "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence", "PatientBirthTime", "PatientAddress",
    "PatientTelephoneNumbers", "PatientAge", "PatientWeight", "PatientSize", "EthnicGroup",
    "PatientComments", "AdditionalPatientHistory", "MedicalRecordLocator", "ImageComments",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "InstitutionCodeSequence",
    "StationName", "OperatorsName", "PerformingPhysicianName", "NameOfPhysiciansReadingStudy",
    "RequestingPhysician", "ScheduledPerformingPhysicianName", "DeviceSerialNumber",
    "RequestAttributesSequence", "ReferencedPatientSequence", "ReferencedStudySequence",
    "PerformedLocation", "ScheduledProcedureStepLocation", "RetrieveAETitle", "StationAETitle",
    "ScheduledStationAETitle", "PerformedStationAETitle",
    # Basic Profile X/Z attributes a clinical study can carry beyond the obvious ones
    "PhysiciansOfRecord", "PhysiciansOfRecordIdentificationSequence", "PatientMotherBirthName",
    "PatientBirthName", "AdmittingDiagnosesDescription", "AdmittingDiagnosesCodeSequence",
    "ReasonForTheRequestedProcedure", "ReasonForStudy", "RequestedProcedureComments",
    "ImagingServiceRequestComments", "InterpretationText", "Occupation", "MilitaryRank", "BranchOfService",
    "CountryOfResidence", "RegionOfResidence", "PatientReligiousPreference", "MedicalAlerts", "Allergies",
    "SmokingStatus", "PregnancyStatus", "PatientInsurancePlanCodeSequence", "PatientState",
    "SourcePatientGroupIdentificationSequence", "GroupOfPatientsIdentificationSequence",
    "PerformingPhysicianIdentificationSequence", "OperatorIdentificationSequence",
    "ReferringPhysicianIdentificationSequence", "PhysiciansReadingStudyIdentificationSequence",
    "DeviceUID", "IssuerOfPatientID", "IssuerOfAccessionNumberSequence",
)
META_REMOVE = ("SourceApplicationEntityTitle", "SendingApplicationEntityTitle",
               "ReceivingApplicationEntityTitle", "PrivateInformationCreatorUID")

UID_KINDS = {"StudyInstanceUID": "study", "SeriesInstanceUID": "series",
             "SOPInstanceUID": "sop", "FrameOfReferenceUID": "frame"}
REGISTRY_ROOT = "1.2.840.10008."
TEXT_VRS = {"AE", "LO", "SH", "LT", "ST", "UT", "UC", "PN", "CS", "UR"}

PATTERNS = {
    "host": re.compile(r"\bHOST-?\d{3,}\b", re.I),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    # Dates and times inside free text. QA headers can carry
    # PatientID = "Daily QA 7/29/2026 1:48 AM" and PatientName = "Constancy 7/3/2023 ...".
    "date_text": re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"
                            r"|\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b", re.I),
    "time_text": re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\b", re.I),
}

PLACEHOLDER_SALTS = {"", "change-me-per-site", "change-me", "changeme", "salt", "test", "set-me"}
MIN_SALT_LEN = 16


class BurnedInAnnotationError(ValueError):
    pass


class WeakSaltError(ValueError):
    pass


def check_salt(salt: str | None) -> str:
    if salt is None or salt.strip().lower() in PLACEHOLDER_SALTS or len(salt) < MIN_SALT_LEN:
        raise WeakSaltError(
            f"de-identification salt is missing, a placeholder, or shorter than {MIN_SALT_LEN} characters. "
            "Set AIROUTER_DEID_SALT to a long random secret, e.g. "
            "python -c \"import secrets; print(secrets.token_hex(32))\"")
    return salt


def anon_uid(orig: str, salt: str) -> str:
    """Deterministic pseudonymous UID under the 2.25 (UUID-derived) root.

    pydicom's generate_uid(prefix=None, ...) always returns a random UUID and
    ignores entropy_srcs, so it cannot give a stable pseudonym. The 128-bit
    value is derived here from a keyed hash instead
    (tests/test_deid_rules.py::test_uids_deterministic_and_reversible).
    """
    digest = hmac.new(salt.encode(), orig.encode(), hashlib.sha256).digest()[:16]
    return f"2.25.{int.from_bytes(digest, 'big')}"


def pseudonym(value: str, salt: str, n: int = 12) -> str:
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:n].upper()


def text_hit(text: str, tokens: set[str] = frozenset()) -> str | None:
    up = text.upper()
    for t in tokens:
        if t and t.upper() in up:
            return "token"
    for name, rx in PATTERNS.items():
        if rx.search(text):
            return name
    return None


def deidentify(ds: Dataset, salt: str, reject_burned_in: bool = True,
               tokens: set[str] = frozenset()) -> tuple[Dataset, list[tuple[str, str, str]]]:
    """Return (de-identified copy, [(anon_uid, orig_uid, kind), ...]).

    `tokens`: extra site identifiers (institution words, host names, operators...)
    whose presence in any text value gets that value redacted. See site_scrub.
    """
    check_salt(salt)
    if reject_burned_in and str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
        raise BurnedInAnnotationError(f"SOP {ds.SOPInstanceUID} has BurnedInAnnotation=YES")

    out = copy.deepcopy(ds)
    # Remove listed attributes at ANY depth, not just the top level (a physician name inside
    # ScheduledProcedureStepSequence is just as identifying).
    _remove_everywhere(out, set(REMOVE))
    for kw in PSEUDONYMISE:
        if kw in out:
            orig = str(out.data_element(kw).value)
            out.data_element(kw).value = ("ANON^" if kw == "PatientName" else "") + pseudonym(orig, salt)
    for kw in EMPTY:
        setattr(out, kw, "")
    out.remove_private_tags()
    for tag in [t for t in out.keys() if 0x5000 <= t.group <= 0x501E or 0x6000 <= t.group <= 0x601E]:
        del out[tag]

    pairs: list[tuple[str, str, str]] = []
    seen: dict[str, str] = {}

    def remap(v: str, kind: str) -> str:
        if v.startswith(REGISTRY_ROOT):
            return v
        if v not in seen:
            seen[v] = anon_uid(v, salt)
            pairs.append((seen[v], v, kind))
        return seen[v]

    top_pseudonymised = {id(out.data_element(k)) for k in PSEUDONYMISE if k in out}

    def visit(_: Dataset, elem: DataElement) -> None:
        if id(elem) in top_pseudonymised or elem.value in (None, ""):
            return
        if elem.VR in ("DA", "DT", "TM"):
            elem.value = ""
        elif elem.keyword in PSEUDONYMISE:
            # A patient name / ID nested somewhere (e.g. another patient in a group sequence)
            vals = [str(v) for v in (elem.value if elem.VM > 1 else [elem.value])]
            new = [("ANON^" if elem.keyword == "PatientName" else "") + pseudonym(v, salt) for v in vals]
            elem.value = new if len(new) > 1 else new[0]
        elif elem.VR == "PN":
            elem.value = ""                     # every other person name, at any depth
        elif elem.VR == "AE":
            elem.value = ""                     # AE titles identify the site's devices
        elif elem.VR == "UI" and elem.keyword.endswith(("SOPClassUID", "TransferSyntaxUID")):
            return                              # a class, not an instance -- incl. vendor-private classes
        elif elem.VR == "UI":
            kind = UID_KINDS.get(elem.keyword, "ref")
            vals = [remap(str(v), kind) for v in (elem.value if elem.VM > 1 else [elem.value])]
            elem.value = vals if len(vals) > 1 else vals[0]
        elif elem.VR in TEXT_VRS:
            vals = list(elem.value) if elem.VM > 1 else [elem.value]
            if any(text_hit(str(v), tokens) for v in vals):
                elem.value = "" if elem.VR in ("CS", "AE") else "REDACTED"

    out.walk(visit)

    if getattr(out, "file_meta", None) is not None:
        for kw in META_REMOVE:
            if kw in out.file_meta:
                del out.file_meta[kw]
        out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.preamble = b"\x00" * 128              # free-form bytes; can carry vendor/site text
    out.PatientIdentityRemoved = "YES"
    out.LongitudinalTemporalInformationModified = "REMOVED"
    out.DeidentificationMethod = "dicom-ai-router deid v2 (PS3.15 Basic Profile subset)"   # LO: max 64
    return out, pairs


def _remove_everywhere(ds: Dataset, keywords: set[str]) -> None:
    for elem in list(ds):
        if elem.keyword in keywords:
            del ds[elem.tag]
        elif elem.VR == "SQ":
            for item in elem.value:
                _remove_everywhere(item, keywords)


PHI_KEYWORDS = PSEUDONYMISE + EMPTY + REMOVE


def find_phi(ds: Dataset, originals: Dataset) -> list[str]:
    """Test helper: list any PHI-bearing tags whose original value survived."""
    leaks = []
    for kw in PHI_KEYWORDS + tuple(UID_KINDS):
        if kw in originals and kw in ds:
            o, n = str(originals.data_element(kw).value), str(ds.data_element(kw).value)
            if o and o == n:
                leaks.append(kw)
    return leaks
