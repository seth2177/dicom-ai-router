"""Independent PHI / site-identifier audit of a folder of DICOM files.

Deliberately written separately from the scrubber: it re-reads the output
from disk and checks it with its own rules, so the scrubber is not grading
its own homework. It walks EVERY element, including nested sequences and
file meta, plus every file and folder name and every published text file.

Independence matters for the token list too. If the audit only used the
scrubber's harvested tokens, a harvesting bug would hide from both. So when
given the originals, the audit builds its OWN list (`independent_tokens`)
with different code: every value of any attribute whose keyword names an
institution, station, operator, physician, location, AE title or serial,
every person-name component, and file-meta AE titles.

Checks
  token      any harvested site token (institution, station, AE, host, serial, operator...)
  pattern    HOST-nnnn names, IPv4 / e-mail addresses, dates and times written in free text
  uid        any UID value that appeared in the ORIGINAL data
  date       any non-empty DA / DT / TM value
  private    any private (odd-group) tag
  overlay    any overlay (60xx) or curve (50xx) group
  identity   PatientName / PatientID not pseudonymised, PatientIdentityRemoved != YES
  burned_in  BurnedInAnnotation = YES (pixel text cannot be verified by header audit)
  filename   token or pattern in a file or folder name
  raw_bytes  token or pattern anywhere in the raw file bytes (preamble, anything the parser skips)
  text_file  token or pattern in any non-DICOM file in the folder (manifests, reports)

The report never prints a token -- not even a hash (a hash of a hospital
name can be matched against a list of hospital names). A hit names the
token only by its index in the local, never-published token list, so the
Markdown report itself is safe to publish.

  python -m airouter.phi_audit <folder> --tokens <tokens.local.json> [--orig-uids <uids.local.json>]
                                        [--originals <folder of the source DICOM files>]
  --originals turns on the audit's own, independent token harvest (recommended).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from pydicom import dcmread

RX = {"host": re.compile(r"\bHOST-?\d{3,}\b", re.I),
      "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
      "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
      "date_text": re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b", re.I),
      "time_text": re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\b", re.I)}
TEXT = {"AE", "LO", "SH", "LT", "ST", "UT", "UC", "PN", "CS", "UR"}
# Explicit list (a substring match on "Location" would also catch SliceLocation, i.e. slice positions).
SITE_KEYWORDS = {
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "StationName", "DeviceSerialNumber",
    "OperatorsName", "PerformingPhysicianName", "ReferringPhysicianName", "NameOfPhysiciansReadingStudy",
    "RequestingPhysician", "ScheduledPerformingPhysicianName", "PerformedLocation", "ScheduledProcedureStepLocation",
    "RetrieveAETitle", "StationAETitle", "ScheduledStationAETitle", "PerformedStationAETitle", "AccessionNumber",
}
# Dictionary words are not identifiers even when a site name contains them ("... Scan Center").
NOT_IDENTIFYING = {
    "PHILIPS", "SIEMENS", "CANON", "TOSHIBA", "RADIOLOGY", "IMAGING", "HOSPITAL", "MEDICAL", "CENTER", "CENTRE",
    "CLINIC", "HEALTH", "HEALTHCARE", "REGIONAL", "GENERAL", "COUNTY", "MEMORIAL", "UNIVERSITY", "DEPARTMENT",
    "SCAN", "SCANS", "SCANNER", "ROOM", "UNKNOWN", "NONE", "SYSTEM", "SERVICE", "SERVICES", "DIAGNOSTIC",
    "DIAGNOSTICS", "HOST", "ADMIN", "OPERATOR", "CHECK", "BODY", "HEAD",
}


def independent_tokens(originals) -> set[str]:
    """The audit's own site/person vocabulary, built from the ORIGINAL datasets."""
    found: set[str] = set()

    def add(value: str) -> None:
        value = str(value).strip()
        if len(value) >= 4 and not (value.isdigit() and len(value) < 5):
            found.add(value.upper())
        for part in re.split(r"[\s^,._/\-]+", value):
            if part.isdigit():
                if len(part) >= 5:                         # serial numbers; shorter numbers are not identifying alone
                    found.add(part)
            elif len(part) >= 4 and part.upper() not in NOT_IDENTIFYING:
                found.add(part.upper())

    for ds in originals:
        def visit(_, e):
            if e.VR == "SQ" or e.value in (None, ""):
                return
            vals = e.value if e.VM > 1 else [e.value]
            if e.keyword in SITE_KEYWORDS:
                for v in vals:
                    add(v)
            elif e.VR == "PN":
                # A person's name ("FAMILY^GIVEN"). Phantom QA headers put a procedure name
                # ("Daily QA") in PatientName with no caret -- that is not a person.
                for v in vals:
                    if "^" in str(v):
                        add(v)
        ds.walk(visit)
        meta = getattr(ds, "file_meta", None)
        if meta is not None:
            for kw in ("SourceApplicationEntityTitle", "SendingApplicationEntityTitle"):
                if kw in meta:
                    add(meta.data_element(kw).value)
    return found


def audit(folder: Path, tokens: set[str], orig_uids: set[str] = frozenset(), originals=None) -> dict:
    """tokens: the scrubber's list. originals: the source datasets, from which the audit
    builds its own independent list as well."""
    own = independent_tokens(originals) if originals else set()
    tokens_up = sorted({t.upper() for t in set(tokens) | own if t})
    hits, files, elements = [], 0, 0

    # A numeric token (a serial) must match as a whole number, not inside a longer digit run
    # such as a 39-digit pseudonymous UID.
    matchers = [(i, re.compile(rf"(?<!\d){re.escape(t)}(?!\d)") if t.isdigit() else t) for i, t in enumerate(tokens_up)]

    def token_hits(up: str) -> list[str]:
        return [f"token:#{i}" for i, m in matchers if (m.search(up) if isinstance(m, re.Pattern) else m in up)]

    def text_hits(s: str) -> list[str]:
        return token_hits(s.upper()) + [f"pattern:{k}" for k, rx in RX.items() if rx.search(s)]

    for path in sorted(folder.rglob("*")):
        rel = str(path.relative_to(folder))
        for h in text_hits(rel):
            hits.append({"file": rel, "check": "filename", "detail": h})
        if not path.is_file():
            continue
        if path.suffix.lower() != ".dcm":
            if path.suffix.lower() in (".json", ".md", ".txt", ".csv") and not path.name.endswith(".local.json"):
                for h in text_hits(path.read_text(encoding="utf-8", errors="replace")):
                    hits.append({"file": rel, "check": "text_file", "detail": h})
            continue
        ds = dcmread(path)
        files += 1
        raw = path.read_bytes()
        pix = len(ds.PixelData) if "PixelData" in ds else 0
        # Scan everything except the pixel payload (random pixel bytes spell short words by chance).
        blob = (raw[:-pix] if pix and raw.endswith(ds.PixelData) else raw).decode("latin-1").upper()
        for h in token_hits(blob):
            hits.append({"file": rel, "check": "raw_bytes", "detail": h})
        for k in ("host", "email"):
            if RX[k].search(blob):
                hits.append({"file": rel, "check": "raw_bytes", "detail": f"pattern:{k}"})

        def add(check, elem, detail="", rel=rel):
            hits.append({"file": rel, "check": check, "tag": str(elem.tag), "keyword": elem.keyword, "detail": detail})

        def visit(_, elem):
            nonlocal elements
            elements += 1
            if elem.tag.is_private:
                add("private", elem)
            if 0x5000 <= elem.tag.group <= 0x501E or 0x6000 <= elem.tag.group <= 0x601E:
                add("overlay", elem)
            if elem.VR == "SQ" or elem.value in (None, ""):
                return
            vals = [str(v) for v in (elem.value if elem.VM > 1 else [elem.value])]
            if elem.VR in ("DA", "DT", "TM") and any(vals):
                add("date", elem)
            elif elem.VR == "UI":
                if any(v in orig_uids for v in vals):
                    add("uid", elem, "original UID survived")
            elif elem.VR in TEXT:
                for v in vals:
                    for h in text_hits(v):
                        add(h.split(":")[0], elem, h)

        ds.walk(visit)
        if ds.file_meta is not None:
            ds.file_meta.walk(visit)
        if not str(ds.get("PatientName", "")).startswith("ANON^"):
            hits.append({"file": rel, "check": "identity", "detail": "PatientName not pseudonymised"})
        if str(ds.get("PatientIdentityRemoved", "")) != "YES":
            hits.append({"file": rel, "check": "identity", "detail": "PatientIdentityRemoved != YES"})
        if str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
            hits.append({"file": rel, "check": "burned_in", "detail": "cannot verify pixel text"})

    checks = ["token", "pattern", "uid", "date", "private", "overlay", "identity", "burned_in", "filename",
              "raw_bytes", "text_file"]
    counts = {c: sum(1 for h in hits if h["check"] == c) for c in checks}
    return {"audited_at": dt.datetime.now().isoformat(timespec="seconds"), "files": files,
            "elements_checked": elements, "tokens_checked": len(tokens_up), "independent_tokens": len(own),
            "original_uids_checked": len(orig_uids), "counts": counts,
            "passed": not hits, "hits": hits[:500]}


def to_markdown(r: dict, title: str = "PHI / site-identifier audit") -> str:
    status = "PASS: no identifiers found" if r["passed"] else f"FAIL: {sum(r['counts'].values())} hits"
    rows = "\n".join(f"| {c} | {n} |" for c, n in r["counts"].items())
    md = f"""# {title}

**Result: {status}**

| | |
|---|---|
| Audited | {r['audited_at']} |
| DICOM files | {r['files']} |
| Data elements checked (incl. nested sequences and file meta) | {r['elements_checked']:,} |
| Site tokens checked | {r['tokens_checked']} ({r.get('independent_tokens', 0)} harvested independently by the audit; never printed) |
| Original UIDs checked | {r['original_uids_checked']:,} |

| Check | Hits |
|---|---|
{rows}

Checks: site tokens harvested from the original headers and package names (institution, station, AE titles, host,
serials, operators); HOST-nnnn / IPv4 / e-mail patterns; any original UID; any non-empty date or time;
private tags; overlays and curves; patient identity pseudonymised; burned-in annotation; file and folder names;
a raw byte scan of every file (catches the preamble and anything the DICOM parser skips); and every non-DICOM
text file in the folder (manifests, reports).
"""
    if r["hits"]:
        md += "\n## Hits\n\n| file | check | tag | detail |\n|---|---|---|---|\n"
        md += "\n".join(f"| {h['file']} | {h['check']} | {h.get('keyword') or h.get('tag', '')} | {h.get('detail', '')} |"
                        for h in r["hits"][:100])
    return md


def _read_folder(folder: Path) -> list:
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            try:
                ds = dcmread(p, force=True, stop_before_pixels=True)
            except Exception:  # noqa: BLE001 -- not DICOM
                continue
            if "SOPInstanceUID" in ds:
                out.append(ds)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--tokens", required=True, help="JSON list of site tokens (local secret file)")
    ap.add_argument("--orig-uids", help="JSON list of original UIDs (local secret file)")
    ap.add_argument("--originals", help="folder of the ORIGINAL DICOM files: enables the independent token harvest")
    ap.add_argument("--report", help="write Markdown report here")
    a = ap.parse_args(argv)
    tokens = set(json.loads(Path(a.tokens).read_text()))
    uids = set(json.loads(Path(a.orig_uids).read_text())) if a.orig_uids else set()
    originals = _read_folder(Path(a.originals)) if a.originals else None
    if a.originals and not originals:
        # Fail closed: a typo in the path must not silently switch the independent check off.
        raise SystemExit(f"phi_audit: --originals {a.originals!r} contains no readable DICOM files")
    if originals is None:
        print("note: no --originals given; checking the supplied token list only (not independent)", file=sys.stderr)
    r = audit(Path(a.folder), tokens, uids, originals=originals)
    md = to_markdown(r)
    if a.report:
        Path(a.report).write_text(md)
    print(md)
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
