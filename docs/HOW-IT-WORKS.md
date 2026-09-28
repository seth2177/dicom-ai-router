# How it works: one study, hop by hop

This follows one chest CT from the moment the tech hits **Send** until the AI result appears in PACS. Each hop covers what the code does, why it's built that way, and what breaks in the field.

```
 CT scanner ──C-STORE──▶ ROUTER ──HTTP (de-identified)──▶ AI MODEL
  CT_SIM01              AIROUTER                          lung-nodule
                          │  ▲                                │
                          │  └────────── findings JSON ◀──────┘
                          │
                          └──C-STORE (SR + key image, re-identified)──▶ PACS
```

---

## Hop 1: The scanner sends images (`tools/modality_sim.py`)

**What happens.** The scanner opens a DICOM *association* to `AIROUTER@host:11112`. It proposes the SOP class (CT Image Storage) and the transfer syntaxes it can speak. Then it sends 24 slices with C-STORE, one per image.

**The synthetic part.** Patients, MRNs and pixels are all fake. Each volume is a body ellipse at +40 HU, two lungs at −850 HU and, sometimes, a spherical nodule. We placed the nodule, so we know the **ground truth** exactly (size, side, slice). That's saved to `data/truth/`, and it's what makes evaluation possible later.

**Field reality.** This is the same sender config you set on every modality: AE title, IP, port. The router
enforces the called AE title (a scanner configured with the wrong one is rejected at association) and can
restrict *calling* AE titles with `allowed_calling_aes`. A wrong port just times out. Either way, no images reach
the router.

---

## Hop 2: The router receives (`airouter/router.py → _on_c_store`)

**What happens.** The handler checks that the Study and SOP Instance UIDs really are UIDs (digits and dots, at most 64
characters). They become file names, so `../../x` is refused with status `0xC000`. It then writes the file under
`data/inbox/<StudyUID>/`, using a temp name plus an atomic rename so a worker never reads half a file, and
immediately returns `0x0000 Success`. That's all it does.

**Why.** *Never make the modality wait.* If the receiver does real work inside the C-STORE, the scanner's send queue backs up behind it. That's the classic "images are slow to arrive" service call. All real work happens later, on worker threads.

**Negotiation.** The router accepts every storage SOP class, but only uncompressed transfer syntaxes (Explicit and Implicit VR Little Endian). If a modality only offers JPEG 2000, that one line in `start()` is what you change.

---

## Hop 3: "Is the study finished?" (`router.py → _watch`)

**What happens.** DICOM has **no end-of-study message**. The router records when each study last received an image. Once it has been quiet for `study_quiet_seconds` (3 s here), the study is marked complete and queued.

**Why it matters.** Every commercial AI router and PACS prefetch engine makes this same guess. If it's too short, a slow scanner's study gets split and the AI sees half a chest. If it's too long, results arrive late. It's a per-site tuning knob, which is why it lives in the config.

---

## Hop 4: Routing (`airouter/rules.py`)

**What happens.** The router reads the header of the first image and walks the rules in `config/router.yaml`. The first rule where every regex matches wins, and names the model. No match means the study is marked `IGNORED` and **nothing leaves the building**. That's why the head CTs in the demo never reach the AI.

**QA and calibration come first.** The first rule holds calibration acquisitions (air calibration, tube
conditioning), which are phantom-like but uncorrected by design. The second sends QA phantoms
(`QualityControlImage = YES`, or vendor naming conventions) to the `ct-qa`
model. A phantom must never reach a clinical model: it produces junk findings, and many AI vendors bill per study.
Rules can use `match` (all conditions) and `match_any` (at least one), and `model: null` means "recognize and hold".

**Field reality.** Headers are messy. `BodyPartExamined` is blank on many scanners, so there's a second rule that falls back to `StudyDescription`. The test suite covers exactly that case. At a real site, most go-live tuning happens here.

---

## Hop 5: De-identification (`airouter/deid.py`)

**What happens.** Before anything goes to the model, each image is copied and scrubbed, following a subset of the
DICOM PS3.15 Basic Confidentiality Profile. It's a documented subset, not a certified implementation.

| Data | Treatment |
|---|---|
| Patient name, ID | keyed pseudonym, HMAC-SHA256 with a secret salt (`ANON^3F9A…`) |
| DOB, sex, accession, referring physician, study ID | emptied (Type 2: kept present, zero-length) |
| Institution, station, operators, device serial, AE titles, image comments, request details | removed |
| **Every** date and time (DA / DT / TM), anywhere, including nested sequences | emptied |
| **Every** UID, anywhere, except DICOM registry UIDs | deterministic pseudonymous `2.25.…` UID. References inside sequences stay consistent |
| Free text containing a HOST-nnnn name, an IP / e-mail address, or a date or time | `REDACTED` |
| Private tags, overlays (60xx), curves (50xx), file-meta AE titles, the 128-byte preamble | removed |
| `BurnedInAnnotation = YES` | that image is dropped, never sent |

**The salt is a secret.** A pseudonym is only as strong as its salt. With a known salt, an MRN, or a date written into
PatientID, can be brute-forced back in seconds, and testing showed exactly that against the old
placeholder salt. So the router reads the salt from `AIROUTER_DEID_SALT` and **refuses to start** if it's missing,
a known placeholder, or shorter than 16 characters.

**The crosswalk.** Every real ↔ fake UID pair goes into SQLite (`data/router.sqlite`) and never leaves the router. That's how the answer finds its way back to the right patient.

**Why deterministic.** The same study with the same salt always gets the same pseudonym. So a re-sent study doesn't
show up as a new patient, and results can be matched after the fact. (pydicom's `generate_uid(prefix=None)` always
returns a random UUID and ignores its entropy argument, so it can't give a stable pseudonym. That's why
`anon_uid()` derives the UID from a keyed hash itself, and `test_uids_deterministic_and_reversible` pins it down.)

**Why reject burned-in annotation.** The pixels aren't touched. If the scanner stamped the patient's name into the
image, scrubbing header tags doesn't help. So that image is dropped, and the drop is audited.

**Which images the model gets.** Real studies carry localizers, dose screens and other captures. The router drops
them *per image* (non-CT SOP classes, `LOCALIZER` image type, burned-in annotation) and records each drop in the
audit, so one dose screen never fails a whole study. A rule can also ask for only the main series
(`series: largest`), which the nodule model uses because it wants one axial volume.

---

## Hop 6: AI inference (`airouter/ai_client.py`, `mock_ai/`)

**One contract for every model.** Each model answers with the same fields: `result`, `summary_lines`,
`key_sop_instance_uid`, `overlays` and `display_window`. So adding a model is config, not router code. The
`ct-qa` model is a real measurement: a center ROI and four edge ROIs, CT number, noise and uniformity,
graded against ACR water criteria or against the scanner's own limits.

**What happens.** The de-identified images are POSTed to the model as a multipart/form-data upload of DICOM Part-10
files. That's a simple stand-in for a vendor API. A standards-based deployment would send a DICOMweb STOW-RS
request (multipart/related), which is a change confined to `ai_client.py`. The model returns JSON with a finding,
side, size in mm, key slice and confidence.

**Retries.** A 5xx or network error gets retried with exponential backoff (0.5 s, 1 s, 2 s…). A 4xx is not retried, because the request itself is wrong. Run `python run_demo.py --fail-rate 0.4` to watch the router ride through a flaky model.

**The mock model** (`mock_ai/detector.py`) is simple geometry, not AI. It finds dense blobs fully enclosed by lung and reports the largest. It **misses nodules of about 3 mm and faint ground-glass ones (≈ −350 HU)**, and finds solid nodules of 5 mm and up. That's on purpose: the eval stage needs real errors to measure.

**Notice** that the model sees `2.25.…` UIDs and `ANON^…` names, not the real ones. Every other person name and AE
title, at any depth, is emptied, and `tests/test_security.py` checks this with nested cases and the independent
audit. Study and series descriptions are kept, still pattern-scrubbed, because they carry clinical meaning. Open any `data/results/*.json` and compare `ai.study_instance_uid` with `study_uid`.

---

## Hop 7: Re-identify and build results (`airouter/pipeline.py`, `airouter/results.py`)

**What happens.** The model's key-slice UID is looked up in the crosswalk to find the real image. Two DICOM objects are then built **inside the original study**: same StudyInstanceUID and real demographics, each in a new series.

1. **Basic Text SR** (series 9901). Structured findings, `VerificationFlag = UNVERIFIED`, a reference to the key
   image, and an evidence sequence listing every source image grouped under its own series. Each text item has its
   own code in a private scheme (`99AIROUTER`), declared in `CodingSchemeIdentificationSequence`. Reporting
   systems can parse this.
2. **Secondary Capture image** (series 9902). The slice in the model's display window (lung window for nodules,
   a narrow window for QA), overlays drawn, "AI RESULT - NOT FOR DIAGNOSIS" burned in, and
   `BurnedInAnnotation = YES` set honestly. With no finding, it's labelled a *result* image, not a key image.

Both objects validate against the DICOM standard (2026d IOD definitions, checked with `dicom-validator`), including
the Type 2 attributes, which are written empty when the source doesn't have them.

**Why both.** The SR is for machines, the key image is for people. Plenty of PACS viewers still don't render SR nicely, but every viewer shows an image.

---

## Hop 8: Back to PACS (`airouter/sender.py`)

**What happens.** A new association to PACS. It proposes **only** the two SOP classes being sent (SR and Secondary
Capture), then C-STOREs, retrying with backoff on failure. Warning statuses (`0xB000`, `0xB006`, `0xB007`) mean
"stored, with coercion", so they count as delivered.

**Field reality.** Proposing every storage class at once is a common reason picky PACS reject associations. So is PACS not having the router configured as a known AE. That's the "AI results never show up" ticket.

---

## Restarts and late images

A study still waiting on its quiet timer, or mid-processing, when the router stops stays in the database as
unfinished, and the next start picks it up again. If more images arrive while a study is being processed, it's run
again after the current run finishes, never twice at once. Result UIDs are deterministic: the series is derived from
the study and the model, and each instance also from a digest of what the result says. A re-run with the same
findings re-sends the identical objects, with no duplicates. A re-run with different findings (late images) adds a
new instance in the same series, as DICOM requires for changed content, so a PACS can't silently keep the stale one. A study that
FAILED can be re-run with `python -m airouter rerun --failed` once the fault is fixed.

## The audit trail (`data/audit.jsonl`)

One JSON line per step, with timing in ms: receive_start, study_complete, load, route, deidentify, ai_inference, build_results, c_store_pacs. It holds no names or MRNs. This is what you hand the hospital's security reviewer, and it's how you prove turnaround time. `python -m airouter status` shows the state of every study.

---

## Where this goes next: llm-eval-radiology

Every study now leaves two files behind: `results/<uid>.json` (what the AI said) and `truth/<uid>.json` (what was really there). Stage 2 adds an LLM that writes a draft impression from the AI findings, then scores both layers against ground truth:

- **Detector:** sensitivity and specificity by nodule size, laterality errors, size error in mm
- **LLM report:** hallucinated findings, omitted findings, flipped laterality, altered measurements, missing follow-up language

That second list is the failure-mode analysis healthcare-AI buyers ask vendors about, and the skill the market currently pays a premium for.
