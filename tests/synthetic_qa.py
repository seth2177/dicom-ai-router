"""Synthetic CT QA and patient images for the security and de-identification tests -- no real data.

Planted on fictional sites and people: a procedure name in PatientName, a date and time inside PatientID,
QualityControlImage=YES, institution, city, station HOST-nnnnnn, device serial, source AE title, operator,
private vendor tags echoing the station, an overlay plane and a nested sequence carrying the site.
"""
from __future__ import annotations

import io

import numpy as np
from pydicom import Dataset, FileMetaDataset, dcmwrite
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

SITE = {
    "institution": "LAKESIDE VALLEY RADIOLOGY RM 3", "city": "Fernhollow ND", "station": "HOST-918273",
    "serial": "918273", "ae": "LKV_CT_3", "operator": "QUINTANILLA^MARISOL", "tech": "MARISOL QUINTANILLA",
}
TOKENS_EXPECTED = ["LAKESIDE", "FERNHOLLOW", "HOST-918273", "918273", "LKV_CT_3", "QUINTANILLA", "MARISOL"]

_uid_n = [0]


def _uid() -> str:
    _uid_n[0] += 1
    return f"1.2.826.0.1.3680043.8.498.1{_uid_n[0]:06d}"


def _phantom(kind: str, n: int = 512, spacing: float = 0.684, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - n / 2, yy - n / 2) * spacing
    img = np.full((n, n), -1000.0)
    if kind == "head":                                   # 200 mm, 10 mm acrylic wall, water inside
        img[rr <= 100] = 120
        img[rr <= 90] = 0
        pin = np.hypot(xx - n / 2, yy - (n / 2 - 78 / spacing)) * spacing <= 3   # fill plug, near 12 o'clock
        img[pin] = 900
        img += rng.normal(0, 2.8, img.shape)
    elif kind == "body":                                 # 300 mm solid +110 HU disc with an insert ring
        img[rr <= 150] = 110
        img[(rr >= 72) & (rr <= 88)] = 138
        img += rng.normal(0, 9.7, img.shape)
    elif kind == "resolution":                           # bar pattern: not uniform
        img[rr <= 100] = 0
        img[(rr <= 60) & ((xx // 6) % 2 == 0)] = 600
        img += rng.normal(0, 3, img.shape)
    elif kind == "air":
        img += rng.normal(0, 5, img.shape)
    elif kind == "chest":
        img[((xx - n / 2) / 230) ** 2 + ((yy - n / 2) / 170) ** 2 <= 1] = 40
        for dx in (-95, 95):
            img[((xx - n / 2 - dx) / 70) ** 2 + ((yy - n / 2) / 120) ** 2 <= 1] = -850
        img += rng.normal(0, 15, img.shape)
    return img


def _ds(kind: str, patient_name: str, patient_id: str, series_desc: str, qc: bool, image_type=("ORIGINAL", "PRIMARY", "AXIAL"),
        study_desc: str = "DAILY QA", seed: int = 0, preamble: bool = True) -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.SourceApplicationEntityTitle = SITE["ae"]
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = _uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = _uid(), _uid(), _uid()
    ds.IrradiationEventUID = _uid()
    ds.ImageType = list(image_type)
    ds.InstanceCreationDate = ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = ds.ContentDate = "20260729"
    ds.StudyTime = ds.AcquisitionTime = "015107.479"
    ds.AcquisitionDateTime = "20260729015129.050"
    ds.Modality, ds.Manufacturer, ds.ManufacturerModelName = "CT", "ExampleVendor", "Example CT 64"
    ds.InstitutionName, ds.InstitutionAddress = SITE["institution"], SITE["city"]
    ds.StationName, ds.DeviceSerialNumber = SITE["station"], SITE["serial"]
    ds.OperatorsName = SITE["operator"]
    ds.StudyDescription, ds.SeriesDescription = study_desc, series_desc
    ds.ImageComments = f"{series_desc} acquired on {SITE['station']}"
    ds.PatientName, ds.PatientID, ds.PatientBirthDate, ds.PatientSex = patient_name, patient_id, "", "O"
    ds.ReferringPhysicianName, ds.AccessionNumber = "", ""
    ds.AcquisitionNumber, ds.PatientPosition, ds.PositionReferenceIndicator = 1, "HFS", ""
    ds.StudyID, ds.SeriesNumber, ds.InstanceNumber = "2123", 10600, 1
    if qc:
        ds.QualityControlImage = "YES"
    # nested sequence carrying the site
    req = Dataset()
    req.ScheduledProcedureStepDescription = f"Daily QA - {SITE['institution']}"
    req.RequestedProcedureID = "RP-2123"
    ds.RequestAttributesSequence = Sequence([req])
    # private vendor block echoing the station name
    blk = ds.private_block(0x2001, "EXAMPLE VENDOR 001", create=True)
    blk.add_new(0x01, "LO", SITE["station"])
    blk.add_new(0x02, "LO", "20260729 01:51")
    # overlay plane
    ds.add_new(0x60000010, "US", 16)
    ds.add_new(0x60000011, "US", 16)
    ds.add_new(0x60003000, "OW", b"\x00" * 32)

    n = 512
    img = _phantom(kind, n, 0.684, seed)
    ds.Rows = ds.Columns = n
    ds.PixelSpacing = [0.684, 0.684]
    ds.SliceThickness, ds.KVP = 10, 120
    ds.ImagePositionPatient, ds.ImageOrientationPatient = [-175, -175, 0], [1, 0, 0, 0, 1, 0]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
    ds.PixelData = np.clip(img + 1024, 0, 4095).astype(np.uint16).tobytes()
    if preamble:
        ds.preamble = (f"{SITE['station']} {SITE['institution']}".encode() + b"\x00" * 128)[:128]
    return ds


def _bytes(ds: Dataset) -> bytes:
    buf = io.BytesIO()
    dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()
