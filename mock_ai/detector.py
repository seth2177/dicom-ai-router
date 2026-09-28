"""A deliberately simple, fully explainable 'lung nodule model'.

It is not a real AI. It stands in for one so the plumbing around it can be
built and tested. It behaves like a real model in the ways that matter to
integration: it takes a CT volume, returns findings with a size, side,
key slice and confidence, and it misses small or faint nodules -- so the
evaluation stage has real errors to measure.

Method, per slice:
  lung mask       = pixels below -500 HU (air-filled lung)
  filled lungs    = lung mask with enclosed holes filled in
  candidates      = inside filled lungs, not lung, denser than -300 HU
  keep candidates between MIN and MAX size; the largest cross-section wins.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

MIN_DIAMETER_MM = 4.0
MAX_DIAMETER_MM = 30.0


def detect(volume_hu: np.ndarray, pixel_spacing_mm: float) -> list[dict]:
    """volume_hu: (slices, rows, cols). Returns 0..1 findings (largest nodule)."""
    best = None
    for z, sl in enumerate(volume_hu):
        lung = sl < -500
        # Only fill holes inside each lung, not the background air around the body.
        body = ndimage.binary_fill_holes(sl > -500)
        lung_in_body = lung & body
        filled = ndimage.binary_fill_holes(lung_in_body)
        cand = filled & ~lung_in_body & (sl > -300)
        cand = ndimage.binary_opening(cand, iterations=1)
        labels, n = ndimage.label(cand)
        for i in range(1, n + 1):
            region = labels == i
            area_px = int(region.sum())
            diam = 2 * np.sqrt(area_px / np.pi) * pixel_spacing_mm
            if not (MIN_DIAMETER_MM <= diam <= MAX_DIAMETER_MM):
                continue
            cy, cx = ndimage.center_of_mass(region)
            contrast = float(sl[region].mean() - sl[lung_in_body].mean())
            if best is None or diam > best["diameter_mm"]:
                best = {"z": z, "diameter_mm": float(diam), "center_px": [float(cx), float(cy)], "contrast": contrast}
    if best is None:
        return []
    cols = volume_hu.shape[2]
    # Radiological convention: patient's RIGHT is on the image LEFT.
    laterality = "right" if best["center_px"][0] < cols / 2 else "left"
    confidence = float(min(0.99, 0.45 + best["diameter_mm"] / 25 + best["contrast"] / 4000))
    return [{"type": "pulmonary_nodule", "present": True, "laterality": laterality,
             "diameter_mm": round(best["diameter_mm"], 1), "slice_index": best["z"],
             "center_px": [round(v, 1) for v in best["center_px"]], "confidence": round(confidence, 2)}]
