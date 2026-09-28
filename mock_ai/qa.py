"""CT phantom QA: CT number accuracy, noise and uniformity.

Unlike the nodule detector, this is a real measurement: the same checks a
physicist or field engineer does by hand with ROIs on a daily-QA image.

  * find the phantom, then its UNIFORM CORE, whatever the material
    (a head QA phantom is water inside an acrylic wall; a body QA phantom
    can be a solid ~+110 HU disc with an insert ring)
  * centre ROI + four edge ROIs (12, 3, 6, 9 o'clock), 1 cm inside the core edge;
    pixels that are not phantom material (pins, inserts) are excluded from every ROI
  * CT number = centre ROI mean
  * noise     = SD over a large central ROI (0.4 x core radius)
  * uniformity = worst |edge mean - centre mean|

Verdict, in order of precedence:
  1. the scanner's own limits, when a reference is supplied   -> PASS / FAIL
  2. ACR CT accreditation criteria, for a WATER phantom      -> PASS / FAIL
       water CT number 0 +/- 7 HU (+/- 5 preferred); edge-to-centre < 5 HU
       (5-7 HU minor deficiency, > 7 HU major)
  3. otherwise (non-water, no reference)                      -> MEASURED, numbers only
Anything that is not a uniform circular phantom is NOT_A_PHANTOM and is not
graded, which is the safety net for a mislabelled patient study on the QA route.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

WATER_TOL_HU = 7.0
WATER_PREFERRED_HU = 5.0
UNIFORMITY_TOL_HU = 5.0
UNIFORMITY_MAJOR_HU = 7.0
NOISE_ROI_FRACTION = 0.4


def _phantom_mask(sl: np.ndarray) -> np.ndarray | None:
    dense = ndimage.binary_opening(sl > -300, iterations=2)
    labels, n = ndimage.label(dense)
    if n == 0:
        return None
    sizes = ndimage.sum(dense, labels, range(1, n + 1))
    return ndimage.binary_fill_holes(labels == (int(np.argmax(sizes)) + 1))


def _roi(sl, cx, cy, r, valid=None):
    """ROI mean/SD over phantom material only. `valid` excludes inserts, pins and fill
    plugs (found on real data: a dense pin sitting in the 12 o'clock ROI read +10.7 HU
    "non-uniformity" on a phantom the scanner correctly passed)."""
    yy, xx = np.ogrid[: sl.shape[0], : sl.shape[1]]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    excluded = 0
    if valid is not None:
        excluded = int((m & ~valid).sum())
        m = m & valid
    v = sl[m]
    return float(v.mean()), float(v.std()), excluded


def _uniform_core(sl: np.ndarray, outer: np.ndarray) -> np.ndarray | None:
    """The uniform region ROIs belong in, whatever the phantom material.

    A head QA phantom is often water inside an acrylic wall (so the outer
    outline overstates the water area), and a body phantom may be a solid
    disc, not water at all. So: take the material
    at the centre, keep everything within 60 HU of it, and use that region.
    Smooth first so a noisy scanner does not punch holes in the mask.
    """
    smooth = ndimage.gaussian_filter(sl, 2)
    cy, cx = ndimage.center_of_mass(outer)
    yy, xx = np.ogrid[: sl.shape[0], : sl.shape[1]]
    centre_val = float(np.median(smooth[(xx - cx) ** 2 + (yy - cy) ** 2 <= 100]))
    water = ndimage.binary_opening((np.abs(smooth - centre_val) < 60) & outer, iterations=2)
    labels, n = ndimage.label(water)
    if n == 0:
        return None
    lab = labels[int(round(cy)), int(round(cx))]
    if lab == 0:                                       # centre is not water -> take largest water region
        lab = int(np.argmax(ndimage.sum(water, labels, range(1, n + 1)))) + 1
    return ndimage.binary_fill_holes(labels == lab)


def analyse_slice(sl: np.ndarray, spacing_mm: float) -> dict:
    outer = _phantom_mask(sl)
    if outer is None:
        return {"is_phantom": False, "reason": "no object in field of view"}
    mask = _uniform_core(sl, outer)
    if mask is None or mask.sum() < 50:
        return {"is_phantom": False, "reason": "no uniform region"}
    area = int(mask.sum())
    cy, cx = ndimage.center_of_mass(mask)
    r_px = float(np.sqrt(area / np.pi))
    # Circularity: a phantom fills its fitted circle; a body/head does not.
    yy, xx = np.ogrid[: sl.shape[0], : sl.shape[1]]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_px ** 2
    fill = float((mask & circle).sum() / circle.sum())
    inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.8 * r_px) ** 2
    # Robust spread (MAD): inserts such as the rod ring in a body IQ phantom are
    # a small fraction of the area and must not disqualify a uniform phantom.
    vals = sl[inner]
    interior_mean = float(np.median(vals))
    interior_sd = float(1.4826 * np.median(np.abs(vals - interior_mean)))
    diameter_mm = 2 * r_px * spacing_mm
    outer_mm = 2 * float(np.sqrt(outer.sum() / np.pi)) * spacing_mm
    is_phantom = fill > 0.95 and interior_sd < 60 and 120 <= diameter_mm <= 450
    if not is_phantom:
        return {"is_phantom": False, "reason": f"not a uniform circular phantom (fill {fill:.2f}, interior SD {interior_sd:.0f} HU, {diameter_mm:.0f} mm)"}
    material = "water" if abs(interior_mean) < 30 else f"non-water ({interior_mean:+.0f} HU)"

    roi_r = max(3.0, 0.05 * 2 * r_px)                       # ~10 mm radius on a 200 mm phantom
    edge_d = r_px - 10.0 / spacing_mm - roi_r                # 1 cm inside the edge
    rois = [("C", cx, cy)] + [(lab, cx + dx * edge_d, cy + dy * edge_d)
                              for lab, dx, dy in (("12", 0, -1), ("3", 1, 0), ("6", 0, 1), ("9", -1, 0))]
    # Exclude anything that is not phantom material: > 5 sigma (min 40 HU) from the core value,
    # dilated 2 px to take the partial-volume halo around a pin or insert with it.
    outlier = np.abs(sl - interior_mean) > max(40.0, 5 * interior_sd)
    valid = mask & ~ndimage.binary_dilation(ndimage.binary_opening(outlier), iterations=2)
    stats = {lab: _roi(sl, x, y, roi_r, valid) for lab, x, y in rois}
    centre_mean = stats["C"][0]
    # Noise over a large central ROI (0.4 R). Calibrated once against the scanner's own QA results on
    # real data: a 0.1 R ROI read body-phantom noise 0.52 HU high; 0.4 R agrees within 0.03 HU on
    # both head and body phantoms. (0.5 R starts to hit the body phantom's insert ring.)
    _, centre_sd, _ = _roi(sl, cx, cy, NOISE_ROI_FRACTION * r_px, valid)
    edge_dev = {lab: stats[lab][0] - centre_mean for lab in ("12", "3", "6", "9")}
    return {
        "is_phantom": True, "diameter_mm": round(diameter_mm, 1), "outer_diameter_mm": round(outer_mm, 1),
        "material": material,
        "ct_hu": round(centre_mean, 2), "noise_hu": round(centre_sd, 2),
        "uniformity_hu": round(max(abs(v) for v in edge_dev.values()), 2),
        "edge_dev_hu": {k: round(v, 2) for k, v in edge_dev.items()},
        "rois": [{"label": lab, "cx": round(x, 1), "cy": round(y, 1), "r": round(roi_r, 1),
                  "mean": round(stats[lab][0], 1), "excluded_px": stats[lab][2]} for lab, x, y in rois],
    }


def grade(water: float, uniformity: float) -> tuple[str, list[str]]:
    """ACR criteria for a WATER phantom."""
    notes: list[str] = []
    failed = False
    if abs(water) > WATER_TOL_HU:
        failed = True
        notes.append(f"water CT number {water:+.1f} HU outside 0 +/- {WATER_TOL_HU:.0f}")
    elif abs(water) > WATER_PREFERRED_HU:
        notes.append(f"water CT number {water:+.1f} HU outside preferred +/- {WATER_PREFERRED_HU:.0f}")
    if uniformity > UNIFORMITY_MAJOR_HU:
        failed = True
        notes.append(f"uniformity {uniformity:.1f} HU: major deficiency (> {UNIFORMITY_MAJOR_HU:.0f})")
    elif uniformity >= UNIFORMITY_TOL_HU:
        failed = True
        notes.append(f"uniformity {uniformity:.1f} HU: minor deficiency (5-7)")
    return ("FAIL" if failed else "PASS"), notes


def grade_against(reference: dict, ct: float, uniformity: float, noise: float) -> tuple[str, list[str]]:
    """Grade against a scanner's own limits, e.g. from its daily QA results:
    reference = {"ct": [lo, hi], "uniformity": [lo, hi], "noise": [lo, hi]}"""
    notes = []
    for name, value in (("ct", ct), ("uniformity", uniformity), ("noise", noise)):
        if name in reference:
            lo, hi = reference[name]
            if not lo <= value <= hi:
                notes.append(f"{name} {value:.1f} HU outside scanner limits [{lo:g}, {hi:g}]")
    return ("FAIL" if notes else "PASS"), notes


def analyse_series(volume_hu: np.ndarray, spacing_mm: float, reference: dict | None = None) -> dict:
    """Grade the middle slices (the ones clear of the phantom's end caps).

    Verdict source, in order: the scanner's own limits if a reference is given;
    ACR water criteria if the phantom is water; otherwise MEASURED (numbers
    reported, no verdict -- there is no universal limit for a non-water phantom).
    """
    n = len(volume_hu)
    idx = list(range(max(0, n // 2 - 1), min(n, n // 2 + 2)))
    per = [(i, analyse_slice(volume_hu[i], spacing_mm)) for i in idx]
    good = [(i, p) for i, p in per if p["is_phantom"]]
    if not good:
        return {"result": "NOT_A_PHANTOM", "reason": per[0][1]["reason"], "slice_index": idx[0]}
    ct = float(np.mean([p["ct_hu"] for _, p in good]))
    noise = float(np.mean([p["noise_hu"] for _, p in good]))
    unif = float(max(p["uniformity_hu"] for _, p in good))
    key_i, key = good[len(good) // 2]
    if reference:
        status, notes = grade_against(reference, ct, unif, noise)
        basis = "scanner reference limits"
    elif key["material"] == "water":
        status, notes = grade(ct, unif)
        basis = "ACR water criteria"
    else:
        status, notes, basis = "MEASURED", [], "none (non-water phantom, no reference limits)"
    return {"result": status, "ct_hu": round(ct, 2), "noise_hu": round(noise, 2),
            "uniformity_hu": round(unif, 2), "diameter_mm": key["diameter_mm"], "material": key["material"],
            "grading_basis": basis, "notes": notes,
            "slice_index": key_i, "rois": key["rois"], "slices_graded": [i for i, _ in good]}
