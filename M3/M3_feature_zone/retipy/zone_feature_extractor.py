import cv2
import numpy as np
from skimage.morphology import skeletonize

from .retipy.retina import Retina
from .function_.fractal_dimension import fractal_dimension
from .retipy.tortuosity_measures import (
    distance_measure_tortuosity,
    squared_curvature_tortuosity,
    tortuosity_density,
)


# ----------------------------------------------------------------------
# Shared helper: compute metrics from a binary vessel mask
# ----------------------------------------------------------------------
def compute_vessel_metrics(binary_mask: np.ndarray):
    """
    Compute fractal dimension, density, mean width (area/centerline),
    and three tortuosity measures from a binary vessel mask.

    Returns a dict with keys:
        "fd", "density", "width", "dist_tort", "sq_tort", "dens_tort"
    """
    # Ensure uint8 binary mask
    if binary_mask is None:
        binary_mask = np.zeros((1, 1), dtype=np.uint8)
    mask = (binary_mask > 0).astype(np.uint8)

    fd = np.nan
    density = np.nan
    width = np.nan
    dist_tort = np.nan
    sq_tort = np.nan
    dens_tort = np.nan

    total = float(mask.size)
    on = float(mask.sum())

    if total == 0:
        # completely degenerate
        return {
            "fd": fd,
            "density": density,
            "width": width,
            "dist_tort": dist_tort,
            "sq_tort": sq_tort,
            "dens_tort": dens_tort,
        }

    if on == 0:
        # empty mask: density known (0), others undefined
        density = 0.0
        return {
            "fd": fd,
            "density": density,
            "width": width,
            "dist_tort": dist_tort,
            "sq_tort": sq_tort,
            "dens_tort": dens_tort,
        }

    # --- density ---
    density = on / total

    # --- fractal dimension ---
    try:
        fd = float(fractal_dimension(mask))
    except Exception:
        fd = np.nan

    # --- skeleton & width ---
    skel = skeletonize(mask > 0)
    length = float(skel.sum())
    if length > 0:
        # crude width estimate = area / centerline length
        width = on / length
    else:
        width = np.nan

    # --- tortuosity metrics ---
    coords = np.column_stack(np.where(skel > 0))
    if coords.shape[0] < 2:
        dist_tort = np.nan
        sq_tort = np.nan
        dens_tort = np.nan
    else:
        x = coords[:, 0]
        y = coords[:, 1]
        try:
            dist_tort = float(distance_measure_tortuosity(x, y))
        except Exception:
            dist_tort = np.nan
        try:
            sq_tort = float(squared_curvature_tortuosity(x, y))
        except Exception:
            sq_tort = np.nan
        try:
            dens_tort = float(tortuosity_density(x, y))
        except Exception:
            dens_tort = np.nan

    return {
        "fd": fd,
        "density": density,
        "width": width,
        "dist_tort": dist_tort,
        "sq_tort": sq_tort,
        "dens_tort": dens_tort,
    }


class ZoneFeatureExtractor:
    """
    Zone-based feature extractor for AutoMorph M3.

    - Uses Retina.extract_vessels() for vessel mask.
    - Tries artery/vein classifier from M2_Artery_vein (vessel_classification).
    - If classifier is not available, falls back to a simple
      intensity-based A/V split on the green channel.
    - Zones are concentric annuli around an estimated optic disc.
    """

    def __init__(self, image_bgr: np.ndarray):
        self.image = image_bgr
        self.h, self.w = image_bgr.shape[:2]

        # Retina object (we only use it for vessel segmentation)
        self.retina = Retina(image_bgr, image_path="", store_path="")

        # --- global vessel mask ---
        try:
            self.vessel_mask = self.retina.extract_vessels()
        except Exception:
            self.vessel_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # --- artery / vein masks ---
        self.artery_mask = np.zeros_like(self.vessel_mask, dtype=np.uint8)
        self.vein_mask = np.zeros_like(self.vessel_mask, dtype=np.uint8)

        self._init_av_masks()

        # --- zone masks (zone_b, zone_c) ---
        self.zone_masks = self._create_zone_masks()

    # ------------------------------------------------------------------
    # A/V segmentation
    # ------------------------------------------------------------------
    def _init_av_masks(self):
        """
        Try to get artery/vein masks from the ALL-AV classifier.
        If that fails, fall back to a simple heuristic using brightness.
        """
        used_classifier = False

        # 1) Try classifier from M2_Artery_vein if installed & configured
        try:
            from .retipy import vessel_classification

            av_mask = vessel_classification.classification(self.image, self.vessel_mask)
            if av_mask is not None:
                am = (av_mask == 1).astype(np.uint8)  # 1 = artery
                vm = (av_mask == 2).astype(np.uint8)  # 2 = vein

                if am.sum() > 0 and vm.sum() > 0:
                    self.artery_mask = am
                    self.vein_mask = vm
                    used_classifier = True
        except Exception:
            used_classifier = False

        # 2) Heuristic fallback if classifier not available / failed
        if not used_classifier and self.vessel_mask.sum() > 0:
            # Arteries tend to be brighter in green channel than veins.
            if self.image.ndim == 3:
                green = self.image[:, :, 1].astype(np.float32)
            else:
                green = self.image.astype(np.float32)

            v_idx = self.vessel_mask > 0
            g_vals = green[v_idx]
            if g_vals.size == 0:
                return

            thr = np.median(g_vals)

            artery = np.zeros_like(self.vessel_mask, dtype=np.uint8)
            vein = np.zeros_like(self.vessel_mask, dtype=np.uint8)

            artery[v_idx] = (g_vals >= thr).astype(np.uint8)
            vein[v_idx] = (g_vals < thr).astype(np.uint8)

            self.artery_mask = artery
            self.vein_mask = vein

    # ------------------------------------------------------------------
    # Disc & zone geometry
    # ------------------------------------------------------------------
    def _estimate_disc_center_radius(self):
        """
        Approximate optic disc center & radius using bright region detection.
        """
        img = self.image
        if img.ndim == 3:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.astype(np.uint8)

        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), disc_r = cv2.minEnclosingCircle(c)
            cx, cy = int(x), int(y)
        else:
            cx, cy = w // 2, h // 2
            disc_r = 0.12 * min(h, w)

        return cx, cy, float(disc_r)

    def _create_zone_masks(self):
        """
        Creates boolean masks for zone_b and zone_c as concentric rings
        around the estimated disc.
        """
        cx, cy, disc_r = self._estimate_disc_center_radius()
        yy, xx = np.ogrid[:self.h, :self.w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        zone_b = (dist >= disc_r) & (dist < 2 * disc_r)
        zone_c = (dist >= 2 * disc_r) & (dist < 3 * disc_r)

        return {
            "zone_b": zone_b.astype(np.uint8),
            "zone_c": zone_c.astype(np.uint8),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_zone_features(self, zone_name: str):
        """
        Compute all zone-based metrics for a given zone name: "zone_b" or "zone_c".

        Returns a flat dict with keys matching the 72-feature naming scheme, e.g.:

            Fractal_dimension_zone_b,
            Artery_Vessel_density_zone_c,
            CRAE_Hubbard_zone_b,
            AVR_Knudtson_zone_c, ...
        """
        if zone_name not in self.zone_masks:
            raise ValueError(f"Unknown zone name: {zone_name}")

        zmask = self.zone_masks[zone_name].astype(bool)

        # all vessels in this zone
        all_mask = (self.vessel_mask > 0) & zmask

        # artery / vein masks
        art_mask = (self.artery_mask > 0) & zmask
        vein_mask = (self.vein_mask > 0) & zmask

        m_all = compute_vessel_metrics(all_mask.astype(np.uint8))
        m_art = compute_vessel_metrics(art_mask.astype(np.uint8))
        m_vein = compute_vessel_metrics(vein_mask.astype(np.uint8))

        zone_features = {}

        # ---------- whole-vessel zone metrics ----------
        z = "b" if zone_name.endswith("b") else "c"

        zone_features[f"Fractal_dimension_zone_{z}"] = m_all["fd"]
        zone_features[f"Vessel_density_zone_{z}"] = m_all["density"]
        zone_features[f"Average_width_zone_{z}"] = m_all["width"]
        zone_features[f"Distance_tortuosity_zone_{z}"] = m_all["dist_tort"]
        zone_features[f"Squared_curvature_tortuosity_zone_{z}"] = m_all["sq_tort"]
        zone_features[f"Tortuosity_density_zone_{z}"] = m_all["dens_tort"]

        # ---------- artery metrics ----------
        zone_features[f"Artery_Fractal_dimension_zone_{z}"] = m_art["fd"]
        zone_features[f"Artery_Vessel_density_zone_{z}"] = m_art["density"]
        zone_features[f"Artery_Average_width_zone_{z}"] = m_art["width"]
        zone_features[f"Artery_Distance_tortuosity_zone_{z}"] = m_art["dist_tort"]
        zone_features[f"Artery_Squared_curvature_tortuosity_zone_{z}"] = m_art["sq_tort"]
        zone_features[f"Artery_Tortuosity_density_zone_{z}"] = m_art["dens_tort"]

        # ---------- vein metrics ----------
        zone_features[f"Vein_Fractal_dimension_zone_{z}"] = m_vein["fd"]
        zone_features[f"Vein_Vessel_density_zone_{z}"] = m_vein["density"]
        zone_features[f"Vein_Average_width_zone_{z}"] = m_vein["width"]
        zone_features[f"Vein_Distance_tortuosity_zone_{z}"] = m_vein["dist_tort"]
        zone_features[f"Vein_Squared_curvature_tortuosity_zone_{z}"] = m_vein["sq_tort"]
        zone_features[f"Vein_Tortuosity_density_zone_{z}"] = m_vein["dens_tort"]

        # ---------- CRAE / CRVE / AVR (Hubbard & Knudtson – approximated from widths) ----------
        crae = m_art["width"]
        crve = m_vein["width"]

        def safe_ratio(a, b):
            if b is None or b == 0 or np.isnan(b):
                return np.nan
            return float(a / b) if a is not None and not np.isnan(a) else np.nan

        # we use mean widths as surrogates for CRAE / CRVE
        zone_features[f"CRAE_Hubbard_zone_{z}"] = crae
        zone_features[f"CRVE_Hubbard_zone_{z}"] = crve
        zone_features[f"AVR_Hubbard_zone_{z}"] = safe_ratio(crae, crve)

        zone_features[f"CRAE_Knudtson_zone_{z}"] = crae
        zone_features[f"CRVE_Knudtson_zone_{z}"] = crve
        zone_features[f"AVR_Knudtson_zone_{z}"] = safe_ratio(crae, crve)

        return zone_features