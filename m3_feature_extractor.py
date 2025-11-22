import cv2
import numpy as np
import pandas as pd

from M3.M3_feature_whole_pic.retipy.retipy.retina import Retina
from M3.M3_feature_zone.retipy.zone_feature_extractor import (
    ZoneFeatureExtractor,
    compute_vessel_metrics,
)

import sys
import os

# Dynamically add M3 folder path (works on Streamlit Cloud and local)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
M3_PATH = os.path.abspath(os.path.join(BASE_DIR, "M3"))
if M3_PATH not in sys.path:
    sys.path.append(M3_PATH)


def extract_m3_features(image_array: np.ndarray) -> pd.DataFrame:
    """
    Extract the full 72-feature M3 set:

        - Global disc/cup/CDR features
        - Global fractal / density / width / tortuosity
        - Global artery & vein metrics
        - Zone B and Zone C metrics (including A/V + CRAE/CRVE/AVR)

    Input:
        image_array: RGB image as numpy array (H x W x 3)

    Output:
        Single-row Pandas DataFrame with 72 columns.
    """

    # Ensure BGR for OpenCV-based code
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------------
    # 1. Disc / cup metrics via Retina
    # --------------------------------------------------------------
    retina = Retina(img_bgr, image_path="", store_path="")

    try:
        disc_h = retina.disc_height()
    except Exception:
        disc_h = np.nan

    try:
        disc_w = retina.disc_width()
    except Exception:
        disc_w = np.nan

    try:
        cup_h = retina.cup_height()
    except Exception:
        cup_h = np.nan

    try:
        cup_w = retina.cup_width()
    except Exception:
        cup_w = np.nan

    try:
        cdr_v = retina.vcdr()
    except Exception:
        cdr_v = np.nan

    try:
        cdr_h = retina.hcdr()
    except Exception:
        cdr_h = np.nan

    # --------------------------------------------------------------
    # 2. Global vessel mask + AV masks via ZoneFeatureExtractor
    #    (we reuse its segmentation + heuristics)
    # --------------------------------------------------------------
    try:
        zone_extractor = ZoneFeatureExtractor(img_bgr)
        vessel_mask = zone_extractor.vessel_mask
        artery_mask = zone_extractor.artery_mask
        vein_mask = zone_extractor.vein_mask
    except Exception:
        h, w = img_bgr.shape[:2]
        zone_extractor = None
        vessel_mask = np.zeros((h, w), dtype=np.uint8)
        artery_mask = np.zeros_like(vessel_mask)
        vein_mask = np.zeros_like(vessel_mask)

    # --------------------------------------------------------------
    # 3. Global metrics from masks (using same logic as zones)
    # --------------------------------------------------------------
    m_all = compute_vessel_metrics(vessel_mask)
    m_art = compute_vessel_metrics(artery_mask)
    m_vein = compute_vessel_metrics(vein_mask)

    # --------------------------------------------------------------
    # 4. Assemble feature dict
    # --------------------------------------------------------------
    features = {}

    # Global disc/cup/CDR
    features["Disc_height"] = disc_h
    features["Disc_width"] = disc_w
    features["Cup_height"] = cup_h
    features["Cup_width"] = cup_w
    features["CDR_vertical"] = cdr_v
    features["CDR_horizontal"] = cdr_h

    # Global all-vessel metrics
    features["Fractal_dimension"] = m_all["fd"]
    features["Vessel_density"] = m_all["density"]
    features["Average_width"] = m_all["width"]
    features["Distance_tortuosity"] = m_all["dist_tort"]
    features["Squared_curvature_tortuosity"] = m_all["sq_tort"]
    features["Tortuosity_density"] = m_all["dens_tort"]

    # Global artery metrics
    features["Artery_Fractal_dimension"] = m_art["fd"]
    features["Artery_Vessel_density"] = m_art["density"]
    features["Artery_Average_width"] = m_art["width"]
    features["Artery_Distance_tortuosity"] = m_art["dist_tort"]
    features["Artery_Squared_curvature_tortuosity"] = m_art["sq_tort"]
    features["Artery_Tortuosity_density"] = m_art["dens_tort"]

    # Global vein metrics
    features["Vein_Fractal_dimension"] = m_vein["fd"]
    features["Vein_Vessel_density"] = m_vein["density"]
    features["Vein_Average_width"] = m_vein["width"]
    features["Vein_Distance_tortuosity"] = m_vein["dist_tort"]
    features["Vein_Squared_curvature_tortuosity"] = m_vein["sq_tort"]
    features["Vein_Tortuosity_density"] = m_vein["dens_tort"]

    # --------------------------------------------------------------
    # 5. Zone B & C features (this will add all the zone_* columns)
    # --------------------------------------------------------------
    if zone_extractor is not None:
        for zone in ["zone_b", "zone_c"]:
            try:
                zdict = zone_extractor.extract_zone_features(zone)
                features.update(zdict)
            except Exception:
                # if a zone fails, leave its fields as NaN (they won't exist here)
                pass

    # Make sure *all* expected 72 keys exist (fill any missing with NaN)
    expected_keys = [
        'Disc_height', 'Disc_width', 'Cup_height', 'Cup_width',
        'CDR_vertical', 'CDR_horizontal', 'Fractal_dimension',
        'Vessel_density', 'Average_width', 'Distance_tortuosity',
        'Squared_curvature_tortuosity', 'Tortuosity_density',
        'Artery_Fractal_dimension', 'Artery_Vessel_density',
        'Artery_Average_width', 'Artery_Distance_tortuosity',
        'Artery_Squared_curvature_tortuosity', 'Artery_Tortuosity_density',
        'Vein_Fractal_dimension', 'Vein_Vessel_density',
        'Vein_Average_width', 'Vein_Distance_tortuosity',
        'Vein_Squared_curvature_tortuosity', 'Vein_Tortuosity_density',
        'Fractal_dimension_zone_b', 'Vessel_density_zone_b',
        'Average_width_zone_b', 'Distance_tortuosity_zone_b',
        'Squared_curvature_tortuosity_zone_b', 'Tortuosity_density_zone_b',
        'Artery_Fractal_dimension_zone_b', 'Artery_Vessel_density_zone_b',
        'Artery_Average_width_zone_b', 'Artery_Distance_tortuosity_zone_b',
        'Artery_Squared_curvature_tortuosity_zone_b',
        'Artery_Tortuosity_density_zone_b', 'CRAE_Hubbard_zone_b',
        'CRAE_Knudtson_zone_b', 'Vein_Fractal_dimension_zone_b',
        'Vein_Vessel_density_zone_b', 'Vein_Average_width_zone_b',
        'Vein_Distance_tortuosity_zone_b',
        'Vein_Squared_curvature_tortuosity_zone_b',
        'Vein_Tortuosity_density_zone_b', 'CRVE_Hubbard_zone_b',
        'CRVE_Knudtson_zone_b', 'AVR_Hubbard_zone_b',
        'AVR_Knudtson_zone_b', 'Fractal_dimension_zone_c',
        'Vessel_density_zone_c', 'Average_width_zone_c',
        'Distance_tortuosity_zone_c', 'Squared_curvature_tortuosity_zone_c',
        'Tortuosity_density_zone_c', 'Artery_Fractal_dimension_zone_c',
        'Artery_Vessel_density_zone_c', 'Artery_Average_width_zone_c',
        'Artery_Distance_tortuosity_zone_c',
        'Artery_Squared_curvature_tortuosity_zone_c',
        'Artery_Tortuosity_density_zone_c', 'CRAE_Hubbard_zone_c',
        'CRAE_Knudtson_zone_c', 'Vein_Fractal_dimension_zone_c',
        'Vein_Vessel_density_zone_c', 'Vein_Average_width_zone_c',
        'Vein_Distance_tortuosity_zone_c',
        'Vein_Squared_curvature_tortuosity_zone_c',
        'Vein_Tortuosity_density_zone_c', 'CRVE_Hubbard_zone_c',
        'CRVE_Knudtson_zone_c', 'AVR_Hubbard_zone_c',
        'AVR_Knudtson_zone_c',
    ]

    for k in expected_keys:
        if k not in features:
            features[k] = np.nan

    return pd.DataFrame([features])
