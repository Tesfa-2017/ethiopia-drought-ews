"""
Official Administrative Boundary Zonal Statistics & EDRMC Risk Evaluation Script.
Processes Regions (Admin 1), Zones (Admin 2), and Woredas (Admin 3) using official Ethiopian Shapefiles.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from pathlib import Path
import torch
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rioxarray

from src.config import (
    CHECKPOINT_DIR, DATA_DIR, EDRMC_PROB_THRESHOLD, EDRMC_AREA_THRESHOLD, EDRMC_SPI3_DROUGHT_VAL
)
from src.dataset import EthiopiaDroughtDataModule, load_nc_dataset
from src.model import PrithviWxCForDrought
from src.inference import find_best_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OfficialZonalStats")


def process_admin_shapefile(shp_path, name_col, parent_col=None, out_filename=None):
    """
    Computes exact raster zonal statistics per polygon in an official shapefile.
    """
    if not os.path.exists(shp_path):
        logger.warning(f"Shapefile not found: {shp_path}")
        return None

    logger.info(f"Loading official shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs("EPSG:4326")

    # Load raster files
    spi3_tif = DATA_DIR / "ethiopia_spi3_predictions.tif"
    prob_tif = DATA_DIR / "ethiopia_drought_probability.tif"

    if not spi3_tif.exists() or not prob_tif.exists():
        logger.warning("GeoTIFF rasters not found. Generating predictions first...")
        from src.inference import run_inference_and_edrmc_eval
        run_inference_and_edrmc_eval()

    r_spi3 = rioxarray.open_rasterio(spi3_tif).squeeze()
    r_prob = rioxarray.open_rasterio(prob_tif).squeeze()

    mean_spi3_list = []
    prob_drought_list = []
    area_drought_pct_list = []
    activation_status_list = []

    logger.info(f"Computing zonal statistics across {len(gdf)} administrative features...")

    for idx, row in gdf.iterrows():
        geom = row.geometry
        try:
            # Clip rasters to feature geometry
            clipped_spi3 = r_spi3.rio.clip([geom], gdf.crs, drop=True)
            clipped_prob = r_prob.rio.clip([geom], gdf.crs, drop=True)

            vals_spi3 = clipped_spi3.values
            vals_spi3 = vals_spi3[~np.isnan(vals_spi3)]

            vals_prob = clipped_prob.values
            vals_prob = vals_prob[~np.isnan(vals_prob)]

            if len(vals_spi3) == 0:
                mean_spi3_val = 0.0
                prob_drought_val = 0.0
                area_drought_pct_val = 0.0
            else:
                mean_spi3_val = float(np.mean(vals_spi3))
                prob_drought_val = float(np.mean(vals_prob))
                area_drought_pct_val = float(np.mean(vals_spi3 < EDRMC_SPI3_DROUGHT_VAL))

            if prob_drought_val >= EDRMC_PROB_THRESHOLD and area_drought_pct_val > EDRMC_AREA_THRESHOLD:
                if mean_spi3_val < EDRMC_SPI3_DROUGHT_VAL:
                    status = "FULL ACTIVATION"
                else:
                    status = "PARTIAL ACTIVATION"
            else:
                status = "NO ACTIVATION"

        except Exception:
            mean_spi3_val = 0.0
            prob_drought_val = 0.0
            area_drought_pct_val = 0.0
            status = "NO ACTIVATION"

        mean_spi3_list.append(round(mean_spi3_val, 3))
        prob_drought_list.append(round(prob_drought_val * 100, 1))
        area_drought_pct_list.append(round(area_drought_pct_val * 100, 1))
        activation_status_list.append(status)

    gdf["mean_spi3"] = mean_spi3_list
    gdf["prob_drought_pct"] = prob_drought_list
    gdf["area_drought_pct"] = area_drought_pct_list
    gdf["edrmc_status"] = activation_status_list

    # Save summary files
    out_dir = DATA_DIR / "official_admin_drought_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_filename:
        geojson_out = out_dir / f"{out_filename}.geojson"
        shp_out = out_dir / f"{out_filename}.shp"
        gdf.to_file(geojson_out, driver="GeoJSON")
        try:
            gdf.to_file(shp_out)
        except Exception as e:
            logger.warning(f"Shapefile export warning: {e}")
        logger.info(f"Successfully exported official summary to: {geojson_out}")

    return gdf


def run_all_official_zonal_evaluations():
    """
    Executes zonal evaluations across official Ethiopian Regions, Zones, and Woredas.
    """
    base_shp_dir = DATA_DIR / "eth_admin_boundaries"

    shp1 = base_shp_dir / "eth_admin1.shp"
    shp2 = base_shp_dir / "eth_admin2.shp"
    shp3 = base_shp_dir / "eth_admin3.shp"

    logger.info("========================================================================")
    logger.info(" Processing Official Ethiopian Regions (Admin 1)...")
    gdf_regions = process_admin_shapefile(shp1, name_col="adm1_name", out_filename="eth_regions_drought_summary")

    if gdf_regions is not None:
        summary_cols = ["adm1_name", "mean_spi3", "prob_drought_pct", "area_drought_pct", "edrmc_status"]
        print("\n================ REGIONAL DROUGHT RISK SUMMARY (ADMIN 1) ================\n")
        print(gdf_regions[summary_cols].to_string(index=False))

    logger.info("========================================================================")
    logger.info(" Processing Official Ethiopian Zones (Admin 2)...")
    gdf_zones = process_admin_shapefile(shp2, name_col="adm2_name", parent_col="adm1_name", out_filename="eth_zones_drought_summary")

    if gdf_zones is not None:
        summary_cols = ["adm1_name", "adm2_name", "mean_spi3", "prob_drought_pct", "area_drought_pct", "edrmc_status"]
        print("\n================ ZONAL DROUGHT RISK SUMMARY (ADMIN 2 - TOP 10) ================\n")
        print(gdf_zones[summary_cols].head(10).to_string(index=False))

    logger.info("========================================================================")
    logger.info(" Processing Official Ethiopian Woredas (Admin 3)...")
    gdf_woredas = process_admin_shapefile(shp3, name_col="adm3_name", parent_col="adm2_name", out_filename="eth_woredas_drought_summary")

    if gdf_woredas is not None:
        summary_cols = ["adm1_name", "adm2_name", "adm3_name", "mean_spi3", "prob_drought_pct", "area_drought_pct", "edrmc_status"]
        print("\n================ WOREDA DROUGHT RISK SUMMARY (ADMIN 3 - TOP 10) ================\n")
        print(gdf_woredas[summary_cols].head(10).to_string(index=False))

    logger.info("\nAll official administrative zonal statistics successfully generated!")


if __name__ == "__main__":
    run_all_official_zonal_evaluations()
