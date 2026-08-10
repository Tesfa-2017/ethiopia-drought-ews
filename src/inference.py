"""
Inference Script & EDRMC Risk Activation Threshold Application.
Loads best trained checkpoint, computes zonal statistics over Ethiopian regions,
and evaluates Full, Partial, or No Activation according to EDRMC guidelines.
"""

import sys
import os
import subprocess

# 1. Prioritize virtualenv site-packages at position 0
venv_site = "/tmp/uv-venv/lib/python3.13/site-packages"
if os.path.exists(venv_site):
    if venv_site in sys.path:
        sys.path.remove(venv_site)
    sys.path.insert(0, venv_site)

# 2. Keep system site-packages at the end of sys.path for pre-installed fallback packages (lightning, etc.)
sys_site = "/usr/local/lib/python3.13/site-packages"
if os.path.exists(sys_site):
    if sys_site in sys.path:
        sys.path.remove(sys_site)
    sys.path.append(sys_site)

# Verify & force install matching torchvision binary if needed
try:
    import torchvision
    import torchvision.ops
except Exception as e:
    print(f"Force aligning torchvision for Python 3.13... ({e})")
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "torchvision"], check=True)
    for mod in list(sys.modules.keys()):
        if mod == "torchvision" or mod.startswith("torchvision."):
            del sys.modules[mod]

# Auto-install missing pipeline dependencies (e.g. earthengine-api, terratorch, rioxarray)
required_pkgs = ["earthengine-api", "terratorch", "torchgeo", "geedim", "geemap", "rioxarray", "xarray", "netcdf4", "mlflow"]
missing = []
for pkg in required_pkgs:
    mod_name = "ee" if pkg == "earthengine-api" else pkg.replace("-", "_")
    try:
        __import__(mod_name)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Auto-installing missing pipeline dependencies in MoLab: {missing}...")
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from pathlib import Path
import shutil
import torch
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray

from src.config import (
    CHECKPOINT_DIR, DATA_DIR, ETHIOPIA_ZONES,
    EDRMC_PROB_THRESHOLD, EDRMC_AREA_THRESHOLD, EDRMC_SPI3_DROUGHT_VAL
)
from src.dataset import EthiopiaDroughtDataModule, load_nc_dataset
from src.model import PrithviWxCForDrought

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InferencePipeline")


def find_best_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """
    Finds the latest or best model checkpoint file.
    """
    ckpts = list(Path(checkpoint_dir).glob("*.ckpt"))
    if not ckpts:
        logger.warning(f"No checkpoint found in {checkpoint_dir}. Searching fallback paths...")
        return None
    best_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found checkpoint file: {best_ckpt}")
    return best_ckpt


def run_inference_and_edrmc_eval():
    """
    Executes inference over the test dataset and applies EDRMC decision thresholds.
    """
    logger.info("========================================================================")
    logger.info(" Starting EDRMC Drought Risk Evaluation over Ethiopian Administrative Zones")
    logger.info("========================================================================")

    dataset_file = DATA_DIR / "ethiopia_drought_dataset.nc"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file missing at: {dataset_file}. Please run data_pipeline.py first.")

    # Step 1: Load DataModule
    datamodule = EthiopiaDroughtDataModule(nc_file_path=dataset_file)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # Step 2: Load Model Checkpoint
    ckpt_path = find_best_checkpoint()
    model = None
    if ckpt_path and ckpt_path.exists():
        temp_ckpt = f"{ckpt_path}.tmp_{os.getpid()}.ckpt"
        try:
            shutil.copy2(ckpt_path, temp_ckpt)
            logger.info(f"Loading weights from checkpoint snapshot: {ckpt_path}")
            model = PrithviWxCForDrought.load_from_checkpoint(temp_ckpt, map_location="cpu")
        except Exception as e:
            logger.warning(f"Could not load checkpoint snapshot ({e}). Falling back to base weights...")
            model = None
        finally:
            if os.path.exists(temp_ckpt):
                try:
                    os.remove(temp_ckpt)
                except Exception:
                    pass

    if model is None:
        logger.warning("Initializing model with base weights...")
        model = PrithviWxCForDrought()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Step 3: Run Batch Predictions
    all_preds = []
    all_targets = []

    logger.info(f"Running predictions on test set ({len(test_loader)} batches)...")
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(test_loader):
            logger.info(f"Evaluating test batch {b_idx + 1}/{len(test_loader)}...")
            x = x.to(device)
            preds = model(x)  # (B, H, W)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)    # (N_test, H, W)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N_test, H, W)

    # Step 4: Reconstruct Spatial Coordinates
    ds = load_nc_dataset(dataset_file)
    lats = ds.lat.values
    lons = ds.lon.values

    # Mean predicted SPI-3 map across test period
    mean_pred_map = np.mean(preds_arr, axis=0)  # (H, W)

    # Step 5: Evaluate EDRMC Activation Thresholds per Zone
    logger.info("\n================ EDRMC ACTIVATION STATUS REPORT PER ZONE ================\n")

    zone_results = []

    for zone_name, bbox in ETHIOPIA_ZONES.items():
        min_lon, min_lat, max_lon, max_lat = bbox

        # Create spatial mask for the zone with floating point tolerance
        lat_mask = (lats >= (min_lat - 1e-4)) & (lats <= (max_lat + 1e-4))
        lon_mask = (lons >= (min_lon - 1e-4)) & (lons <= (max_lon + 1e-4))

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        zone_pixels = mean_pred_map[np.ix_(lat_mask, lon_mask)]
        zone_all_preds = preds_arr[:, lat_indices[:, None], lon_indices]

        if zone_pixels.size == 0:
            logger.warning(f"Zone {zone_name} has no overlapping grid pixels.")
            continue

        # Metric 1: Mean SPI-3 in zone
        mean_spi3_val = float(np.mean(zone_pixels))

        # Metric 2: Percentage of Area (pixels) with SPI-3 < -1
        area_drought_pct = float(np.mean(zone_pixels < EDRMC_SPI3_DROUGHT_VAL))

        # Metric 3: Probability of Drought (SPI-3 < -1) across time predictions
        prob_drought = float(np.mean(zone_all_preds < EDRMC_SPI3_DROUGHT_VAL))

        # EDRMC Threshold Logic Matrix
        # FULL ACTIVATION: Prob >= 45% AND Area > 50% AND mean SPI-3 < -1
        # PARTIAL ACTIVATION: Prob >= 45% AND Area > 50% AND mean SPI-3 >= -1
        # NO ACTIVATION: otherwise
        if prob_drought >= EDRMC_PROB_THRESHOLD and area_drought_pct > EDRMC_AREA_THRESHOLD:
            if mean_spi3_val < EDRMC_SPI3_DROUGHT_VAL:
                activation_status = "FULL ACTIVATION"
            else:
                activation_status = "PARTIAL ACTIVATION"
        else:
            activation_status = "NO ACTIVATION"

        zone_results.append({
            "Zone": zone_name,
            "Mean SPI-3": round(mean_spi3_val, 3),
            "Prob(SPI-3 < -1)": f"{prob_drought * 100:.1f}%",
            "Area(SPI-3 < -1)": f"{area_drought_pct * 100:.1f}%",
            "EDRMC Activation Status": activation_status
        })

    # Print Summary Table
    df_results = pd.DataFrame(zone_results)
    print(df_results.to_string(index=False))
    print("\n========================================================================\n")

    # Step 6: Export Spatial GeoTIFF Rasters (EPSG:4326 WGS84)
    logger.info("Exporting Spatial GeoTIFF Rasters for GIS mapping...")

    # 1. Mean SPI-3 Predicted Map Raster (.tif)
    da_spi3 = xr.DataArray(
        mean_pred_map.astype(np.float32),
        dims=["lat", "lon"],
        coords={"lat": lats.astype(np.float32), "lon": lons.astype(np.float32)}
    )
    da_spi3 = da_spi3.rio.write_crs("EPSG:4326")
    da_spi3 = da_spi3.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    geotiff_spi3_file = DATA_DIR / "ethiopia_spi3_predictions.tif"
    try:
        da_spi3.rio.to_raster(geotiff_spi3_file)
        logger.info(f"Mean SPI-3 GeoTIFF raster exported to: {geotiff_spi3_file}")
    except Exception as e:
        logger.warning(f"Could not overwrite {geotiff_spi3_file} ({e}). File is currently open in QGIS.")

    # 2. Drought Probability Map (SPI-3 < -1) Raster (.tif)
    prob_map = np.mean((preds_arr < EDRMC_SPI3_DROUGHT_VAL).astype(np.float32), axis=0)
    da_prob = xr.DataArray(
        prob_map.astype(np.float32),
        dims=["lat", "lon"],
        coords={"lat": lats.astype(np.float32), "lon": lons.astype(np.float32)}
    )
    da_prob = da_prob.rio.write_crs("EPSG:4326")
    da_prob = da_prob.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    geotiff_prob_file = DATA_DIR / "ethiopia_drought_probability.tif"
    try:
        da_prob.rio.to_raster(geotiff_prob_file)
        logger.info(f"Drought Probability GeoTIFF raster exported to: {geotiff_prob_file}")
    except Exception as e:
        logger.warning(f"Could not overwrite {geotiff_prob_file} ({e}). File is currently open in QGIS.")

    # Step 7: Run Official Ethiopian Administrative Boundary Evaluation (Regions, Zones, Woredas)
    try:
        from src.official_zonal_stats import run_all_official_zonal_evaluations
        run_all_official_zonal_evaluations()
    except Exception as e:
        logger.warning(f"Could not run official administrative shapefile stats: {e}")

    return df_results


if __name__ == "__main__":
    run_inference_and_edrmc_eval()
