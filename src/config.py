"""
Central Configuration for Ethiopia Prithvi WxC Drought Prediction Pipeline
Strictly clipped and bounded to Ethiopia AOI.
"""

import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Paths & Directories
BASE_DIR = Path("/app") if os.path.exists("/app") else Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", BASE_DIR / "checkpoints"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{BASE_DIR / 'mlruns'}")
GEE_SERVICE_ACCOUNT_KEY = os.getenv("GEE_SERVICE_ACCOUNT_KEY", str(BASE_DIR / "credentials" / "gee-key.json"))

for d in [DATA_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Spatial Constraints & AOI Setup (Ethiopia Only)
# -----------------------------------------------------------------------------
GAUL_COLLECTION = "FAO/GAUL/2015/level0"
GAUL_COUNTRY_FIELD = "ADM0_NAME"
GAUL_COUNTRY_NAME = "Ethiopia"

# Hard Fallback Bounding Box for Ethiopia [min_lon, min_lat, max_lon, max_lat]
ETHIOPIA_BBOX_FALLBACK = [33.0, 3.0, 48.0, 15.0]

# Unified Spatial Grid Resolution
SPATIAL_RESOLUTION = 0.05  # degrees (~5.5 km grid cell)


def assert_ethiopia_bounds(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    """
    Enforces that spatial datasets lie strictly within Ethiopia's extent boundaries.
    """
    margin = 0.5  # Allow minor tolerance for cell edge rounding
    assert min_lon >= (ETHIOPIA_BBOX_FALLBACK[0] - margin), \
        f"Bounding box min_lon {min_lon} extends west outside Ethiopia ({ETHIOPIA_BBOX_FALLBACK[0]})"
    assert max_lon <= (ETHIOPIA_BBOX_FALLBACK[2] + margin), \
        f"Bounding box max_lon {max_lon} extends east outside Ethiopia ({ETHIOPIA_BBOX_FALLBACK[2]})"
    assert min_lat >= (ETHIOPIA_BBOX_FALLBACK[1] - margin), \
        f"Bounding box min_lat {min_lat} extends south outside Ethiopia ({ETHIOPIA_BBOX_FALLBACK[1]})"
    assert max_lat <= (ETHIOPIA_BBOX_FALLBACK[3] + margin), \
        f"Bounding box max_lat {max_lat} extends north outside Ethiopia ({ETHIOPIA_BBOX_FALLBACK[3]})"
    return True


# -----------------------------------------------------------------------------
# Data Sources & Index URLs
# -----------------------------------------------------------------------------
URL_ENSO_NINO34 = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
URL_IOD_DMI = "https://www.bom.gov.au/climate/enso/indices/australian_dmi.csv"

GEE_COLLECTIONS = {
    "precipitation": "UCSB-CHG/CHIRPS/DAILY",
    "temperature": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "soil_moisture": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "ndvi": "MODIS/061/MOD13Q1"
}

# -----------------------------------------------------------------------------
# Input / Output Tensor Specifications
# -----------------------------------------------------------------------------
NUM_TIMESTEPS = 2  # 2 preceding context months (t-2, t-1)
NUM_INPUT_CHANNELS = 6
# Channels: 0: Precip, 1: Temp, 2: Soil Moisture, 3: NDVI, 4: ENSO, 5: IOD
INPUT_CHANNELS = ["precipitation", "temperature", "soil_moisture", "ndvi", "enso_nino34", "iod_dmi"]
NUM_OUTPUT_CHANNELS = 1  # Target SPI-3 map

# Historical Range for Training Data
START_YEAR = 2006
END_YEAR = 2026

# -----------------------------------------------------------------------------
# Model & Fine-Tuning Hyperparameters
# -----------------------------------------------------------------------------
PRETRAINED_MODEL_ID = "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SCHEDULER_NAME = "CosineAnnealingLR"
LOSS_FUNCTION = "MSE"
BATCH_SIZE = 8
PRECISION = "16-mixed"  # Mixed precision (fp16)
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Data Splits (Temporally Stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# -----------------------------------------------------------------------------
# EDRMC Zone Bounding Boxes & Risk Thresholds
# -----------------------------------------------------------------------------
ETHIOPIA_ZONES = {
    "Tigray": [36.5, 12.2, 39.9, 14.9],
    "Amhara": [35.2, 8.7, 40.2, 13.8],
    "Oromia": [34.1, 3.4, 42.9, 10.3],
    "SNNP": [34.3, 4.4, 39.1, 8.4],
    "Somali": [40.0, 4.0, 48.0, 11.0]
}

EDRMC_PROB_THRESHOLD = 0.45   # 45% probability threshold
EDRMC_AREA_THRESHOLD = 0.50   # 50% spatial area percentage threshold
EDRMC_SPI3_DROUGHT_VAL = -1.0 # SPI-3 < -1 threshold
