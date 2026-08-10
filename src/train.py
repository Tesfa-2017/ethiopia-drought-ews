"""
Main Training Loop for Fine-Tuning Prithvi WxC on Ethiopia Drought (SPI-3).
Integrates MLflow Logging, Early Stopping, Model Checkpoint, and Mixed Precision (FP16).
"""

import sys
import os
import subprocess

# Purge cached broken system torchvision from sys.modules
for mod in list(sys.modules.keys()):
    if mod == "torchvision" or mod.startswith("torchvision."):
        del sys.modules[mod]

# Fix MoLab path conflict: Prioritize virtualenv site-packages over system site-packages
venv_site = "/tmp/uv-venv/lib/python3.13/site-packages"
if os.path.exists(venv_site):
    if venv_site in sys.path:
        sys.path.remove(venv_site)
    sys.path.insert(0, venv_site)
sys_site = "/usr/local/lib/python3.13/site-packages"
if sys_site in sys.path:
    sys.path.remove(sys_site)

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.config import (
    DATA_DIR, CHECKPOINT_DIR, MLFLOW_TRACKING_URI,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, PRECISION
)
from src.data_pipeline import initialize_gee, generate_ethiopia_grid_dataset
from src.dataset import EthiopiaDroughtDataModule
from src.model import PrithviWxCForDrought

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")


def run_training():
    """
    Executes the full pipeline: Data preparation, DataModule setup, MLflow logging,
    and PyTorch Lightning training.
    """
    logger.info("========================================================================")
    logger.info(" Starting Prithvi WxC Fine-Tuning Pipeline for Ethiopia Drought (SPI-3)")
    logger.info("========================================================================")

    # Step 1: Ensure dataset exists or generate it
    dataset_file = DATA_DIR / "ethiopia_drought_dataset.nc"
    if not dataset_file.exists():
        logger.info("Dataset not found. Running GEE Data Extraction & Resampling pipeline...")
        initialize_gee()
        generate_ethiopia_grid_dataset()
    else:
        logger.info(f"Existing dataset found at: {dataset_file}")

    # Step 2: Initialize DataModule & Model
    datamodule = EthiopiaDroughtDataModule(nc_file_path=dataset_file)
    model = PrithviWxCForDrought()

    # Step 3: Configure MLflow Logger
    logger.info(f"Initializing MLflow Logger with Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow_logger = MLFlowLogger(
        experiment_name="ethiopia-prithvi-drought-spi3",
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # Step 4: Configure Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best_prithvi_ethiopia_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        mode="min",
        verbose=True
    )

    # Step 5: Hardware Acceleration & Precision
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    logger.info(f"Training Hardware: Accelerator={accelerator.upper()}, Devices={devices}, Precision={PRECISION}")

    # Step 6: Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        precision=PRECISION if accelerator == "gpu" else 32,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=5
    )

    # Step 7: Train Model (Auto-resume from latest checkpoint if available)
    ckpts = [p for p in CHECKPOINT_DIR.glob("*.ckpt") if not p.name.endswith(".tmp.ckpt")]
    resume_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None
    if resume_ckpt and resume_ckpt.exists():
        logger.info(f"Resuming training seamlessly from checkpoint: {resume_ckpt}")
    else:
        resume_ckpt = None

    logger.info("Fitting model...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # Step 8: Test Model
    logger.info("Running evaluation on test set...")
    trainer.test(model, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path
    logger.info(f"Training completed successfully! Best model saved at: {best_path}")
    return best_path


if __name__ == "__main__":
    run_training()
