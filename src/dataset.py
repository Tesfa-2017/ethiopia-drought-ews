"""
PyTorch Dataset and LightningDataModule for Ethiopia Prithvi WxC Fine-Tuning.
Formats (batch, time_steps=2, channels=6, H, W) input tensors and target SPI-3 maps (batch, H, W).
"""

import os
import shutil
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import xarray as xr
import numpy as np

from src.config import (
    DATA_DIR, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, NUM_TIMESTEPS
)

logger = logging.getLogger("EthiopiaDataset")


def load_nc_dataset(nc_file_path):
    """
    Loads NetCDF dataset safely into memory.
    If the file is locked by a writer process, falls back to a file copy snapshot.
    """
    try:
        with xr.open_dataset(nc_file_path) as open_ds:
            return open_ds.load()
    except Exception as e:
        logger.warning(f"Direct NetCDF load failed ({e}). Attempting file copy snapshot read...")
        import uuid
        temp_path = f"{nc_file_path}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}.nc"
        shutil.copy2(nc_file_path, temp_path)
        try:
            with xr.open_dataset(temp_path) as open_ds:
                ds = open_ds.load()
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return ds


class EthiopiaDroughtDataset(Dataset):
    """
    PyTorch Dataset for Drought Prediction over Ethiopia.
    Input Tensor Shape: (time_steps=2, channels=6, H, W)
      Channels 0-3: Spatial variables (Precip, Temp, Soil Moisture, NDVI)
      Channels 4-5: ENSO & IOD climate indices (replicated across grid H x W)
    Target Tensor Shape: (H, W) -> SPI-3 map at target month t
    """
    def __init__(self, nc_file_path, indices):
        super().__init__()
        self.nc_file_path = nc_file_path
        self.indices = indices

        # Load NetCDF dataset into memory safely
        self.ds = load_nc_dataset(nc_file_path)

        # Normalize features
        self.means = {
            "precipitation": float(self.ds["precipitation"].mean()),
            "temperature": float(self.ds["temperature"].mean()),
            "soil_moisture": float(self.ds["soil_moisture"].mean()),
            "ndvi": float(self.ds["ndvi"].mean()),
            "enso_nino34": float(self.ds["enso_nino34"].mean()),
            "iod_dmi": float(self.ds["iod_dmi"].mean()),
        }
        self.stds = {
            "precipitation": float(self.ds["precipitation"].std()) + 1e-6,
            "temperature": float(self.ds["temperature"].std()) + 1e-6,
            "soil_moisture": float(self.ds["soil_moisture"].std()) + 1e-6,
            "ndvi": float(self.ds["ndvi"].std()) + 1e-6,
            "enso_nino34": float(self.ds["enso_nino34"].std()) + 1e-6,
            "iod_dmi": float(self.ds["iod_dmi"].std()) + 1e-6,
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        target_t = self.indices[idx]
        # 2 preceding context months (t-2, t-1)
        t_start = target_t - NUM_TIMESTEPS
        t_end = target_t

        precip = (self.ds["precipitation"].isel(time=slice(t_start, t_end)).values - self.means["precipitation"]) / self.stds["precipitation"]
        temp = (self.ds["temperature"].isel(time=slice(t_start, t_end)).values - self.means["temperature"]) / self.stds["temperature"]
        soil = (self.ds["soil_moisture"].isel(time=slice(t_start, t_end)).values - self.means["soil_moisture"]) / self.stds["soil_moisture"]
        ndvi = (self.ds["ndvi"].isel(time=slice(t_start, t_end)).values - self.means["ndvi"]) / self.stds["ndvi"]

        enso = (self.ds["enso_nino34"].isel(time=slice(t_start, t_end)).values - self.means["enso_nino34"]) / self.stds["enso_nino34"]
        iod = (self.ds["iod_dmi"].isel(time=slice(t_start, t_end)).values - self.means["iod_dmi"]) / self.stds["iod_dmi"]

        # Shape of spatial features: (2, H, W)
        _, H, W = precip.shape

        # Replicate scalar ENSO and IOD across spatial grid (2, H, W)
        enso_grid = np.repeat(enso[:, None, None], H, axis=1)
        enso_grid = np.repeat(enso_grid, W, axis=2)

        iod_grid = np.repeat(iod[:, None, None], H, axis=1)
        iod_grid = np.repeat(iod_grid, W, axis=2)

        # Stack into 6 input channels: (2, 6, H, W)
        # channels 0: precip, 1: temp, 2: soil, 3: ndvi, 4: enso, 5: iod
        inputs = np.stack([precip, temp, soil, ndvi, enso_grid, iod_grid], axis=1)

        # Target SPI-3 map at month target_t: (H, W)
        target_spi3 = self.ds["spi3"].isel(time=target_t).values

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target_spi3, dtype=torch.float32)


class EthiopiaDroughtDataModule(pl.LightningDataModule):
    """
    Lightning DataModule with Temporally Stratified Splitting (70% Train, 15% Val, 15% Test).
    """
    def __init__(self, nc_file_path=None, batch_size=BATCH_SIZE):
        super().__init__()
        self.nc_file_path = nc_file_path or (DATA_DIR / "ethiopia_drought_dataset.nc")
        self.batch_size = batch_size

    def setup(self, stage=None):
        ds = load_nc_dataset(self.nc_file_path)
        total_timesteps = len(ds.time)

        # Valid indices start at NUM_TIMESTEPS (month index 2 onwards)
        valid_indices = list(range(NUM_TIMESTEPS, total_timesteps))
        n_samples = len(valid_indices)

        n_train = int(n_samples * TRAIN_RATIO)
        n_val = int(n_samples * VAL_RATIO)

        train_indices = valid_indices[:n_train]
        val_indices = valid_indices[n_train:n_train + n_val]
        test_indices = valid_indices[n_train + n_val:]

        logger.info(f"Dataset Split - Total: {n_samples} | Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

        self.train_dataset = EthiopiaDroughtDataset(self.nc_file_path, train_indices)
        self.val_dataset = EthiopiaDroughtDataset(self.nc_file_path, val_indices)
        self.test_dataset = EthiopiaDroughtDataset(self.nc_file_path, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
