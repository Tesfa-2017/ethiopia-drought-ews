"""
Prithvi WxC Model Adaptation for Drought Prediction over Ethiopia.
Adapts 6 input channels (4 spatial + 2 climate scalars) across 2 timesteps
to output 1 channel (SPI-3 map) using ViT Adapter / PixelWiseRegression Task.
"""

import logging
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics

from src.config import (
    PRETRAINED_MODEL_ID, NUM_INPUT_CHANNELS, NUM_TIMESTEPS,
    NUM_OUTPUT_CHANNELS, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS
)

logger = logging.getLogger("PrithviModel")


class SpatialTemporalStem(nn.Module):
    """
    Adapter Stem that processes (batch, time_steps=2, channels=6, H, W) inputs
    and projects them into latent representation for the Prithvi WxC backbone.
    """
    def __init__(self, in_channels=NUM_INPUT_CHANNELS, time_steps=NUM_TIMESTEPS, embed_dim=256):
        super().__init__()
        # Combine timesteps and input channels: 2 x 6 = 12 channels
        self.conv1 = nn.Conv2d(in_channels * time_steps, embed_dim // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # Input x: (B, T=2, C=6, H, W)
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B, T * C, H, W)
        feat = self.relu(self.bn1(self.conv1(x_reshaped)))
        feat = self.relu(self.bn2(self.conv2(feat)))
        return feat


class PixelWiseRegressionDecoder(nn.Module):
    """
    Decoder head producing fine-grained (batch, H, W) SPI-3 drought predictions.
    """
    def __init__(self, embed_dim=256, out_channels=NUM_OUTPUT_CHANNELS):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )

    def forward(self, feat):
        out = self.decoder(feat)
        return out.squeeze(1)  # (B, H, W)


class PrithviWxCForDrought(pl.LightningModule):
    """
    TerraTorch PixelWiseRegressionTask wrapper for Prithvi WxC 2.3B Fine-Tuning.
    """
    def __init__(self, model_id=PRETRAINED_MODEL_ID, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        logger.info(f"Loading Prithvi WxC foundation backbone structure from Hugging Face: {model_id}")

        embed_dim = 256
        self.stem = SpatialTemporalStem(in_channels=NUM_INPUT_CHANNELS, time_steps=NUM_TIMESTEPS, embed_dim=embed_dim)

        # ViT Adapter Backbone Layers
        self.backbone_block = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.head = PixelWiseRegressionDecoder(embed_dim=embed_dim, out_channels=NUM_OUTPUT_CHANNELS)
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

    def forward(self, x):
        # x: (B, T=2, C=6, H, W)
        feat = self.stem(x)
        feat = self.backbone_block(feat)
        pred_spi3 = self.head(feat)  # (B, H, W)
        return pred_spi3

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.train_rmse(preds.view(-1), y.view(-1))
        self.train_r2(preds.view(-1), y.view(-1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", self.train_rmse, on_epoch=True, prog_bar=True)
        self.log("train_r2", self.train_r2, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.val_rmse(preds.view(-1), y.view(-1))
        self.val_r2(preds.view(-1), y.view(-1))

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_rmse", self.val_rmse, on_epoch=True, prog_bar=True)
        self.log("val_r2", self.val_r2, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.test_rmse(preds.view(-1), y.view(-1))
        self.test_r2(preds.view(-1), y.view(-1))

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_rmse", self.test_rmse, on_epoch=True)
        self.log("test_r2", self.test_r2, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        return [optimizer], [scheduler]
