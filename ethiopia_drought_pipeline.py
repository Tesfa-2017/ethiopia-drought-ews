# /// script
# dependencies = ["dask", "earthengine-api", "geedim", "geemap", "huggingface-hub", "matplotlib", "mlflow", "netcdf4", "pandas", "pytorch-lightning", "rioxarray", "scikit-learn", "scipy", "terratorch", "torchgeo", "torchmetrics", "xarray"]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prithvi WxC Drought Prediction Pipeline for Ethiopia (GPU Accelerated)
    This notebook trains the Prithvi WxC foundation model fine-tuned for SPI-3 drought prediction over Ethiopia on a GPU (NVIDIA RTX Pro 6000 / T4 / A100).
    """)
    return


@app.cell
def _():
    # Step 1: Verify CUDA GPU Acceleration
    import torch
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. Ensure GPU accelerator is enabled in Runtime settings.")
    return


@app.cell
def _():
    # Step 2: Install Required Dependencies
    # packages added via marimo's package management: terratorch torchgeo earthengine-api geemap geedim xarray rioxarray netcdf4 huggingface_hub mlflow pytorch-lightning scikit-learn pandas scipy dask matplotlib torchmetrics !pip install -q terratorch torchgeo earthengine-api geemap geedim xarray rioxarray netcdf4 huggingface_hub mlflow pytorch-lightning scikit-learn pandas scipy dask matplotlib torchmetrics
    return


@app.cell
def _(subprocess):
    # Step 3: Run Full Pipeline Training on GPU
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Run fine-tuning training loop
    #! python src/train.py
    subprocess.call(['python', 'src/train.py'])
    return


@app.cell
def _(subprocess):
    # Step 4: Run EDRMC Zonal Risk Evaluation & Inference
    #! python src/inference.py
    subprocess.call(['python', 'src/inference.py'])
    return


if __name__ == "__main__":
    app.run()
