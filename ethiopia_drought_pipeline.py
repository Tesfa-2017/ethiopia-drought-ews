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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prithvi WxC Drought Prediction Pipeline for Ethiopia (GPU Accelerated)
    This notebook trains the IBM/NASA Prithvi WxC foundation model fine-tuned for SPI-3 drought prediction over Ethiopia on a GPU (NVIDIA RTX / T4 / A100 / L4).
    """)
    return


@app.cell
def _():
    # Core imports cell
    import os
    import sys
    import subprocess
    import shutil
    import torch
    return os, sys, subprocess, shutil, torch


@app.cell
def _(torch):
    # Step 1: Verify CUDA GPU Hardware Acceleration
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. Ensure GPU accelerator is enabled in Runtime settings.")
    return


@app.cell
def _(os, sys, subprocess, shutil):
    # Step 2: Auto-sync repository modules and ensure matching torchvision/torch installation
    try:
        import torchvision
        import torchvision.ops
    except Exception as e:
        print(f"Aligning torch & torchvision versions for Python 3.13... ({e})")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "torch", "torchvision"], check=True)

    if not os.path.exists("src"):
        print("Syncing project files from GitHub into MoLab environment...")
        subprocess.run(["git", "clone", "https://github.com/Tesfa-2017/ethiopia-drought-ews.git", "_repo_tmp"], check=True)
        if os.path.exists("_repo_tmp/src"):
            shutil.copytree("_repo_tmp/src", "src", dirs_exist_ok=True)
        if os.path.exists("_repo_tmp/data"):
            shutil.copytree("_repo_tmp/data", "data", dirs_exist_ok=True)
        shutil.rmtree("_repo_tmp")
        print("Project modules (src/ & data/) synced successfully!")
    else:
        print("src/ module directory verified.")

    curr_dir = os.path.abspath(".")
    if curr_dir not in sys.path:
        sys.path.insert(0, curr_dir)
    return


@app.cell
def _(os):
    # Step 3: Run Full Pipeline Training on GPU
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    from src.train import run_training

    print("Starting Prithvi WxC Fine-Tuning Training Loop...")
    best_ckpt_path = run_training()
    print(f"Training completed! Best checkpoint: {best_ckpt_path}")
    return (best_ckpt_path,)


@app.cell
def _(best_ckpt_path):
    # Step 4: Run EDRMC Zonal Risk Evaluation & Inference
    from src.inference import run_inference_and_edrmc_eval

    print("Evaluating EDRMC Zonal Drought Risk Activation Status...")
    df_results = run_inference_and_edrmc_eval()
    return (df_results,)


if __name__ == "__main__":
    app.run()
