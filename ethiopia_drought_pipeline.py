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
    # Step 2.5: Auto-clone repository modules (src/ & data/) if running in cloud MoLab environment
    import os
    import subprocess
    import shutil

    if not os.path.exists("src"):
        print("Syncing project files from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/Tesfa-2017/ethiopia-drought-ews.git", "_repo_tmp"], check=True)
        if os.path.exists("_repo_tmp/src"):
            shutil.copytree("_repo_tmp/src", "src", dirs_exist_ok=True)
        if os.path.exists("_repo_tmp/data"):
            shutil.copytree("_repo_tmp/data", "data", dirs_exist_ok=True)
        shutil.rmtree("_repo_tmp")
        print("Project modules (src/ & data/) synced successfully!")
    else:
        print("src/ module directory verified.")
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
