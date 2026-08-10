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
    # Step 1: Verify CUDA GPU Hardware Acceleration
    import torch as _torch
    print("CUDA Available:", _torch.cuda.is_available())
    if _torch.cuda.is_available():
        print("GPU Device Name:", _torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. Ensure GPU accelerator is enabled in Runtime settings.")
    return


@app.cell
def _():
    # Step 2: Auto-sync repository modules (src/ and data/) in MoLab environment
    import os as _os
    import sys as _sys
    import subprocess as _subprocess
    import shutil as _shutil

    _venv_site = "/tmp/uv-venv/lib/python3.13/site-packages"
    if _os.path.exists(_venv_site):
        if _venv_site in _sys.path:
            _sys.path.remove(_venv_site)
        _sys.path.insert(0, _venv_site)

    if not _os.path.exists("src"):
        print("Syncing project files from GitHub into MoLab environment...")
        _subprocess.run(["git", "clone", "https://github.com/Tesfa-2017/ethiopia-drought-ews.git", "_repo_tmp"], check=True)
        if _os.path.exists("_repo_tmp/src"):
            _shutil.copytree("_repo_tmp/src", "src", dirs_exist_ok=True)
        if _os.path.exists("_repo_tmp/data"):
            _shutil.copytree("_repo_tmp/data", "data", dirs_exist_ok=True)
        _shutil.rmtree("_repo_tmp")
        print("Project modules (src/ & data/) synced successfully!")
    else:
        print("src/ module directory verified.")

    _curr_dir = _os.path.abspath(".")
    if _curr_dir not in _sys.path:
        _sys.path.insert(0, _curr_dir)
    return


@app.cell
def _():
    # Step 3: Run Full Pipeline Training on GPU
    import os as _os
    import sys as _sys

    _venv_site = "/tmp/uv-venv/lib/python3.13/site-packages"
    if _os.path.exists(_venv_site):
        if _venv_site in _sys.path:
            _sys.path.remove(_venv_site)
        _sys.path.insert(0, _venv_site)

    _sys_site = "/usr/local/lib/python3.13/site-packages"
    if _os.path.exists(_sys_site):
        if _sys_site in _sys.path:
            _sys.path.remove(_sys_site)
        _sys.path.append(_sys_site)

    _curr_dir = _os.path.abspath(".")
    if _curr_dir not in _sys.path:
        _sys.path.insert(0, _curr_dir)

    _os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    from src.train import run_training

    print("Starting Prithvi WxC Fine-Tuning Training Loop...")
    best_ckpt_path = run_training()
    print(f"Training completed! Best checkpoint: {best_ckpt_path}")
    return (best_ckpt_path,)


@app.cell
def _(best_ckpt_path):
    # Step 4: Run EDRMC Zonal Risk Evaluation & Inference
    import os as _os
    import sys as _sys

    _venv_site = "/tmp/uv-venv/lib/python3.13/site-packages"
    if _os.path.exists(_venv_site):
        if _venv_site in _sys.path:
            _sys.path.remove(_venv_site)
        _sys.path.insert(0, _venv_site)

    _curr_dir = _os.path.abspath(".")
    if _curr_dir not in _sys.path:
        _sys.path.insert(0, _curr_dir)

    from src.inference import run_inference_and_edrmc_eval

    print("Evaluating EDRMC Zonal Drought Risk Activation Status...")
    df_results = run_inference_and_edrmc_eval()
    return (df_results,)


if __name__ == "__main__":
    app.run()
