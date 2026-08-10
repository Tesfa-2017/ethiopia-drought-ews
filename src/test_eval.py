import sys
import os
import traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
logging.basicConfig(level=logging.INFO)

from src.inference import run_inference_and_edrmc_eval

if __name__ == "__main__":
    try:
        run_inference_and_edrmc_eval()
    except Exception as e:
        traceback.print_exc()
