"""
Master Pipeline Orchestrator
============================
Author: [Your Name/Institution]
Description: This master script runs the entire machine learning pipeline sequentially.
It ensures reproducibility of all data processing, modeling, and evaluation steps
required for the publication.

Outputs:
  - Updates all files in `reports/` and `visualizations/` directories.
"""

import subprocess
import sys
import time
import logging
from typing import List

# ── Configure Professional Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_script(script_name: str) -> None:
    """
    Executes a given python script via subprocess and logs its runtime.

    Args:
        script_name (str): The relative path to the script inside the 'scripts/' directory.

    Raises:
        SystemExit: If the script fails, it exits the pipeline to prevent cascading errors.
    """
    logging.info("=" * 60)
    logging.info(f"RUNNING: {script_name}")
    logging.info("=" * 60)

    start_time = time.time()
    try:
        # Execute the script in a separate process
        subprocess.run([sys.executable, f"scripts/{script_name}"], check=True)
        elapsed = time.time() - start_time
        logging.info(f"[SUCCESS] {script_name} completed in {elapsed:.1f}s")
    except subprocess.CalledProcessError as e:
        logging.error(f"[FAILED] {script_name} exited with code {e.returncode}")
        sys.exit(1)


def main() -> None:
    """
    Main orchestration function to run the machine learning pipeline sequentially.
    """
    logging.info("STARTING COMPLETE REPRODUCIBILITY PIPELINE...")

    scripts: List[str] = [
        "02_modeling/tn_no_lag_model.py",
        "03_evaluation/error_analysis.py",
        "02_modeling/per_commodity_pipeline.py",
        "03_evaluation/conformal_prediction.py",
        "03_evaluation/granger_causality.py",
        "03_evaluation/generate_paper_tables.py",
    ]

    for script in scripts:
        run_script(script)

    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE. ALL RESULTS GENERATED SUCCESSFULLY.")
    logging.info("Please check the 'reports/' and 'visualizations/' directories.")


if __name__ == "__main__":
    main()
