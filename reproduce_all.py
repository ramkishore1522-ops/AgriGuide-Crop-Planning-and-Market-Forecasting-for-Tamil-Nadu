import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        subprocess.run([sys.executable, f"scripts/{script_name}"], check=True)
        print(f"[SUCCESS] {script_name} completed in {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    print("STARTING COMPLETE REPRODUCIBILITY PIPELINE...")
    
    scripts = [
        "error_analysis.py",
        "per_commodity_pipeline.py",
        "hybrid_model.py",
        "conformal_prediction.py",
        "granger_causality.py",
        "generate_paper_tables.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\n" + "="*60)
    print("PIPELINE COMPLETE. ALL RESULTS GENERATED SUCCESSFULLY.")
    print("Check the 'reports/' and 'visualizations/' directories.")

if __name__ == "__main__":
    main()
