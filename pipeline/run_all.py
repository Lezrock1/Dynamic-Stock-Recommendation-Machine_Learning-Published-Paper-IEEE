"""One-shot entrypoint: refresh data, retrain, and report - re-run this any time
to extend the recommendations to the current date.

Usage (from repo root, with the venv activated):
    source .venv/bin/activate
    python pipeline/run_all.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = Path(__file__).resolve().parent


def run(script: str, *args: str):
    print(f"\n=== {script} {' '.join(args)} ===")
    subprocess.run([sys.executable, str(PIPELINE / script), *args], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    run("build_dataset.py")
    run("run_model.py")
    run("report.py")
    run("paper_performance.py")
