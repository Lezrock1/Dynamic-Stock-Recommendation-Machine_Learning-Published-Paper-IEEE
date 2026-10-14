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
    print(
        "\nDaily performance reports are a separate step. Provide raw close prices, "
        "a complete corporate-actions file, three fixed target sets, and a verified "
        "total-return benchmark to pipeline/paper_performance.py; the legacy adjusted "
        "price/SPX archives are not a validated substitute."
    )
