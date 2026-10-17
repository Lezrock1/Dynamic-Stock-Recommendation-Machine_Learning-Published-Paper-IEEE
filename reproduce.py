"""Portable, offline-first entry point; all default paths are checkout-relative."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEMO_DB = "state/reproduction/synthetic.sqlite3"


def dependency_errors(root: Path = ROOT) -> list[str]:
    errors = []
    for line in (root / "pipeline/requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==", 1)
        try:
            installed = version(name)
        except PackageNotFoundError:
            errors.append(f"{name}: missing (required {expected})")
        else:
            if installed != expected:
                errors.append(f"{name}: installed {installed}, required {expected}")
    return errors


def command(action: str, host: str = "127.0.0.1", port: int = 8765) -> list[str]:
    if action == "test":
        return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    if action == "demo":
        return [sys.executable, "-m", "pipeline.paper_portfolio",
                "--env", "synthetic", "--db", DEMO_DB, "demo"]
    if action == "dashboard":
        return [sys.executable, "-m", "pipeline.paper_portfolio",
                "--env", "synthetic", "--db", DEMO_DB, "dashboard",
                "--host", host, "--port", str(port)]
    raise ValueError(f"Unknown action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "test", "demo", "dashboard"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("Native Windows is not supported: POSIX process locks require Linux/WSL2 or macOS")
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required; the reference interpreter is Python 3.12.2")
    errors = dependency_errors()
    if errors:
        print("Dependency check failed:\n" + "\n".join(errors), file=sys.stderr)
        print("Install from the checkout: python -m pip install -r pipeline/requirements.txt",
              file=sys.stderr)
        return 1
    if args.action == "check":
        print("Pinned direct dependencies match. No private state or network required.")
        return 0
    if args.action == "dashboard" and not (ROOT / DEMO_DB).is_file():
        parser.error("No synthetic demo database. Run: python reproduce.py demo")
    return subprocess.run(command(args.action, args.host, args.port), cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
