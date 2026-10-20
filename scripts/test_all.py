"""Run the Python/SQL checks and frontend lint, build, and utility tests."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> int:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main() -> int:
    if run([sys.executable, "-m", "pytest", "-q"], ROOT):
        return 1
    npm = shutil.which("npm")
    if not npm:
        print("npm is required for frontend checks. Install Node.js 20 or 22 and run npm ci in churn-dashboard.", file=sys.stderr)
        return 2
    return run([npm, "test"], ROOT / "churn-dashboard")


if __name__ == "__main__":
    raise SystemExit(main())
