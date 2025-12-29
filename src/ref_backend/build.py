import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=repo_root)


if __name__ == "__main__":
    main()
