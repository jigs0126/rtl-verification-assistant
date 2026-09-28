"""
Entry point: launches the Streamlit UI (app/main.py).

Usage:
    python run.py
    # equivalent to: streamlit run app/main.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()
