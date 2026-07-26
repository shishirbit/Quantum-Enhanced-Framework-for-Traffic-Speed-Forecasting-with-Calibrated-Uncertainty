"""Mount Google Drive when executed inside Google Colab."""
from __future__ import annotations

from pathlib import Path


def main():
    try:
        from google.colab import drive
    except ImportError:
        print("Not running in Google Colab; no Drive mount performed.")
        return
    mount = "/content/drive"
    drive.mount(mount)
    root = Path(mount) / "MyDrive" / "QUARTS"
    for folder in ("data", "checkpoints", "logs", "figures", "tables", "predictions"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    print(f"Persistent QUARTS root: {root}")


if __name__ == "__main__":
    main()

