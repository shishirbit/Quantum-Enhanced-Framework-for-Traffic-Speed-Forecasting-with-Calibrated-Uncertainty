"""Download the canonical files linked by the official DCRNN repository.

The source is the Google Drive folder linked in liyaguang/DCRNN README. This
script does not rehost data and does not grant redistribution rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gdown
import requests

OFFICIAL_FOLDER = "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
ADJACENCY_URLS = {
    "adj_mx.pkl": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl",
    "adj_mx_bay.pkl": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/dcrnn")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(OFFICIAL_FOLDER, output=str(output), quiet=False, use_cookies=False)
    for filename, url in ADJACENCY_URLS.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        (output / filename).write_bytes(response.content)
    files = [p for p in output.rglob("*") if p.is_file()]
    provenance = {
        "source": OFFICIAL_FOLDER,
        "linked_from": "https://github.com/liyaguang/DCRNN#data-preparation",
        "adjacency_sources": ADJACENCY_URLS,
        "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
