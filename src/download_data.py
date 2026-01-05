"""Download or copy the Heart Disease dataset CSV to a target path.
Simple convenience wrapper so pipelines can fetch data reproducibly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import requests

DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("http"):
        r = requests.get(url)
        r.raise_for_status()
        out.write_bytes(r.content)
    else:
        # treat as local file
        shutil.copy(url, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    download(args.url, Path(args.output))


if __name__ == "__main__":
    main()
