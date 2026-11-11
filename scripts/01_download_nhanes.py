"""Download the NHANES August 2021 - August 2023 files this analysis uses.

Saves each SAS transport file unchanged in data/raw/ with a manifest of
sizes and checksums, so the analysis can be re-run offline and anyone can
confirm they are working from the same files.

    python scripts/01_download_nhanes.py

Source: National Center for Health Statistics, National Health and
Nutrition Examination Survey. https://wwwn.cdc.gov/nchs/nhanes/
Uses only the Python standard library.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"

FILES = {
    "DEMO_L": "Demographics, sample weights and survey design variables",
    "DIQ_L": "Diabetes questionnaire: has a doctor told you that you have diabetes?",
    "GHB_L": "Glycohemoglobin (HbA1c), measured from a blood sample",
    "BMX_L": "Body measures: weight, height, BMI, waist",
    "BPXO_L": "Blood pressure, measured three times",
    "BPQ_L": "Blood pressure and cholesterol questionnaire",
    "HIQ_L": "Health insurance",
    "HUQ_L": "Hospital utilisation and access to care",
}


def download(name: str) -> bytes:
    url = f"{BASE}{name}.xpt"
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "portfolio-analysis/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - any network failure is retried
            print(f"  retry {attempt + 1} for {name}: {type(error).__name__}", flush=True)
            time.sleep(4 + 3 * attempt)
    raise SystemExit(f"Could not download {url}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "NHANES August 2021 - August 2023, National Center for Health Statistics",
                "base_url": BASE, "files": {}}
    for name, description in FILES.items():
        content = download(name)
        if not content.startswith(b"HEADER RECORD"):
            raise SystemExit(f"{name}: the download is not a SAS transport file")
        (RAW / f"{name}.xpt").write_bytes(content)
        manifest["files"][f"{name}.xpt"] = {"description": description, "bytes": len(content),
                                            "sha256": hashlib.sha256(content).hexdigest()}
        print(f"{name:<8} {len(content) / 1_000_000:>5.1f} MB  {description}")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
