from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
FILES = ("classification.csv.gz", "coordinates.csv.gz")


def download(url: str, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed with HTTP {exc.code}. If the public static URL is unavailable, "
            "download the same files from https://codex.flywire.ai/api/download?dataset=fafb "
            f"and place them in {destination.parent}."
        ) from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # Fail early on HTML/error pages saved with a .gz name.
    try:
        with gzip.open(tmp, "rt", encoding="utf-8-sig") as handle:
            header = handle.readline().strip()
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded file is not valid gzip data: {url}") from exc

    if "root_id" not in header:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Unexpected CSV header in {url}: {header[:160]}")

    tmp.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MiB)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_out = repo_root / "data" / "flywire" / "fafb783"

    parser = argparse.ArgumentParser(description="Download the small FAFB v783 files needed by the 3D soma viewer.")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = args.out / filename
        if destination.exists() and not args.force:
            print(f"Already present: {destination}")
            continue
        download(f"{BASE_URL}/{filename}", destination)

    print("\nFAFB v783 soma layout files are ready.")
    print(f"Data directory: {args.out.resolve()}")
    print("Restart the DrosoMath backend; /api/layout will switch from mock to real FlyWire coordinates automatically.")


if __name__ == "__main__":
    main()
