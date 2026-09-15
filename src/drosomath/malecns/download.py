from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
DEFAULT_DATA_DIR = Path("data/malecns_v1")


def path_for(key: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    if key not in FILES:
        raise KeyError(f"unknown MaleCNS file key: {key}")
    return Path(data_dir) / FILES[key]


def missing_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[str, ...]:
    root = Path(data_dir)
    return tuple(
        key
        for key, filename in FILES.items()
        if not (root / filename).is_file() or (root / filename).stat().st_size == 0
    )


def _download_with_curl(url: str, destination: Path) -> None:
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl is None:
        raise FileNotFoundError("curl executable not found")
    subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "-C",
            "-",
            "-o",
            str(destination),
            url,
        ],
        check=True,
    )


def _download_with_urllib(url: str, destination: Path) -> None:
    # Fallback path for systems without curl. It does not resume partial files.
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(destination)


def download_malecns(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    keys: tuple[str, ...] = ("annotations", "neurotransmitters", "weights"),
) -> tuple[Path, ...]:
    """Download the three public MaleCNS v1.0 flat-connectome tables we need.

    Existing non-empty files are kept. curl is preferred because the 1.1 GB
    weights table can be resumed; urllib is a portable fallback.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for key in keys:
        destination = path_for(key, root)
        if destination.is_file() and destination.stat().st_size > 0:
            downloaded.append(destination)
            continue

        url = BASE_URL + FILES[key]
        try:
            _download_with_curl(url, destination)
        except FileNotFoundError:
            _download_with_urllib(url, destination)
        downloaded.append(destination)

    return tuple(downloaded)
