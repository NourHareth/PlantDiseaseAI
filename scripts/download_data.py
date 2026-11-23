"""Download the tomato subsets of PlantVillage and PlantDoc.

    python -m scripts.download_data              # both datasets
    python -m scripts.download_data plantdoc     # just one

Only the tomato folders are fetched (about 0.6 GB + 0.3 GB) into
``<workdir>/datasets/raw/`` (see ``local_paths.json`` / PLANTDISEASE_WORKDIR).

How: a blob-less ``git clone`` provides the exact file list (a few MB), then the
files are downloaded in parallel from raw.githubusercontent.com. This is much
faster than a sparse checkout, whose server-side packing of ~36,000 files can
stall for many minutes. The download is resumable: existing files are skipped.

Sources
-------
* PlantVillage (Hughes & Salathe, 2015): lab photos of single leaves, plus
  background-removed ("segmented") versions used here as leaf masks.
  https://github.com/spMohanty/PlantVillage-Dataset
* PlantDoc (Singh et al., 2020, CC BY 4.0): real-world photos from the web.
  https://github.com/pratikkayal/PlantDoc-Dataset
"""

from __future__ import annotations

import argparse
import http.client
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.download_data)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/download_data.py)

from plantdisease import config

SOURCES = {
    "plantvillage": {
        "repo": "spMohanty/PlantVillage-Dataset",
        "branch": "master",
        "prefixes": ["raw/color/Tomato___", "raw/segmented/Tomato___"],
    },
    "plantdoc": {
        "repo": "pratikkayal/PlantDoc-Dataset",
        "branch": "master",
        "prefixes": ["train/Tomato", "test/Tomato", "LICENSE.txt", "README.md"],
    },
}
USER_AGENT = "plantdisease-ai-dataset-downloader"
RAW_HOST = "raw.githubusercontent.com"


def list_files(name: str, destination: Path) -> list[str]:
    """Paths in the repository that match the source's prefixes (trees only, no blobs)."""
    source = SOURCES[name]
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", "--branch",
             source["branch"], f"https://github.com/{source['repo']}.git", str(destination)],
            check=True,
        )
    listing = subprocess.run(
        ["git", "-C", str(destination), "ls-tree", "-r", "-z", "--name-only", "HEAD"],
        check=True, capture_output=True,
    ).stdout.decode("utf-8")
    return [p for p in listing.split("\0") if p and any(p.startswith(prefix) for prefix in source["prefixes"])]


_thread_state = threading.local()


def _connection() -> http.client.HTTPSConnection:
    """One keep-alive connection per worker thread (a new TLS handshake per file is ~3x slower)."""
    conn = getattr(_thread_state, "conn", None)
    if conn is None:
        conn = http.client.HTTPSConnection(RAW_HOST, timeout=60)
        _thread_state.conn = conn
    return conn


def _reset_connection() -> None:
    conn = getattr(_thread_state, "conn", None)
    if conn is not None:
        conn.close()
    _thread_state.conn = None


_ILLEGAL = re.compile(r'[<>:"|?*\\]')


def local_path(destination: Path, repo_path: str) -> Path:
    """Where a repository file is stored locally.

    Some PlantDoc file names come from web URLs and contain characters Windows
    forbids (``latest?cb=2010.jpg``); those are replaced by ``_``.
    """
    parts = [_ILLEGAL.sub("_", part).rstrip(" .") or "_" for part in repo_path.split("/")]
    return destination.joinpath(*parts)


def fetch_one(url_path: str, target: Path, retries: int = 6) -> int:
    if target.exists() and target.stat().st_size > 0:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            conn = _connection()
            conn.request("GET", url_path, headers={"User-Agent": USER_AGENT})
            response = conn.getresponse()
            data = response.read()
            if response.status in (429, 403, 500, 502, 503, 504):
                raise ConnectionError(f"HTTP {response.status}")
            if response.status != 200:
                raise RuntimeError(f"https://{RAW_HOST}{url_path}: HTTP {response.status}")
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(data)
            partial.replace(target)
            return len(data)
        except (http.client.HTTPException, OSError, ConnectionError) as exc:
            _reset_connection()
            if attempt == retries - 1:
                raise RuntimeError(f"https://{RAW_HOST}{url_path}: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))
    return 0


def download(name: str, destination: Path, workers: int) -> None:
    source = SOURCES[name]
    paths = list_files(name, destination)
    print(f"{name}: {len(paths)} files to check", flush=True)
    base = f"/{source['repo']}/{source['branch']}/"
    done = downloaded = failed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, base + quote(p), local_path(destination, p)): p for p in paths}
        for future in as_completed(futures):
            done += 1
            try:
                downloaded += future.result()
            except RuntimeError as exc:
                failed += 1
                print(f"  failed: {exc}", flush=True)
            if done % 2000 == 0 or done == len(paths):
                rate = downloaded / 1e6 / max(time.time() - started, 1e-6)
                print(f"  {done}/{len(paths)} files, {downloaded / 1e6:.0f} MB new ({rate:.1f} MB/s)", flush=True)
    if failed:
        raise SystemExit(f"{name}: {failed} files failed; run the command again to retry them.")
    print(f"{name}: complete in {destination}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help=f"any of: {', '.join(SOURCES)} (default: all)")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args(argv)
    unknown = set(args.datasets) - set(SOURCES)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for name in args.datasets or SOURCES:
        download(name, args.raw_dir / name, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
