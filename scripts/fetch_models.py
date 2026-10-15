#!/usr/bin/env python3
"""Fetch pinned glTF sample models for waterui-model3d tests and examples.

Downloads single-file GLB assets listed in ``scripts/models.json`` from
KhronosGroup/glTF-Sample-Assets. Every URL is built from the exact commit
pinned in the manifest, so downloads never track a moving branch.

Integrity: a downloaded file is SHA-256 verified *before* it is installed
into the output directory (write to temp, verify, rename). A file that
already exists in the output directory is reused only when it verifies
against the manifest hash; a hash mismatch is a hard failure so a tampered
or stale cache is never silently replaced.

Licensing/attribution: each model's copyright, SPDX license id and
attribution string are recorded in ``scripts/models.json`` (sourced from the
model READMEs upstream). Review them before redistributing any asset —
e.g. DamagedHelmet derives from a CC BY-NC 4.0 work. Downloaded assets are
gitignored and must never be committed.

Usage:
    python3 scripts/fetch_models.py                      # fetch all models
    python3 scripts/fetch_models.py --model Fox          # fetch a subset
    python3 scripts/fetch_models.py --model Fox --model BoxTextured
    python3 scripts/fetch_models.py --output-dir /tmp/models
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MAX_WORKERS = 4
CHUNK_SIZE = 1 << 20  # 1 MiB

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = SCRIPT_DIR / "models.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "models"


class FetchError(Exception):
    pass


@dataclass(frozen=True)
class Model:
    name: str
    file: str
    path: str
    sha256: str
    bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> tuple[dict, list[Model]]:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    models = [
        Model(
            name=entry["name"],
            file=entry["file"],
            path=entry["path"],
            sha256=entry["sha256"],
            bytes=entry["bytes"],
        )
        for entry in manifest["models"]
    ]
    return manifest["source"], models


def model_url(source: dict, model: Model) -> str:
    # URL always contains the exact pinned commit from the manifest.
    return f"{source['raw_base']}/{source['commit']}/{model.path}"


def fetch_model(source: dict, model: Model, output_dir: Path) -> str:
    dest = output_dir / model.file
    if dest.exists():
        actual = sha256_file(dest)
        if actual == model.sha256:
            return f"{model.name}: cached ({model.bytes} bytes)"
        raise FetchError(
            f"{model.name}: existing file {dest} fails sha256 verification "
            f"(expected {model.sha256}, got {actual}); refusing to reuse a "
            f"tampered or stale file — delete it and re-run."
        )

    fd, tmp_name = tempfile.mkstemp(dir=output_dir, prefix=f".{model.file}.")
    tmp = Path(tmp_name)
    try:
        url = model_url(source, model)
        with os.fdopen(fd, "wb") as handle:
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, handle, CHUNK_SIZE)
        actual = sha256_file(tmp)
        if actual != model.sha256:
            raise FetchError(
                f"{model.name}: downloaded bytes fail sha256 verification "
                f"(expected {model.sha256}, got {actual}) from {url}"
            )
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return f"{model.name}: downloaded {model.bytes} bytes (sha256 {model.sha256})"


def parse_args(models: list[Model]) -> argparse.Namespace:
    names = sorted(m.name for m in models)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory to install GLB files into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=names,
        metavar="NAME",
        help="fetch only this model; repeat to select several "
        f"(default: all). Choices: {', '.join(names)}",
    )
    return parser.parse_args()


def main() -> int:
    source, models = load_manifest()
    args = parse_args(models)
    selected = (
        [m for m in models if m.name in set(args.model)] if args.model else models
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = {
            pool.submit(fetch_model, source, model, output_dir): model
            for model in selected
        }
        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            except FetchError as error:
                failures.append(str(error))
                print(f"error: {error}", file=sys.stderr)

    if failures:
        print(f"{len(failures)} model(s) failed verification", file=sys.stderr)
        return 1
    total = sum(m.bytes for m in selected)
    print(f"{len(selected)} model(s) ready in {output_dir} ({total} bytes total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
