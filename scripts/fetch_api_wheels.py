#!/usr/bin/env python3
"""Fetch pinned API wheels directly from official PyPI release metadata.

Fallback for unreliable package-index access. No packages are installed.
Each wheel is selected for this interpreter and verified against PyPI's SHA256.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import threading
import time
import urllib.request
from urllib.parse import urlsplit

try:
    from packaging.specifiers import SpecifierSet
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ImportError:
    # A normal venv includes pip; avoid requiring another download first.
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import Version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=root / "requirements-api.txt")
    parser.add_argument("--wheelhouse", type=Path, default=root / ".local/api-wheelhouse")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--timeout", type=float, default=45, help="Idle socket timeout in seconds")
    parser.add_argument("--retries", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    requirements = []
    for line in args.requirements.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            parser.error("This helper accepts only exact name==version pins, without extras or URLs.")
        requirements.append(match.groups())
    args.wheelhouse.mkdir(parents=True, exist_ok=True)
    tags = {tag: index for index, tag in enumerate(sys_tags())}
    connections = threading.Semaphore(args.workers)

    def release(name, version):
        for attempt in range(args.retries):
            try:
                with connections, urllib.request.urlopen(
                    f"https://pypi.org/pypi/{name}/{version}/json", timeout=args.timeout
                ) as response:
                    data = json.load(response)
                if canonicalize_name(data["info"]["name"]) != canonicalize_name(name) or Version(data["info"]["version"]) != Version(version):
                    raise ValueError("PyPI returned mismatched release metadata")
                return data
            except Exception:
                if attempt + 1 == args.retries:
                    raise
                time.sleep(1)

    def transfer(url, path, start, end, ranged):
        size = end - start + 1
        for attempt in range(args.retries):
            offset = path.stat().st_size if path.exists() else 0
            if offset == size:
                return path
            if offset > size:
                raise ValueError(f"Oversized partial file: {path.name}")
            headers = {"Accept-Encoding": "identity"}
            if ranged or offset:
                headers["Range"] = f"bytes={start + offset}-{end}"
            try:
                request = urllib.request.Request(url, headers=headers)
                with connections, urllib.request.urlopen(request, timeout=args.timeout) as response:
                    if ranged or offset:
                        expected = f"bytes {start + offset}-{end}/"
                        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                            raise RuntimeError("Server did not return the requested byte range")
                    with path.open("ab") as output:
                        while offset < size:
                            block = response.read(min(65536, size - offset))
                            if not block:
                                raise RuntimeError("Truncated wheel download")
                            output.write(block)
                            offset += len(block)
                return path
            except Exception:
                if attempt + 1 == args.retries:
                    raise
                time.sleep(1)

    def fetch(item):
        name, version = item
        data = release(name, version)
        candidates = []
        for entry in data["urls"]:
            if entry["packagetype"] != "bdist_wheel" or entry.get("yanked"):
                continue
            if entry.get("requires_python") and not SpecifierSet(entry["requires_python"]).contains(platform.python_version()):
                continue
            wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(entry["filename"])
            common = wheel_tags & set(tags)
            if canonicalize_name(wheel_name) == canonicalize_name(name) and wheel_version == Version(version) and common:
                candidates.append((min(tags[tag] for tag in common), entry))
        if not candidates:
            raise RuntimeError(f"No compatible wheel for {name}=={version} on this interpreter")
        entry = min(candidates, key=lambda item: item[0])[1]
        filename, digest, size = entry["filename"], entry["digests"]["sha256"], entry["size"]
        url = urlsplit(entry["url"])
        if Path(filename).name != filename or not re.fullmatch(r"[0-9a-f]{64}", digest) or size <= 0:
            raise ValueError("Invalid wheel metadata")
        if url.scheme != "https" or url.hostname != "files.pythonhosted.org" or url.username:
            raise ValueError("Wheel URL is not an official PyPI artifact URL")
        target = args.wheelhouse / filename
        if not target.exists() or target.stat().st_size != size or sha256(target) != digest:
            parts = args.wheelhouse / ".parts" / filename
            parts.mkdir(parents=True, exist_ok=True)
            count = args.workers if size >= 8 * 1024 * 1024 else 1
            chunk = (size + count - 1) // count
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [pool.submit(transfer, entry["url"], parts / str(i), i * chunk,
                                       min(size, (i + 1) * chunk) - 1, count > 1) for i in range(count)]
                paths = [future.result() for future in futures]
            temporary = target.with_suffix(".part")
            with temporary.open("wb") as output:
                for part in paths:
                    with part.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            output.write(block)
            if temporary.stat().st_size != size or sha256(temporary) != digest:
                raise RuntimeError(f"SHA256/size mismatch for {filename}; no wheel was accepted")
            temporary.replace(target)
        print(f"Verified {filename}", flush=True)
        return {"name": name, "version": version, "filename": filename,
                "url": entry["url"], "sha256": digest, "size": size}

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, item): item for item in requirements}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"requirement": "==".join(futures[future]), "error": str(exc)})
                print(f"Failed {failures[-1]['requirement']}: {exc}", flush=True)
    manifest = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "wheels": sorted(records, key=lambda item: item["name"]), "failures": failures}
    (args.wheelhouse / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished: {len(records)} verified wheels; {len(failures)} failures.", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
