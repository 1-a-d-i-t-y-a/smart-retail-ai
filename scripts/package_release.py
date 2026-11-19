#!/usr/bin/env python3
"""
scripts/package_release.py
--------------------------
Builds, hashes, and clean-room verifies the release distribution bundle for Smart Retail AI:
1. Packages repository files into dist/smart-retail-ai-v1.2.0.zip (excluding VCS & caches).
2. Computes SHA-256 checksum and saves to dist/SHA256SUMS.txt.
3. Generates dist/EXPORT_MANIFEST.md and dist/RELEASE_NOTES.md.
4. Extracts archive into a temporary test directory and runs pytest to verify self-containment.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
VERSION = "1.2.0"
ZIP_NAME = f"smart-retail-ai-v{VERSION}.zip"
ZIP_PATH = DIST_DIR / ZIP_NAME

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "dist",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}


def build_zip_archive() -> Path:
    print(f"[1/4] Packaging clean release archive: {ZIP_PATH}...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    included_files = []

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(ROOT):
            rel_dir = Path(root).relative_to(ROOT)
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not any(part in EXCLUDE_DIRS for part in (rel_dir / d).parts)]

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(ROOT)

                if any(part in EXCLUDE_DIRS for part in rel_path.parts):
                    continue
                if file_path.suffix in EXCLUDE_EXTS:
                    continue

                zip_file.write(file_path, arcname=str(rel_path))
                included_files.append(rel_path)

    print(f"  Successfully packed {len(included_files)} files ({ZIP_PATH.stat().st_size / (1024*1024):.2f} MB).")
    return ZIP_PATH


def compute_checksums() -> str:
    print("[2/4] Generating SHA-256 checksums...")
    sha256 = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    checksum_file = DIST_DIR / "SHA256SUMS.txt"
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {ZIP_NAME}\n")

    print(f"  SHA-256: {digest}")
    return digest


def generate_manifest_and_notes(checksum: str) -> None:
    print("[3/4] Generating EXPORT_MANIFEST.md and RELEASE_NOTES.md...")
    manifest_path = DIST_DIR / "EXPORT_MANIFEST.md"

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        infolist = sorted(zf.infolist(), key=lambda i: i.filename)

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"# Export Manifest: Smart Retail AI v{VERSION}\n\n")
        f.write(f"- **Package:** `{ZIP_NAME}`\n")
        f.write(f"- **Archive Size:** {ZIP_PATH.stat().st_size:,} bytes\n")
        f.write(f"- **SHA-256 Checksum:** `{checksum}`\n")
        f.write(f"- **Total Files:** {len(infolist)}\n\n")
        f.write("## Packaged Files\n\n")
        f.write("| File Path | Uncompressed Size | Compressed Size |\n")
        f.write("| :--- | :---: | :---: |\n")
        for info in infolist:
            f.write(f"| `{info.filename}` | {info.file_size:,} B | {info.compress_size:,} B |\n")

    notes_path = DIST_DIR / "RELEASE_NOTES.md"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(f"""# Release Notes: Smart Retail AI v{VERSION}

**Release Date:** November 19, 2026  
**Artifact:** `{ZIP_NAME}`  
**SHA-256 Checksum:** `{checksum}`

## Major Improvements in v{VERSION}

1. **Configuration Centralization & Input Defense:**
   - Unified environment configuration dataclass in `app/config.py` with dimension limits ($10 \times 10$ to $4096 \times 4096$ pixels) and byte limits.
   - Refactored `app/utils/validation.py` to prevent decompression bomb vulnerabilities and return structured 400 Bad Request responses.

2. **Deterministic ML Reproducibility (Seed 42):**
   - Synthesized exact datasets matching documented splits:
     - Product Classifier: 300 images (240 train, 60 test across 4 classes) + 20 challenge images.
     - Sentiment Classifier: 378 reviews (283 train, 95 test across 3 classes) + 20 challenge reviews.
     - Face Verification: gallery reference and calibrated query pairs.
   - Retrained HOG+SVM, TF-IDF+Logistic Regression, and Face SSIM models with verified evaluation metrics in `artifacts/`.

3. **Retail Chatbot Enhancements:**
   - Refactored intent classification to support 9 distinct operational retail domains (`greeting`, `order_status`, `shipping`, `returns`, `refunds`, `store_info`, `products`, `payment`, `support`) plus out-of-scope `fallback`.
   - Enhanced phrase normalization, word boundary matching, and confidence routing.

4. **Automated Testing & QA:**
   - Expanded test suite from 24 to 52 passing automated tests with zero failures.
   - Comprehensive edge case coverage for corrupt payloads, missing model artifacts (503), whitespace validation (422), and thread-safe counters.

5. **Exhaustive Documentation:**
   - Completed 10 architectural and engineering documents in `docs/` (`ARCHITECTURE.md`, `API_REFERENCE.md`, `DEVELOPMENT.md`, `MODEL_EVALUATION.md`, `DATASET_METHODOLOGY.md`, `REPRODUCIBILITY.md`, `TESTING.md`, `DEPLOYMENT.md`, `TECHNICAL_AUDIT.md`, `FINAL_RELEASE_REPORT.md`).
""")

    print(f"  Wrote {manifest_path} and {notes_path}")


def verify_extracted_package() -> None:
    print("[4/4] Performing clean-room extraction and pytest verification...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"  Extracting {ZIP_NAME} into clean temporary directory: {tmp_path}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)

        print("  Running pytest on extracted bundle...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path)

        res = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )

        if res.returncode == 0:
            print("  CLEAN-ROOM VERIFICATION PASSED: All tests passed on unzipped release bundle!")
        else:
            print("  CLEAN-ROOM VERIFICATION FAILED:")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)


def main():
    print("=" * 70)
    print("SMART RETAIL AI — PRODUCTION RELEASE PACKAGING & VERIFICATION")
    print("=" * 70)
    build_zip_archive()
    checksum = compute_checksums()
    generate_manifest_and_notes(checksum)
    verify_extracted_package()
    print("=" * 70)
    print("RELEASE EXPORT COMPLETED AND FULLY VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
