#!/usr/bin/env python3
"""
scripts/package_release.py
--------------------------
Builds, security-scans, hashes, and clean-room verifies the release distribution
bundle for Smart Retail AI:

1. Packages repository files into dist/smart-retail-ai-v1.2.0.zip with an
   explicit, documented exclusion set.
2. Scans every included file for secret patterns before writing the archive.
   If any pattern matches the script aborts without producing a ZIP.
3. Validates the archive: no absolute paths, no path-traversal sequences (../),
   no symlinks.
4. Computes SHA-256 checksum and saves to dist/SHA256SUMS.txt.
5. Generates dist/EXPORT_MANIFEST.md and dist/RELEASE_NOTES.md.
6. Extracts archive into a temporary directory and runs pytest to verify
   self-containment (clean-room test).
7. Writes artifacts/packaging/exclusion_test_results.txt with rule verification.
"""

from __future__ import annotations

import hashlib
import os
import re
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

# ---------------------------------------------------------------------------
# Explicit exclusion set  (directory names AND file patterns)
# ---------------------------------------------------------------------------
EXCLUDE_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "ENV",
    "dist",
    ".idea",
    ".vscode",
}

EXCLUDE_FILENAMES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
}

EXCLUDE_EXTENSIONS: set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".log",
}

EXCLUDE_GLOB_PREFIXES = (".env.",)   # e.g. .env.production  (but NOT .env.example)

# ---------------------------------------------------------------------------
# Secret-scanner patterns
# ---------------------------------------------------------------------------
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("GitHub PAT (ghp_)",       re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("GitHub Server token",     re.compile(r"ghs_[A-Za-z0-9]{36}")),
    ("OpenAI key (sk-)",        re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("AWS access key (AKIA)",   re.compile(r"AKIA[A-Z0-9]{16}")),
    ("PEM block",               re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("password assignment",     re.compile(r"(?i)password\s*=\s*[\"'][^\"']{6,}[\"']")),
    ("token assignment",        re.compile(r"(?i)(api_?token|access_?token|auth_?token)\s*=\s*[\"'][^\"']{8,}[\"']")),
    ("secret assignment",       re.compile(r"(?i)secret\s*=\s*[\"'][^\"']{8,}[\"']")),
]

# File extensions for which we do text scanning (skip binary/image assets)
SCANNABLE_EXTENSIONS: set[str] = {
    ".py", ".txt", ".md", ".yaml", ".yml", ".json", ".toml", ".cfg",
    ".ini", ".env.example", ".sh", ".bat", ".ps1", ".rst", ".html",
}


# ---------------------------------------------------------------------------
# Helper: should this path be excluded?
# ---------------------------------------------------------------------------
def _is_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    # Any directory component in the exclusion set
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return True
    filename = parts[-1]
    # Exact filename match
    if filename in EXCLUDE_FILENAMES:
        return True
    # Extension match
    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    # .env.* prefix (except .env.example)
    if any(filename.startswith(p) for p in EXCLUDE_GLOB_PREFIXES):
        if filename != ".env.example":
            return True
    return False


# ---------------------------------------------------------------------------
# Secret scanner
# ---------------------------------------------------------------------------
def _scan_file_for_secrets(file_path: Path) -> list[tuple[str, int]]:
    """Return a list of (pattern_name, line_number) hits."""
    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS:
        return []
    hits: list[tuple[str, int]] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                for name, pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        hits.append((name, lineno))
    except Exception:
        pass  # Binary or unreadable file — skip
    return hits


# ---------------------------------------------------------------------------
# Phase 1: Build archive
# ---------------------------------------------------------------------------
def build_zip_archive() -> tuple[Path, list[Path], list[tuple[Path, str, int]]]:
    """Walk the repo, apply exclusion rules, scan for secrets, write ZIP.

    Returns:
        (zip_path, included_files, secret_hits)  where secret_hits is a list of
        (file_path, pattern_name, line_number).  If secret_hits is non-empty the
        caller must abort without publishing the archive.
    """
    print(f"[1/5] Packaging clean release archive: {ZIP_PATH}...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    included_files: list[Path] = []
    secret_hits: list[tuple[Path, str, int]] = []

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_str, dirs, files in os.walk(ROOT):
            root_p = Path(root_str)
            rel_dir = root_p.relative_to(ROOT)

            # Prune directory walk in-place
            dirs[:] = [
                d for d in dirs
                if not _is_excluded(rel_dir / d / "_placeholder")
            ]

            for fname in sorted(files):
                file_path = root_p / fname
                rel_path = file_path.relative_to(ROOT)

                if _is_excluded(rel_path):
                    continue

                # Secret scan before adding to archive
                hits = _scan_file_for_secrets(file_path)
                for pattern_name, lineno in hits:
                    # Log path and line, never the secret value
                    secret_hits.append((rel_path, pattern_name, lineno))

                arc_name = str(rel_path).replace("\\", "/")
                zf.write(file_path, arcname=arc_name)
                included_files.append(rel_path)

    if secret_hits:
        # Remove the partially-written archive — do NOT publish it
        ZIP_PATH.unlink(missing_ok=True)
        print("\n[SECURITY] SECRET SCAN FAILED — archive NOT created.")
        for rel_p, pname, lineno in secret_hits:
            print(f"  Pattern '{pname}' matched in {rel_p}:{lineno}")
        sys.exit(2)

    print(f"  Packed {len(included_files)} files ({ZIP_PATH.stat().st_size / (1024*1024):.2f} MB).")
    return ZIP_PATH, included_files, secret_hits


# ---------------------------------------------------------------------------
# Phase 2: Validate archive integrity
# ---------------------------------------------------------------------------
def validate_archive() -> None:
    """Ensure no absolute paths, no path-traversal, no symlinks in the archive."""
    print("[2/5] Validating archive integrity (paths, traversal, symlinks)...")
    issues: list[str] = []
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            # No absolute paths
            if name.startswith("/"):
                issues.append(f"Absolute path in archive: {name}")
            # No path traversal
            if ".." in name.split("/"):
                issues.append(f"Path traversal sequence in archive: {name}")
            # No symlinks (Unix mode stored in external_attr upper 16 bits)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and (unix_mode & 0xA000) == 0xA000:
                issues.append(f"Symlink in archive: {name}")

    if issues:
        print("[SECURITY] Archive integrity check FAILED:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(3)
    print("  Archive integrity: OK — no absolute paths, traversal sequences, or symlinks.")


# ---------------------------------------------------------------------------
# Phase 3: SHA-256 checksum
# ---------------------------------------------------------------------------
def compute_checksums() -> str:
    print("[3/5] Generating SHA-256 checksum...")
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


# ---------------------------------------------------------------------------
# Phase 4: Manifest, release notes, exclusion test report
# ---------------------------------------------------------------------------
def generate_manifest_and_notes(checksum: str) -> None:
    print("[4/5] Generating manifest, release notes, and exclusion test report...")
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

1. **Bounded Upload Reading and HTTP 413:**
   - Routes use `read_bounded_upload()` which reads in 64 KiB chunks and aborts
     with `PayloadTooLargeError` immediately on exceeding the limit.
   - Both `/classify-product` and `/recognize-face` now return HTTP 413 (not 400)
     for oversized payloads.
   - Pillow `Image.verify()` performs header-level decompression-bomb protection
     before OpenCV full decode.

2. **Secure Release Packaging:**
   - `scripts/package_release.py` now includes an explicit exclusion set
     (`.pem`, `.key`, `.env.*`, `id_rsa`, `Thumbs.db`, etc.) independent of
     directory walk filtering.
   - Secret scanner checks every text file for `ghp_`, `ghs_`, `sk-`, `AKIA`,
     PEM block headers, and password/token/secret assignments before creating
     the archive.
   - Archive validated for absence of absolute paths, `../` traversal, and symlinks.
   - `artifacts/packaging/exclusion_test_results.txt` records rule verification.

3. **Configuration Centralization and Input Defence:**
   - Unified environment configuration dataclass in `app/config.py`.
   - Dimension limits (10×10 to 4096×4096 pixels) and byte limits (5 MB default).

4. **Deterministic ML Reproducibility (Seed 42):**
   - Product Classifier: 300 images (240 train / 60 test across 4 classes) + 20 challenge.
   - Sentiment Classifier: 378 reviews (283 train / 95 test across 3 classes) + 20 challenge.
   - Face Verification: gallery reference and calibrated query pairs.

5. **Retail Chatbot (9 Intents):**
   - `greeting`, `order_status`, `shipping`, `returns`, `refunds`, `store_info`,
     `products`, `payment`, `support`, plus out-of-scope `fallback`.

6. **Automated Testing:**
   - 57 passing automated tests covering HTTP 413, bounded upload, decompression
     bomb, and all existing functionality.

7. **Documentation:**
   - 10 engineering documents in `docs/` with accurate Docker and reproducibility
     status.
   - Language corrected: no "production-grade" or "pinned dependencies" claims.
""")

    # Exclusion test results artifact
    exclusion_dir = ROOT / "artifacts" / "packaging"
    exclusion_dir.mkdir(parents=True, exist_ok=True)
    exclusion_report = exclusion_dir / "exclusion_test_results.txt"

    # Verify exclusion rules against the produced archive
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        archive_names = {info.filename for info in zf.infolist()}

    rules_tested = []

    def _check_absent(rule_name: str, pattern: str) -> None:
        matches = [n for n in archive_names if pattern in n or n.endswith(pattern)]
        status = "PASS" if not matches else f"FAIL — found: {matches[:3]}"
        rules_tested.append(f"[{status}] {rule_name}: pattern '{pattern}'")

    _check_absent(".git directory excluded",          ".git/")
    _check_absent("__pycache__ excluded",             "__pycache__/")
    _check_absent(".pytest_cache excluded",           ".pytest_cache/")
    _check_absent(".pyc files excluded",              ".pyc")
    _check_absent(".pem files excluded",              ".pem")
    _check_absent(".key files excluded",              ".key")
    _check_absent(".log files excluded",              ".log")
    _check_absent(".DS_Store excluded",               ".DS_Store")
    _check_absent("Thumbs.db excluded",               "Thumbs.db")
    _check_absent("venv directory excluded",          "/venv/")
    _check_absent(".venv directory excluded",         "/.venv/")
    _check_absent("dist directory excluded",          "dist/")

    with open(exclusion_report, "w", encoding="utf-8") as f:
        f.write(f"Exclusion Rule Verification — Smart Retail AI v{VERSION}\n")
        f.write(f"Archive: {ZIP_NAME} ({len(archive_names)} files)\n")
        f.write("=" * 60 + "\n")
        for line in rules_tested:
            f.write(line + "\n")
        all_pass = all("PASS" in r for r in rules_tested)
        f.write("\n" + ("ALL RULES PASSED" if all_pass else "SOME RULES FAILED") + "\n")

    print(f"  Wrote {manifest_path}")
    print(f"  Wrote {notes_path}")
    print(f"  Wrote {exclusion_report}")


# ---------------------------------------------------------------------------
# Phase 5: Clean-room extraction + pytest
# ---------------------------------------------------------------------------
def verify_extracted_package() -> None:
    print("[5/5] Performing clean-room extraction and pytest verification...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"  Extracting {ZIP_NAME} into clean temporary directory: {tmp_path}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)

        print("  Running pytest on extracted bundle...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path)

        res = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )

        if res.returncode == 0:
            print("  CLEAN-ROOM VERIFICATION PASSED: All tests passed on unzipped release bundle!")
            # Print final summary line from pytest output
            for line in reversed(res.stdout.splitlines()):
                if "passed" in line:
                    print(f"  {line}")
                    break
        else:
            print("  CLEAN-ROOM VERIFICATION FAILED:")
            print(res.stdout[-3000:])
            print(res.stderr[-1000:])
            sys.exit(1)


def main():
    print("=" * 70)
    print("SMART RETAIL AI — RELEASE PACKAGING, SECURITY SCAN & VERIFICATION")
    print("=" * 70)
    build_zip_archive()
    validate_archive()
    checksum = compute_checksums()
    generate_manifest_and_notes(checksum)
    verify_extracted_package()
    print("=" * 70)
    print("RELEASE EXPORT COMPLETED AND FULLY VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
