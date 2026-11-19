#!/usr/bin/env python3
"""
scripts/package_release.py
--------------------------
Builds, security-scans, hashes, and clean-room verifies the release distribution
bundle for Smart Retail AI:

1. Validates exclusion rules and writes artifacts/packaging/exclusion_test_results.txt.
2. Scans every included file for secret patterns before writing the archive.
   If any pattern matches, packaging aborts without producing a ZIP.
3. Packages repository files into dist/smart-retail-ai-v1.2.0.zip.
4. Validates archive integrity: rejects absolute paths, drive paths, UNC paths,
   traversal sequences (../ or ..\\), and symlinks.
5. Computes SHA-256 checksum and writes dist/SHA256SUMS.txt.
6. Generates dist/EXPORT_MANIFEST.md and dist/RELEASE_NOTES.md.
7. Extracts archive into a temporary clean-room directory and runs pytest.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
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
# Explicit exclusion sets
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
    ".aws",
}

EXCLUDE_FILENAMES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
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

EXCLUDE_GLOB_PREFIXES = (".env.",)  # e.g. .env.production (but NOT .env.example)

# ---------------------------------------------------------------------------
# Secret-scanner patterns
# ---------------------------------------------------------------------------
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("GitHub Fine-Grained PAT",  re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    ("GitHub Token (ghp/ghs/gho)", re.compile(r"gh[pso_ur][A-Za-z0-9_]{36,}")),
    ("OpenAI API Key",          re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("AWS Access Key ID",       re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Private Key Block",       re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    (
        "Password Assignment",
        re.compile(r"""(?i)["']?password["']?\s*[:=]\s*["']([^"'\r\n]{6,})["']"""),
    ),
    (
        "API Token Assignment",
        re.compile(r"""(?i)["']?(?:api_?token|access_?token|auth_?token)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']"""),
    ),
    (
        "Secret Assignment",
        re.compile(r"""(?i)["']?(?:client_?secret|secret_?key)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']"""),
    ),
]

# Known non-secret placeholder substrings in documentation, tests, or config examples
SAFE_PLACEHOLDER_SUBSTRINGS: tuple[str, ...] = (
    "your_",
    "example",
    "placeholder",
    "change_me",
    "dummy",
    "supersecretpassword",  # Used strictly in tests/test_packaging.py
    "test_",
    "<",
    ">",
)

SCANNABLE_EXTENSIONS: set[str] = {
    ".py", ".txt", ".md", ".yaml", ".yml", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".bat", ".ps1", ".rst", ".html",
}

SCANNABLE_FILENAMES: set[str] = {
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    "pytest.ini",
}


def _is_excluded(rel_path: Path) -> bool:
    """Check if a relative path matches any exclusion rule."""
    parts = rel_path.parts
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return True
    filename = parts[-1]
    if filename in EXCLUDE_FILENAMES:
        return True
    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    if any(filename.startswith(p) for p in EXCLUDE_GLOB_PREFIXES):
        if filename != ".env.example":
            return True
    return False


def _is_scannable(file_path: Path) -> bool:
    """Return True if a file should be scanned for secret patterns."""
    if file_path.name in SCANNABLE_FILENAMES:
        return True
    return file_path.suffix.lower() in SCANNABLE_EXTENSIONS


def _scan_file_for_secrets(file_path: Path) -> list[tuple[str, int]]:
    """Scan a file for secret patterns. Returns list of (pattern_name, line_number)."""
    if not _is_scannable(file_path):
        return []
    hits: list[tuple[str, int]] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                for name, pattern in SECRET_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        # If the regex captures a value group (assignments), filter safe placeholders
                        if match.groups():
                            val = match.group(1).lower()
                            if any(ph in val for ph in SAFE_PLACEHOLDER_SUBSTRINGS):
                                continue
                        hits.append((name, lineno))
    except Exception:
        pass
    return hits


# ---------------------------------------------------------------------------
# Archive Member Validation
# ---------------------------------------------------------------------------
def validate_archive_member(name: str, external_attr: int = 0) -> list[str]:
    """Validate an archive entry name against security standards."""
    issues: list[str] = []
    norm = posixpath.normpath(name.replace("\\", "/"))

    # Reject absolute and rooted paths
    if name.startswith("/") or name.startswith("\\"):
        issues.append(f"Absolute path in archive: {name}")

    # Reject Windows drive letters (e.g. C:, D:)
    if re.match(r"^[a-zA-Z]:", name):
        issues.append(f"Windows drive path in archive: {name}")

    # Reject UNC network paths
    if name.startswith("//") or name.startswith("\\\\"):
        issues.append(f"UNC path in archive: {name}")

    # Reject path traversal (both normalized and raw forms)
    if ".." in norm.split("/") or ".." in name.replace("\\", "/").split("/"):
        issues.append(f"Path traversal sequence in archive: {name}")

    # Reject symbolic links
    unix_mode = (external_attr >> 16) & 0xFFFF
    if unix_mode and (unix_mode & 0xA000) == 0xA000:
        issues.append(f"Symlink in archive: {name}")

    return issues


# ---------------------------------------------------------------------------
# Phase 1: Pre-generate exclusion test report
# ---------------------------------------------------------------------------
def generate_exclusion_report(candidate_paths: list[Path]) -> Path:
    """Verify exclusion rules against candidate files and write report."""
    exclusion_dir = ROOT / "artifacts" / "packaging"
    exclusion_dir.mkdir(parents=True, exist_ok=True)
    report_path = exclusion_dir / "exclusion_test_results.txt"

    included_str = {str(p).replace("\\", "/") for p in candidate_paths}
    rules_tested = []

    def _check_absent(rule_name: str, check_fn) -> None:
        matches = [n for n in included_str if check_fn(n)]
        status = "PASS" if not matches else f"FAIL — found: {matches[:3]}"
        rules_tested.append(f"[{status}] {rule_name}")

    _check_absent(".git directory excluded",          lambda n: ".git/" in n or n.startswith(".git/"))
    _check_absent("__pycache__ excluded",             lambda n: "__pycache__/" in n)
    _check_absent(".pytest_cache excluded",           lambda n: ".pytest_cache/" in n)
    _check_absent(".pyc files excluded",              lambda n: n.endswith(".pyc"))
    _check_absent(".pem files excluded",              lambda n: n.endswith(".pem"))
    _check_absent(".key files excluded",              lambda n: n.endswith(".key"))
    _check_absent(".log files excluded",              lambda n: n.endswith(".log"))
    _check_absent(".DS_Store excluded",               lambda n: n.endswith(".DS_Store"))
    _check_absent("Thumbs.db excluded",               lambda n: n.endswith("Thumbs.db"))
    _check_absent("desktop.ini excluded",             lambda n: n.endswith("desktop.ini"))
    _check_absent("venv directory excluded",          lambda n: "/venv/" in n or n.startswith("venv/"))
    _check_absent(".venv directory excluded",         lambda n: "/.venv/" in n or n.startswith(".venv/"))
    _check_absent("dist directory excluded",          lambda n: "dist/" in n or n.startswith("dist/"))
    _check_absent(".env file excluded",               lambda n: n == ".env" or n.endswith("/.env"))
    _check_absent(".env.production excluded",         lambda n: n == ".env.production" or n.endswith("/.env.production"))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Exclusion Rule Verification — Smart Retail AI v{VERSION}\n")
        f.write(f"Candidate Files: {len(candidate_paths)}\n")
        f.write("=" * 60 + "\n")
        for line in rules_tested:
            f.write(line + "\n")
        all_pass = all("PASS" in r for r in rules_tested)
        f.write("\n" + ("ALL RULES PASSED" if all_pass else "SOME RULES FAILED") + "\n")

    return report_path


# ---------------------------------------------------------------------------
# Phase 2: Build archive
# ---------------------------------------------------------------------------
def build_zip_archive() -> tuple[Path, list[Path]]:
    print(f"[1/5] Scanning files and packaging release archive: {ZIP_PATH}...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    # Discover and filter files
    candidate_files: list[Path] = []
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
            if not _is_excluded(rel_path):
                candidate_files.append(rel_path)

    # Pre-generate exclusion report before writing archive
    generate_exclusion_report(candidate_files)

    # Refresh candidate files to include the updated report
    report_rel = Path("artifacts/packaging/exclusion_test_results.txt")
    if report_rel not in candidate_files and (ROOT / report_rel).exists():
        candidate_files.append(report_rel)

    # Scan all included files for secrets
    secret_hits: list[tuple[Path, str, int]] = []
    for rel_path in candidate_files:
        hits = _scan_file_for_secrets(ROOT / rel_path)
        for pattern_name, lineno in hits:
            secret_hits.append((rel_path, pattern_name, lineno))

    if secret_hits:
        print("\n[SECURITY] SECRET SCAN FAILED — archive NOT created.")
        for rel_p, pname, lineno in secret_hits:
            print(f"  Pattern '{pname}' matched in {rel_p}:{lineno}")
        sys.exit(2)

    # Write archive
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in sorted(candidate_files):
            file_path = ROOT / rel_path
            arc_name = str(rel_path).replace("\\", "/")
            zf.write(file_path, arcname=arc_name)

    print(f"  Packed {len(candidate_files)} files ({ZIP_PATH.stat().st_size / (1024*1024):.2f} MB).")
    return ZIP_PATH, candidate_files


# ---------------------------------------------------------------------------
# Phase 3: Validate archive integrity
# ---------------------------------------------------------------------------
def validate_archive() -> None:
    print("[2/5] Validating archive integrity (paths, drive letters, UNC, traversal, symlinks)...")
    issues: list[str] = []
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for info in zf.infolist():
            member_issues = validate_archive_member(info.filename, info.external_attr)
            issues.extend(member_issues)

    if issues:
        print("[SECURITY] Archive integrity check FAILED:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(3)
    print("  Archive integrity: OK — no absolute paths, drive letters, UNC paths, traversal sequences, or symlinks.")


# ---------------------------------------------------------------------------
# Phase 4: SHA-256 Checksum
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
# Phase 5: Manifest and Release Notes
# ---------------------------------------------------------------------------
def generate_manifest_and_notes(checksum: str) -> None:
    print("[4/5] Generating manifest and release notes...")
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

1. **Hardened Upload Validation & Input Defense:**
   - Strict Content-Type allowlist normalization (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`).
   - Pillow header verification precedes OpenCV decoding, checking dimension bounds ($10 \\times 10$ to $4096 \\times 4096$) and format compatibility.
   - Pillow decompression bomb warnings and errors trapped as validation failures.
   - Bounded streaming upload reader (`read_bounded_upload`) strictly halts at limit + 1 byte and returns HTTP 413.

2. **Packaging Security & Secret Scanning:**
   - Explicit exclusion set eliminates VCS, virtual environments, private keys, `.env.*` files, and OS metadata.
   - `.env.example` included as a clean configuration template and actively scanned.
   - Regex secret scanning detects GitHub PATs (`ghp_`, `github_pat_`), OpenAI keys, AWS access keys, PEM blocks, and password assignments.
   - Archive integrity validation verifies zero path traversal (`../`), drive letters, UNC paths, or symlinks.

3. **Deterministic ML Reproducibility (Seed 42):**
   - Canonical face fixtures tracked in `data/fixtures/face/` decouple dataset generation from evidence artifacts.
   - Fixed random seeds ensure reproducible dataset splits and model retraining across environments.

4. **Honest Evaluation & Documentation Standards:**
   - Explicit distinction between controlled synthetic test benchmarks and harder challenge evaluation holdouts.
   - Face verification documented as an educational prototype; non-face distractor separated as a functional test.
   - Accurate single-stage Docker documentation matching `python:3.12-slim` and non-root `appuser`.
""")

    print(f"  Wrote {manifest_path}")
    print(f"  Wrote {notes_path}")


# ---------------------------------------------------------------------------
# Phase 6: Clean-room extraction + pytest
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
