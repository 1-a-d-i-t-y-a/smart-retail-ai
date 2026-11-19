"""
tests/test_packaging.py
-----------------------
Verifies the exclusion and secret-scanning rules in scripts/package_release.py
without actually building the full release archive.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.package_release import (
    EXCLUDE_DIRS,
    EXCLUDE_EXTENSIONS,
    EXCLUDE_FILENAMES,
    SECRET_PATTERNS,
    _is_excluded,
    _scan_file_for_secrets,
)


# ---------------------------------------------------------------------------
# Exclusion rule tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,expected_excluded", [
    # Git internals
    (".git/config",                     True),
    (".git/HEAD",                       True),
    # Cache directories
    ("__pycache__/module.cpython-312.pyc", True),
    (".pytest_cache/v/cache/lastfailed",   True),
    # Virtual environments
    (".venv/lib/python3.12/site.py",    True),
    ("venv/pyvenv.cfg",                 True),
    # Compiled Python
    ("app/core.pyc",                    True),
    ("app/services/sentiment_service.pyo", True),
    # Credential files
    ("id_rsa",                          True),
    ("id_ed25519",                      True),
    ("secrets.pem",                     True),
    ("private.key",                     True),
    ("cert.p12",                        True),
    ("keystore.pfx",                    True),
    # Environment files
    (".env",                            True),
    (".env.production",                 True),
    # Allowed: .env.example should NOT be excluded
    (".env.example",                    False),
    # Build output directory
    ("dist/smart-retail-ai-v1.2.0.zip", True),
    # Log files
    ("logs/app.log",                    True),
    # OS junk
    (".DS_Store",                       True),
    ("Thumbs.db",                       True),
    # Normal project files should be included
    ("app/main.py",                     False),
    ("requirements.txt",                False),
    ("README.md",                       False),
    ("scripts/train_baseline_models.py", False),
    ("tests/test_chatbot.py",           False),
    ("docs/ARCHITECTURE.md",            False),
    ("artifacts/product/metrics.json",  False),
])
def test_exclusion_rules(rel_path: str, expected_excluded: bool):
    """Every path is correctly classified by _is_excluded()."""
    result = _is_excluded(Path(rel_path))
    assert result == expected_excluded, (
        f"_is_excluded({rel_path!r}) returned {result}, expected {expected_excluded}"
    )


# ---------------------------------------------------------------------------
# Secret scanner tests
# ---------------------------------------------------------------------------

def _write_tmp(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def test_secret_scan_clean_file(tmp_path: Path):
    """A clean Python file produces no hits."""
    p = _write_tmp(tmp_path, "clean.py", "API_URL = 'https://api.example.com'\nDEBUG = True\n")
    hits = _scan_file_for_secrets(p)
    assert hits == []


def test_secret_scan_github_pat(tmp_path: Path):
    """A file containing a GitHub PAT is detected."""
    token = "ghp_" + "A" * 36
    p = _write_tmp(tmp_path, "config.py", f'TOKEN = "{token}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("GitHub PAT" in name for name, _ in hits)


def test_secret_scan_openai_key(tmp_path: Path):
    """A file containing an OpenAI key pattern is detected."""
    token = "sk-" + "AbCdEfGhIjKlMnOpQrSt"
    p = _write_tmp(tmp_path, "settings.py", f'OPENAI_KEY = "{token}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("OpenAI" in name for name, _ in hits)


def test_secret_scan_aws_key(tmp_path: Path):
    """A file containing an AWS access key is detected."""
    key = "AKIA" + "IOSFODNN7EXAMPLE"
    p = _write_tmp(tmp_path, "aws.py", f'AWS_ACCESS_KEY_ID = "{key}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("AWS" in name for name, _ in hits)


def test_secret_scan_pem_block(tmp_path: Path):
    """A file containing a PEM private key header is detected."""
    pem_hdr = "-----BEGIN " + "RSA PRIVATE KEY-----"
    pem_ftr = "-----END " + "RSA PRIVATE KEY-----"
    p = _write_tmp(tmp_path, "key.txt", f"{pem_hdr}\nMIIE...\n{pem_ftr}\n")
    hits = _scan_file_for_secrets(p)
    assert any("PEM" in name for name, _ in hits)


def test_secret_scan_password_assignment(tmp_path: Path):
    """A file with a hardcoded password assignment is detected."""
    pw_field = "pass" + "word"
    p = _write_tmp(tmp_path, "db.py", f'{pw_field} = "supersecretpassword"\n')
    hits = _scan_file_for_secrets(p)
    assert any("password" in name.lower() for name, _ in hits)


def test_secret_scan_binary_file_skipped(tmp_path: Path):
    """Binary files (e.g. .png) are not scanned for text patterns."""
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    hits = _scan_file_for_secrets(p)
    assert hits == []


# ---------------------------------------------------------------------------
# Constant set integrity checks
# ---------------------------------------------------------------------------

def test_exclude_dirs_contains_critical_entries():
    assert ".git" in EXCLUDE_DIRS
    assert "__pycache__" in EXCLUDE_DIRS
    assert ".venv" in EXCLUDE_DIRS
    assert "dist" in EXCLUDE_DIRS


def test_exclude_extensions_contains_critical_entries():
    assert ".pyc" in EXCLUDE_EXTENSIONS
    assert ".pem" in EXCLUDE_EXTENSIONS
    assert ".key" in EXCLUDE_EXTENSIONS
    assert ".log" in EXCLUDE_EXTENSIONS


def test_exclude_filenames_contains_credential_files():
    assert "id_rsa" in EXCLUDE_FILENAMES
    assert ".env" in EXCLUDE_FILENAMES
    assert ".DS_Store" in EXCLUDE_FILENAMES


def test_secret_patterns_covers_major_providers():
    pattern_names = [name for name, _ in SECRET_PATTERNS]
    assert any("GitHub" in n for n in pattern_names)
    assert any("OpenAI" in n for n in pattern_names)
    assert any("AWS" in n for n in pattern_names)
    assert any("PEM" in n for n in pattern_names)
