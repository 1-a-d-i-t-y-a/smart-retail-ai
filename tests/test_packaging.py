"""
tests/test_packaging.py
-----------------------
Unit tests verifying exclusion rules, secret scanning, archive path safety,
and packaging constant integrity in scripts/package_release.py.
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
    _is_scannable,
    _scan_file_for_secrets,
    validate_archive_member,
)


# ---------------------------------------------------------------------------
# Exclusion rule tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel_path,expected_excluded", [
    # VCS & caches
    (".git/config", True),
    (".git/HEAD", True),
    ("__pycache__/module.cpython-312.pyc", True),
    (".pytest_cache/v/cache/lastfailed", True),
    (".mypy_cache/3.12/index.data.json", True),
    (".ruff_cache/content", True),
    # Virtual environments
    (".venv/lib/python3.12/site.py", True),
    ("venv/pyvenv.cfg", True),
    ("env/bin/activate", True),
    ("ENV/bin/python", True),
    # Compiled Python
    ("app/core.pyc", True),
    ("app/services/sentiment_service.pyo", True),
    # Credential & key files
    ("id_rsa", True),
    ("id_ed25519", True),
    ("id_ecdsa", True),
    ("id_dsa", True),
    ("secrets.pem", True),
    ("server.key", True),
    ("cert.p12", True),
    ("keystore.pfx", True),
    (".aws/credentials", True),
    # Environment files
    (".env", True),
    (".env.production", True),
    (".env.local", True),
    (".env.staging", True),
    # Allowed: .env.example MUST be included
    (".env.example", False),
    # Build output
    ("dist/smart-retail-ai-v1.2.0.zip", True),
    # Log files
    ("logs/app.log", True),
    ("error.log", True),
    # OS junk
    (".DS_Store", True),
    ("Thumbs.db", True),
    ("desktop.ini", True),
    # Normal project files MUST be included
    ("app/main.py", False),
    ("requirements.txt", False),
    ("README.md", False),
    ("scripts/train_baseline_models.py", False),
    ("tests/test_chatbot.py", False),
    ("docs/ARCHITECTURE.md", False),
    ("artifacts/product/metrics.json", False),
    ("data/fixtures/face/canonical_reference.png", False),
    ("data/chatbot/retail_knowledge.json", False),
])
def test_exclusion_rules(rel_path: str, expected_excluded: bool):
    """Verify that _is_excluded accurately classifies paths."""
    result = _is_excluded(Path(rel_path))
    assert result == expected_excluded, (
        f"_is_excluded({rel_path!r}) returned {result}, expected {expected_excluded}"
    )


# ---------------------------------------------------------------------------
# Scannable files tests (.env.example inclusion)
# ---------------------------------------------------------------------------
def test_is_scannable_env_example():
    """Verify that .env.example is recognized as scannable."""
    assert _is_scannable(Path(".env.example")) is True


def test_is_scannable_code_and_config():
    assert _is_scannable(Path("app/main.py")) is True
    assert _is_scannable(Path("Dockerfile")) is True
    assert _is_scannable(Path("docker-compose.yml")) is True
    assert _is_scannable(Path("pytest.ini")) is True


def test_is_not_scannable_binary():
    assert _is_scannable(Path("model.joblib")) is False
    assert _is_scannable(Path("image.png")) is False


# ---------------------------------------------------------------------------
# Secret scanner tests
# ---------------------------------------------------------------------------
def _write_tmp(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def test_secret_scan_clean_file(tmp_path: Path):
    p = _write_tmp(tmp_path, "clean.py", "API_URL = 'https://api.example.com'\nDEBUG = True\n")
    hits = _scan_file_for_secrets(p)
    assert hits == []


def test_secret_scan_github_fine_grained_pat(tmp_path: Path):
    # Construct dynamically to avoid flagging test file itself
    token = "github_pat_" + "11ABCD" + "A" * 76
    p = _write_tmp(tmp_path, "config.py", f'TOKEN = "{token}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("Fine-Grained" in name for name, _ in hits)


def test_secret_scan_github_classic_pat(tmp_path: Path):
    token = "ghp_" + "A" * 36
    p = _write_tmp(tmp_path, "ci.py", f'GH_TOKEN = "{token}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("GitHub Token" in name for name, _ in hits)


def test_secret_scan_github_oauth_token(tmp_path: Path):
    token = "gho_" + "B" * 36
    p = _write_tmp(tmp_path, "auth.py", f'AUTH = "{token}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("GitHub Token" in name for name, _ in hits)


def test_secret_scan_openai_key(tmp_path: Path):
    key = "sk-" + "proj-" + "1234567890abcdef12345678"
    p = _write_tmp(tmp_path, "settings.py", f'OPENAI_KEY = "{key}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("OpenAI" in name for name, _ in hits)


def test_secret_scan_aws_key(tmp_path: Path):
    key = "AKIA" + "IOSFODNN7EXAMPLE"
    p = _write_tmp(tmp_path, "aws.py", f'AWS_ACCESS_KEY_ID = "{key}"\n')
    hits = _scan_file_for_secrets(p)
    assert any("AWS" in name for name, _ in hits)


def test_secret_scan_pem_block(tmp_path: Path):
    pem_hdr = "-----BEGIN " + "RSA PRIVATE KEY-----"
    pem_ftr = "-----END " + "RSA PRIVATE KEY-----"
    p = _write_tmp(tmp_path, "key.txt", f"{pem_hdr}\nMIIE...\n{pem_ftr}\n")
    hits = _scan_file_for_secrets(p)
    assert any("Private Key" in name for name, _ in hits)


def test_secret_scan_password_json_assignment(tmp_path: Path):
    pw_field = "pass" + "word"
    p = _write_tmp(tmp_path, "credentials.json", f'{{"{pw_field}": "unmasked_secret_val"}}\n')
    hits = _scan_file_for_secrets(p)
    assert any("Password" in name for name, _ in hits)


def test_secret_scan_api_token_assignment(tmp_path: Path):
    tok_field = "api_" + "token"
    p = _write_tmp(tmp_path, "service.py", f'{tok_field} = "real_production_token_12345"\n')
    hits = _scan_file_for_secrets(p)
    assert any("Token" in name for name, _ in hits)


def test_secret_scan_safe_placeholder_ignored(tmp_path: Path):
    # Safe placeholders should not trigger false positive assignment alerts
    pw_field = "pass" + "word"
    p = _write_tmp(tmp_path, "sample.env", f'{pw_field}="your_password_here"\n')
    hits = _scan_file_for_secrets(p)
    assert hits == []


# ---------------------------------------------------------------------------
# Archive Path & Traversal Validation Tests
# ---------------------------------------------------------------------------
def test_validate_archive_member_safe_path():
    assert validate_archive_member("app/main.py") == []
    assert validate_archive_member("docs/ARCHITECTURE.md") == []
    assert validate_archive_member("data/chatbot/retail_knowledge.json") == []


def test_validate_archive_member_absolute_path():
    issues = validate_archive_member("/etc/passwd")
    assert any("Absolute path" in i for i in issues)
    issues_win = validate_archive_member("\\Windows\\System32")
    assert any("Absolute path" in i for i in issues_win)


def test_validate_archive_member_windows_drive_path():
    issues = validate_archive_member("C:/Users/app.py")
    assert any("Windows drive path" in i for i in issues)
    issues_d = validate_archive_member("D:\\data\\file.txt")
    assert any("Windows drive path" in i for i in issues_d)


def test_validate_archive_member_unc_path():
    issues_fwd = validate_archive_member("//192.168.1.1/share/file.txt")
    assert any("UNC path" in i for i in issues_fwd)
    issues_bwd = validate_archive_member("\\\\network-share\\file.txt")
    assert any("UNC path" in i for i in issues_bwd)


def test_validate_archive_member_path_traversal():
    issues = validate_archive_member("../evil.py")
    assert any("Path traversal" in i for i in issues)
    issues_nested = validate_archive_member("app/../../secret.txt")
    assert any("Path traversal" in i for i in issues_nested)
    issues_backslash = validate_archive_member("app\\..\\secret.txt")
    assert any("Path traversal" in i for i in issues_backslash)


def test_validate_archive_member_symlink():
    # 0xA000 indicates a symbolic link in the unix mode bits
    symlink_attr = 0xA1ED << 16
    issues = validate_archive_member("link_to_etc", external_attr=symlink_attr)
    assert any("Symlink" in i for i in issues)


# ---------------------------------------------------------------------------
# Constant Set Integrity
# ---------------------------------------------------------------------------
def test_exclude_dirs_integrity():
    for required in [".git", "__pycache__", ".venv", "venv", "dist", ".aws"]:
        assert required in EXCLUDE_DIRS


def test_exclude_extensions_integrity():
    for required in [".pyc", ".pem", ".key", ".p12", ".pfx", ".log"]:
        assert required in EXCLUDE_EXTENSIONS


def test_exclude_filenames_integrity():
    for required in ["id_rsa", "id_ed25519", ".env", ".DS_Store", "Thumbs.db"]:
        assert required in EXCLUDE_FILENAMES
