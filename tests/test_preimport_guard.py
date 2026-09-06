from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from aether_context.guard import (
    GuardFailure,
    _audit_import_roots,
    _audit_site_import_surface,
    _canonical,
    _verified_manifest_payload,
    _verify_ed25519,
)


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "aether_context" / "guard.py"


def _lock_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _signed_files(tmp_path, payload):
    key = Ed25519PrivateKey.generate()
    key_id = "runtime-build-test"
    signature = key.sign(_canonical({"key_id": key_id, "payload": payload}))
    envelope = {
        "payload": payload,
        "key_id": key_id,
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    manifest = tmp_path / "manifest.json"
    keys = tmp_path / "keys.json"
    manifest.write_text(json.dumps(envelope), encoding="utf-8")
    keys.write_text(
        json.dumps({key_id: base64.b64encode(public).decode("ascii")}),
        encoding="utf-8",
    )
    return manifest, keys, envelope


def test_manifest_is_authenticated_before_any_payload_field_is_trusted(tmp_path):
    payload = {
        "schema_version": "ContextRuntimeBuildManifestV3",
        "runtime_environment": {"dependency_import_closure": ["not-trusted-yet"]},
    }
    manifest, keys, envelope = _signed_files(tmp_path, payload)
    assert _verified_manifest_payload(manifest, keys) == payload

    envelope["payload"]["runtime_environment"]["dependency_import_closure"] = []
    manifest.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(GuardFailure, match="context_guard_manifest_signature_invalid"):
        _verified_manifest_payload(manifest, keys)


def test_manifest_signature_verifier_rejects_noncanonical_and_small_order_keys():
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    message = b"guard-bootstrap-vector"
    signature = key.sign(message)
    _verify_ed25519(public, signature, message)
    with pytest.raises(GuardFailure, match="context_guard_manifest_signature_invalid"):
        _verify_ed25519(public, signature, message + b"-tampered")
    with pytest.raises(GuardFailure, match="context_guard_manifest_signature_invalid"):
        _verify_ed25519(b"\x01" + (b"\x00" * 31), b"\x00" * 64, message)


@pytest.mark.parametrize(
    ("distribution", "import_root"),
    (("aether-context", "aether_context"), ("pydantic", "pydantic")),
)
def test_importable_timestamp_pyc_marker_is_rejected_in_isolated_subprocess(
    tmp_path, distribution, import_root
):
    site = tmp_path / "site-packages"
    package = site / import_root
    package.mkdir(parents=True)
    source = package / "__init__.py"
    benign = "VALUE = 'SAFE'\n"
    malicious = "VALUE = 'PWN!'\n"
    assert len(benign) == len(malicious)
    timestamp = 1_700_000_000
    source.write_text(malicious, encoding="utf-8")
    os.utime(source, (timestamp, timestamp))
    py_compile.compile(
        str(source),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    source.write_text(benign, encoding="utf-8")
    os.utime(source, (timestamp, timestamp))
    _lock_tree(site)

    imported = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import sys;sys.path.append(sys.argv[1]);"
            "print(__import__(sys.argv[2]).VALUE)",
            str(site),
            import_root,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert imported.stdout.strip() == "PWN!"

    records = json.dumps({f"{import_root}/__init__.py": None})
    audit_code = (
        "import json,os,runpy,sys;"
        "g=runpy.run_path(sys.argv[1],run_name='guard_test');"
        "audit=g['_audit_import_roots'];failure=g['GuardFailure'];"
        "site=__import__('pathlib').Path(sys.argv[2]);"
        "\ntry:audit(site,sys.argv[3],json.loads(sys.argv[4]),owner_uid=os.stat(site).st_uid)"
        "\nexcept failure as exc:print(str(exc));sys.exit(0)"
        "\nsys.exit(9)"
    )
    guarded = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            audit_code,
            str(GUARD),
            str(site),
            distribution,
            records,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert guarded.returncode == 0, guarded.stderr
    assert guarded.stdout.strip() == "context_guard_bytecode_present"


def test_extension_shadow_and_unlocked_site_module_are_rejected(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    module = site / "typing_extensions.py"
    module.write_text("VALUE = 'safe'\n", encoding="utf-8")
    shadow = site / "typing_extensions.so"
    shadow.write_bytes(b"native-shadow")
    module.chmod(0o444)
    shadow.chmod(0o444)
    owner = site.stat().st_uid
    with pytest.raises(GuardFailure, match="context_guard_untracked_import_artifact"):
        _audit_import_roots(
            site,
            "typing_extensions",
            {"typing_extensions.py": None},
            owner_uid=owner,
        )

    shadow.chmod(0o666)
    shadow.unlink()
    unlocked = site / "sitecustomize.py"
    unlocked.write_text("VALUE = 'unlocked'\n", encoding="utf-8")
    unlocked.chmod(0o444)
    with pytest.raises(GuardFailure, match="context_guard_unlocked_import_artifact"):
        _audit_site_import_surface(
            site, ["typing_extensions"], owner_uid=owner
        )


@pytest.mark.parametrize(
    "relative",
    ("aether_context.so", "aether_context/__init__.so"),
)
def test_record_owned_native_package_collision_is_still_rejected(tmp_path, relative):
    site = tmp_path / "site-packages"
    package = site / "aether_context"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("MARKER = 'trusted-source'\n", encoding="utf-8")
    collision = site / relative
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"record-owned-native-marker")
    source.chmod(0o444)
    collision.chmod(0o444)
    package.chmod(0o555)
    records = {
        "aether_context/__init__.py": "sha256=placeholder",
        relative: "sha256=placeholder",
    }
    with pytest.raises(GuardFailure, match="context_guard_package_import_collision"):
        _audit_import_roots(
            site,
            "aether-context",
            records,
            owner_uid=site.stat().st_uid,
        )


def test_systemd_execs_guard_in_the_daemon_process_before_imports():
    unit = (ROOT / "deploy" / "aether-contextd.service").read_text(encoding="utf-8")
    exec_start = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    assert "python -I -S -B" in exec_start
    assert "/site-packages/aether_context/guard.py" in exec_start
    assert "--runtime-build-manifest ${CREDENTIALS_DIRECTORY}/" in exec_start
    assert "MemoryMax=512M" in unit
