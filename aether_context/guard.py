"""Stdlib-only pre-import guard for the production context daemon."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import platform
import re
import sqlite3
import stat
import sys
import sysconfig
from email.parser import Parser
from pathlib import Path, PurePosixPath


_IMPORT_ROOTS = {
    "aether-context": ("aether_context",),
    "annotated-types": ("annotated_types",),
    "bcrypt": ("bcrypt",),
    "cffi": ("cffi", "_cffi_backend"),
    "cryptography": ("cryptography",),
    "numpy": ("numpy",),
    "pydantic": ("pydantic",),
    "pydantic_core": ("pydantic_core",),
    "pycparser": ("pycparser",),
    "typing-inspection": ("typing_inspection",),
    "typing_extensions": ("typing_extensions",),
}
_REQUIRED_DEPENDENCY_CLOSURE = {
    "annotated-types",
    "cffi",
    "cryptography",
    "numpy",
    "pydantic",
    "pydantic_core",
    "pycparser",
    "typing-inspection",
    "typing_extensions",
}
_OPTIONAL_IMPORT_DISTRIBUTIONS = {"bcrypt"}
_IMPORTABLE_SUFFIXES = {".py", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,199}$")
_ED25519_Q = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493
_ED25519_D = (-121665 * pow(121666, _ED25519_Q - 2, _ED25519_Q)) % _ED25519_Q
_ED25519_I = pow(2, (_ED25519_Q - 1) // 4, _ED25519_Q)
_ED25519_IDENTITY = (0, 1, 1, 0)


class GuardFailure(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    def validate(item: object) -> None:
        if item is None or isinstance(item, (bool, str)):
            return
        if isinstance(item, int) and abs(item) <= 9007199254740991:
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(
            isinstance(key, str) and key.isascii() for key in item
        ):
            for child in item.values():
                validate(child)
            return
        raise GuardFailure("context_guard_manifest_invalid")

    validate(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normalized(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _immutable(path: Path, *, owner_uid: int) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise GuardFailure("context_guard_path_unavailable") from exc
    if path.is_symlink() or info.st_uid != owner_uid or stat.S_IMODE(info.st_mode) & 0o022:
        raise GuardFailure("context_guard_path_mutable")


def _secure_path(
    raw: str | Path, *, owner_uid: int, trust_root: str | Path = "/"
) -> Path:
    """Resolve a path only after every existing component passes lstat policy."""

    path = Path(raw).absolute()
    root = Path(trust_root).absolute()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GuardFailure("context_guard_path_outside_root") from exc
    chain = [path]
    while chain[-1] != root:
        parent = chain[-1].parent
        if parent == chain[-1]:
            raise GuardFailure("context_guard_path_outside_root")
        chain.append(parent)
    for component in reversed(chain):
        _immutable(component, owner_uid=owner_uid)
    return path.resolve(strict=True)


def _json_no_duplicates(raw: str) -> object:
    def closed_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise GuardFailure("context_guard_manifest_invalid")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=closed_object)
    except (TypeError, ValueError) as exc:
        raise GuardFailure("context_guard_manifest_invalid") from exc


def _edwards_add(first, second):
    x1, y1, z1, t1 = first
    x2, y2, z2, t2 = second
    a = ((y1 - x1) * (y2 - x2)) % _ED25519_Q
    b = ((y1 + x1) * (y2 + x2)) % _ED25519_Q
    c = (2 * _ED25519_D * t1 * t2) % _ED25519_Q
    d = (2 * z1 * z2) % _ED25519_Q
    e, f, g, h = b - a, d - c, d + c, b + a
    return (
        (e * f) % _ED25519_Q,
        (g * h) % _ED25519_Q,
        (f * g) % _ED25519_Q,
        (e * h) % _ED25519_Q,
    )


def _edwards_multiply(point, scalar: int):
    result = _ED25519_IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        scalar >>= 1
    return result


def _recover_x(y: int) -> int:
    yy = y * y % _ED25519_Q
    xx = (yy - 1) * pow(_ED25519_D * yy + 1, _ED25519_Q - 2, _ED25519_Q)
    x = pow(xx, (_ED25519_Q + 3) // 8, _ED25519_Q)
    if (x * x - xx) % _ED25519_Q:
        x = x * _ED25519_I % _ED25519_Q
    if (x * x - xx) % _ED25519_Q:
        raise GuardFailure("context_guard_manifest_signature_invalid")
    return x


def _decode_ed25519_point(raw: bytes):
    if len(raw) != 32:
        raise GuardFailure("context_guard_manifest_signature_invalid")
    encoded = int.from_bytes(raw, "little")
    sign = encoded >> 255
    y = encoded & ((1 << 255) - 1)
    if y >= _ED25519_Q:
        raise GuardFailure("context_guard_manifest_signature_invalid")
    x = _recover_x(y)
    if (x & 1) != sign:
        x = _ED25519_Q - x
    if x == 0 and sign:
        raise GuardFailure("context_guard_manifest_signature_invalid")
    if (-x * x + y * y - 1 - _ED25519_D * x * x * y * y) % _ED25519_Q:
        raise GuardFailure("context_guard_manifest_signature_invalid")
    return (x, y, 1, x * y % _ED25519_Q)


def _same_point(first, second) -> bool:
    return (
        (first[0] * second[2] - second[0] * first[2]) % _ED25519_Q == 0
        and (first[1] * second[2] - second[1] * first[2]) % _ED25519_Q == 0
    )


_ED25519_BASE = _decode_ed25519_point(
    bytes.fromhex("5866666666666666666666666666666666666666666666666666666666666666")
)


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    if len(public_key) != 32 or len(signature) != 64:
        raise GuardFailure("context_guard_manifest_signature_invalid")
    public_point = _decode_ed25519_point(public_key)
    encoded_r, encoded_s = signature[:32], signature[32:]
    r_point = _decode_ed25519_point(encoded_r)
    scalar = int.from_bytes(encoded_s, "little")
    if scalar >= _ED25519_L or _same_point(public_point, _ED25519_IDENTITY):
        raise GuardFailure("context_guard_manifest_signature_invalid")
    if not _same_point(
        _edwards_multiply(public_point, _ED25519_L), _ED25519_IDENTITY
    ) or not _same_point(
        _edwards_multiply(r_point, _ED25519_L), _ED25519_IDENTITY
    ):
        raise GuardFailure("context_guard_manifest_signature_invalid")
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_key + message).digest(), "little"
    ) % _ED25519_L
    if not _same_point(
        _edwards_multiply(_ED25519_BASE, scalar),
        _edwards_add(r_point, _edwards_multiply(public_point, challenge)),
    ):
        raise GuardFailure("context_guard_manifest_signature_invalid")


def _verified_manifest_payload(manifest_file: Path, key_file: Path) -> dict:
    if manifest_file.stat().st_size > 4_000_000 or key_file.stat().st_size > 64_000:
        raise GuardFailure("context_guard_manifest_invalid")
    envelope = _json_no_duplicates(manifest_file.read_text(encoding="utf-8"))
    keys = _json_no_duplicates(key_file.read_text(encoding="utf-8"))
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"payload", "key_id", "signature"}
        or not isinstance(envelope.get("payload"), dict)
        or not isinstance(envelope.get("key_id"), str)
        or _IDENT.fullmatch(envelope["key_id"]) is None
        or not isinstance(envelope.get("signature"), str)
        or not isinstance(keys, dict)
        or not keys
        or len(keys) > 32
        or any(
            not isinstance(key_id, str)
            or _IDENT.fullmatch(key_id) is None
            or not isinstance(value, str)
            for key_id, value in keys.items()
        )
        or envelope["key_id"] not in keys
    ):
        raise GuardFailure("context_guard_manifest_invalid")
    try:
        public_key = base64.b64decode(keys[envelope["key_id"]], validate=True)
        signature = base64.b64decode(envelope["signature"], validate=True)
    except (ValueError, TypeError) as exc:
        raise GuardFailure("context_guard_manifest_signature_invalid") from exc
    if (
        base64.b64encode(public_key).decode("ascii") != keys[envelope["key_id"]]
        or base64.b64encode(signature).decode("ascii") != envelope["signature"]
    ):
        raise GuardFailure("context_guard_manifest_signature_invalid")
    _verify_ed25519(
        public_key,
        signature,
        _canonical({"key_id": envelope["key_id"], "payload": envelope["payload"]}),
    )
    return envelope["payload"]


def _distribution(site: Path, name: str, *, owner_uid: int):
    matches = []
    for metadata_dir in site.glob("*.dist-info"):
        metadata_path = metadata_dir / "METADATA"
        try:
            _immutable(metadata_dir, owner_uid=owner_uid)
            _immutable(metadata_path, owner_uid=owner_uid)
            metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if _normalized(metadata.get("Name", "")) == _normalized(name):
            matches.append((metadata_dir, metadata))
    if len(matches) != 1:
        raise GuardFailure("context_guard_distribution_ambiguous")
    metadata_dir, metadata = matches[0]
    _immutable(metadata_dir, owner_uid=owner_uid)
    record_path = metadata_dir / "RECORD"
    _immutable(record_path, owner_uid=owner_uid)
    records: dict[str, str | None] = {}
    try:
        with record_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) != 3 or row[0] in records:
                    raise GuardFailure("context_guard_record_invalid")
                records[row[0].replace("\\", "/")] = row[1] or None
    except OSError as exc:
        raise GuardFailure("context_guard_record_invalid") from exc
    return metadata, records


def _record_artifact_digest(
    site: Path, records: dict[str, str | None], *, owner_uid: int
) -> str:
    entries = []
    for relative in sorted(records):
        record_path = PurePosixPath(relative)
        if record_path.is_absolute() or not record_path.parts:
            raise GuardFailure("context_guard_record_invalid")
        path = site
        for part in record_path.parts:
            if part == ".":
                continue
            path = path.parent if part == ".." else path / part
            _immutable(path, owner_uid=owner_uid)
        path = path.resolve(strict=True)
        venv = Path(sys.executable).absolute().parent.parent
        try:
            path.relative_to(venv)
        except ValueError as exc:
            raise GuardFailure("context_guard_record_invalid") from exc
        if not path.is_file():
            raise GuardFailure("context_guard_record_invalid")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = records[relative]
        if recorded is not None:
            try:
                algorithm, expected = recorded.split("=", 1)
                observed = base64.urlsafe_b64encode(
                    hashlib.new(algorithm, path.read_bytes()).digest()
                ).rstrip(b"=").decode()
            except (ValueError, OSError) as exc:
                raise GuardFailure("context_guard_record_invalid") from exc
            if observed != expected:
                raise GuardFailure("context_guard_record_mismatch")
        entries.append({"path": relative, "sha256": sha256})
    return _digest(entries)


def _audit_site_import_surface(site: Path, closure: list[str], *, owner_uid: int) -> None:
    """Reject executable top-level paths outside the signed runtime closure."""

    allowed_roots = {
        root
        for distribution in ("aether-context", *closure)
        for root in _IMPORT_ROOTS[distribution]
    }
    for path in site.iterdir():
        _immutable(path, owner_uid=owner_uid)
        name = path.name
        if name == "__pycache__" or path.suffix.lower() in {".pyc", ".pyo", ".pth"}:
            raise GuardFailure("context_guard_unlocked_import_artifact")
        if path.is_dir():
            if name.isidentifier() and name not in allowed_roots:
                raise GuardFailure("context_guard_unlocked_import_artifact")
            continue
        root = name.split(".", 1)[0]
        if (
            path.suffix.lower() in _IMPORTABLE_SUFFIXES
            and root.isidentifier()
            and root not in allowed_roots
        ):
            raise GuardFailure("context_guard_unlocked_import_artifact")


def _import_paths(site: Path, distribution: str) -> list[Path]:
    roots = _IMPORT_ROOTS.get(distribution)
    if roots is None:
        raise GuardFailure("context_guard_import_closure_unknown")
    values: list[Path] = []
    for root in roots:
        candidates = [site / root, *sorted(site.glob(root + ".*"))]
        values.extend(
            item
            for item in candidates
            if item.is_dir()
            or (item.is_file() and item.suffix.lower() in _IMPORTABLE_SUFFIXES)
        )
    if not values:
        raise GuardFailure("context_guard_import_root_missing")
    return values


def _audit_import_roots(
    site: Path,
    distribution: str,
    records: dict[str, str | None],
    *,
    owner_uid: int,
) -> None:
    roots = _import_paths(site, distribution)
    if distribution == "aether-context":
        expected = site / "aether_context"
        if roots != [expected] or not expected.is_dir():
            raise GuardFailure("context_guard_package_import_collision")
    for root in roots:
        candidates = [root] if root.is_file() else [root, *root.rglob("*")]
        for path in candidates:
            _immutable(path, owner_uid=owner_uid)
            if path.is_dir():
                if path.name == "__pycache__":
                    raise GuardFailure("context_guard_bytecode_present")
                continue
            relative = path.relative_to(site).as_posix()
            if path.suffix.lower() in {".pyc", ".pyo"}:
                raise GuardFailure("context_guard_bytecode_present")
            if distribution == "aether-context" and path.suffix.lower() in {
                ".so",
                ".pyd",
                ".dll",
                ".dylib",
            }:
                raise GuardFailure("context_guard_package_import_collision")
            if relative not in records:
                raise GuardFailure("context_guard_untracked_import_artifact")
            if path.suffix.lower() in {".so", ".pyd", ".dll", ".dylib"} and not records[
                relative
            ]:
                raise GuardFailure("context_guard_unpinned_native_artifact")


def verify_environment(
    site_packages: str | Path,
    manifest_path: str | Path,
    manifest_keys_path: str | Path,
    *,
    owner_uid: int = 0,
    trust_root: str | Path = "/",
) -> None:
    site = _secure_path(
        site_packages, owner_uid=owner_uid, trust_root=trust_root
    )
    manifest_file = _secure_path(
        manifest_path, owner_uid=owner_uid, trust_root=trust_root
    )
    key_file = _secure_path(
        manifest_keys_path, owner_uid=owner_uid, trust_root=trust_root
    )
    _secure_path(sys.executable, owner_uid=owner_uid, trust_root=trust_root)
    try:
        payload = _verified_manifest_payload(manifest_file, key_file)
        environment = payload["runtime_environment"]
        closure = environment["dependency_import_closure"]
    except (OSError, ValueError, KeyError, TypeError, GuardFailure) as exc:
        if isinstance(exc, GuardFailure):
            raise
        raise GuardFailure("context_guard_manifest_invalid") from exc
    if (
        payload.get("schema_version") != "ContextRuntimeBuildManifestV3"
        or environment.get("schema_version") != "ContextRuntimeEnvironmentV2"
        or not _REQUIRED_DEPENDENCY_CLOSURE.issubset(set(closure))
        or not set(closure).issubset(set(_IMPORT_ROOTS) - {"aether-context"})
        or closure != sorted(set(closure))
        or set(environment.get("dependency_versions", {})) != set(closure)
        or set(environment.get("dependency_artifact_digests", {})) != set(closure)
    ):
        raise GuardFailure("context_guard_manifest_invalid")
    _audit_site_import_surface(site, closure, owner_uid=owner_uid)
    for optional in _OPTIONAL_IMPORT_DISTRIBUTIONS - set(closure):
        if any(
            candidate.exists()
            for root in _IMPORT_ROOTS[optional]
            for candidate in (site / root, *site.glob(root + ".*"))
        ):
            raise GuardFailure("context_guard_unlocked_optional_import")
    artifacts = {}
    versions = {}
    for name in ("aether-context", *closure):
        metadata, records = _distribution(site, name, owner_uid=owner_uid)
        _audit_import_roots(site, name, records, owner_uid=owner_uid)
        artifact_digest = _record_artifact_digest(site, records, owner_uid=owner_uid)
        if name != "aether-context":
            versions[name] = metadata.get("Version")
            artifacts[name] = artifact_digest
    package_root = (site / "aether_context").resolve(strict=True)
    package_entries = []
    for path in sorted(package_root.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix in {".py", ".json", ".typed"}:
            package_entries.append(
                {
                    "path": path.relative_to(package_root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    abi = sysconfig.get_config_var("SOABI") or getattr(
        sys.implementation, "cache_tag", None
    )
    observed_environment = {
        "schema_version": "ContextRuntimeEnvironmentV2",
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_abi": abi,
        "python_executable_digest": hashlib.sha256(
            Path(sys.executable).resolve(strict=True).read_bytes()
        ).hexdigest(),
        "dependency_import_closure": closure,
        "dependency_versions": versions,
        "dependency_artifact_digests": artifacts,
        "sqlite_version": sqlite3.sqlite_version,
        "package_import_root_digest": _digest(str(package_root.parent)),
        "bytecode_policy": "preimport-guard/source-only/v2",
    }
    if (
        payload.get("package_tree_digest") != _digest(package_entries)
        or environment != observed_environment
        or payload.get("locked_environment_digest") != _digest(observed_environment)
    ):
        raise GuardFailure("context_guard_runtime_mismatch")


def _site_packages() -> Path:
    guard = Path(__file__).absolute()
    if guard.parent.name != "aether_context":
        raise GuardFailure("context_guard_site_packages_ambiguous")
    return guard.parent.parent


def _argument(name: str) -> str:
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except (ValueError, IndexError):
        raise GuardFailure("context_guard_argument_required") from None
    if not value or value.startswith("--"):
        raise GuardFailure("context_guard_argument_required")
    return value


def main() -> None:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise SystemExit("context_guard_flags_required")
    try:
        site = _site_packages()
        credentials = Path(_argument("--credentials"))
        manifest_path = _argument("--runtime-build-manifest")
        verify_environment(
            site,
            manifest_path,
            credentials / "context-runtime-build-keys.json",
        )
    except GuardFailure as exc:
        raise SystemExit(str(exc)) from None
    sys.path.append(str(site.resolve(strict=True)))
    from aether_context.service import main as service_main

    service_main()


if __name__ == "__main__":
    main()
