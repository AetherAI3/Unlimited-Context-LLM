import argparse
import json
from pathlib import Path
import sqlite3

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from aether_context.contracts import ContextProfileV1, RuntimeEnvironmentV2, canonical, digest
from bench import hosted_scale


def test_bounded_harness_registers_durability_and_hydrates_v2(
    tmp_path, monkeypatch
):
    source_revision = "a" * 40
    dependencies = [
        "annotated-types",
        "cffi",
        "cryptography",
        "numpy",
        "pycparser",
        "pydantic",
        "pydantic_core",
        "typing-inspection",
        "typing_extensions",
    ]
    environment = RuntimeEnvironmentV2(
        python_implementation="cpython",
        python_version="3.13.0",
        python_abi="cp313-test",
        python_executable_digest=digest("python"),
        dependency_import_closure=dependencies,
        dependency_versions={name: "1.0" for name in dependencies},
        dependency_artifact_digests={name: digest(name) for name in dependencies},
        sqlite_version=sqlite3.sqlite_version,
        package_import_root_digest=digest("site-packages"),
    )
    package_digest = digest("package-tree")
    monkeypatch.setattr(hosted_scale, "_assert_module_origins", lambda _: None)
    monkeypatch.setattr(hosted_scale, "_git_source", lambda _: source_revision)
    monkeypatch.setattr(
        hosted_scale, "runtime_environment_identity", lambda: environment
    )
    monkeypatch.setattr(
        hosted_scale,
        "runtime_package_tree_digest",
        lambda **_: package_digest,
    )
    monkeypatch.setattr(
        "aether_context.engine.runtime_environment_identity",
        lambda **_: environment,
    )
    monkeypatch.setattr(
        "aether_context.engine.runtime_package_tree_digest",
        lambda **_: package_digest,
    )
    monkeypatch.setattr(
        hosted_scale,
        "run_namespace_isolation",
        lambda: {
            "cases": 1_000_000,
            "unauthorized_records": 0,
            "result_digest": digest("bounded-isolation"),
        },
    )

    profile = ContextProfileV1(
        index_version=sqlite3.sqlite_version,
        max_records=20,
        max_indexed_bytes=1024,
        max_record_bytes=256,
        max_operations_per_cycle=64,
        max_concurrent_cycles=1,
        disk_free_floor=0,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(canonical(profile.model_dump(mode="json")))
    records = []
    queries = []
    for index in range(20):
        token = f"unique{index:02d}"
        text = (token + " " + ("x" * 80))[:47]
        records.append({"id": f"record-{index}", "text": text})
        queries.append(
            {
                "id": f"query-{index}",
                "query": token,
                "expected_record_ids": [f"record-{index}"],
            }
        )
    dataset_path = tmp_path / "recall.json"
    dataset_path.write_bytes(
        canonical(
            {
                "schema_version": "HostedRecallDatasetV1",
                "name": "bounded-harness/v1",
                "records": records,
                "queries": queries,
            }
        )
    )
    signing_key = tmp_path / "benchmark.key"
    signing_key.write_bytes(Ed25519PrivateKey.generate().private_bytes_raw())
    output = tmp_path / "receipt.json"
    work = tmp_path / "work"
    args = argparse.Namespace(
        repository=Path(__file__).resolve().parents[1],
        profile=profile_path,
        recall_dataset=dataset_path,
        work_directory=work,
        signing_key=signing_key,
        key_id="bounded-benchmark",
        output=output,
        receipt_bound_profile_output=None,
        executor_stability_receipt=None,
        executor_stability_keys=None,
        data_plane_isolation_receipt=None,
        data_plane_isolation_keys=None,
        retrieval_samples=20,
        concurrency=1,
        validity_seconds=3600,
    )
    envelope = hosted_scale.run(args)
    assert json.loads(output.read_text()) == envelope
    assert envelope["payload"]["checkpoint_receipt_digest"]
    assert envelope["payload"]["hydrate_receipt_digest"]
    with sqlite3.connect(work / "replacement.sqlite3") as db:
        restored = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_legacy,"
            "checkpoint_durability_receipt FROM cycles"
        ).fetchone()
    assert restored[:2] == ("remote_registered", 0)
    assert restored[2] is not None
    args.receipt_bound_profile_output = tmp_path / "qualified-profile.json"
    with pytest.raises(SystemExit, match="independent executor and data-plane"):
        hosted_scale.run(args)
    assert not args.receipt_bound_profile_output.exists()
