"""Produce signed, measured evidence for one exact hosted context profile.

This intentionally expensive harness exercises the real append, retrieve,
checkpoint, replacement-host hydrate, and startup-audit paths. It never enables
a reach claim by itself: the deployed profile must point at the resulting signed
envelope and the daemon must trust its independent benchmark key.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

import aether_context
from aether_context.contracts import (
    AppendRequestV1,
    CapabilityV1,
    ContextBindingV1,
    ContextProfileV1,
    ExecutorStabilityReceiptV1,
    HostedScaleReceiptV1,
    NamespaceV1,
    RetrieveRequestV1,
    canonical,
    digest,
)
from aether_context.crypto import (
    EnvelopeCipher,
    ReceiptSigner,
    require_disjoint_keys,
    verify,
)
from aether_context.engine import ContextEngine
from aether_context.retention import FileCycleKeyProvider
from aether_context.scale import profile_benchmark_digest, run_namespace_isolation


def _percentile(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile / 100) - 1)]


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise SystemExit("unable to measure peak process RSS")
        return max(1, int(counters.PeakWorkingSetSize))
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max(1, int(value if os.uname().sysname == "Darwin" else value * 1024))


def _git_source(repository: Path) -> str:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    revision = git("rev-parse", "HEAD")
    if len(revision) != 40 or git("status", "--porcelain=v1", "--untracked-files=all"):
        raise SystemExit("benchmark source must be an exact clean Git HEAD")
    submodules = git("submodule", "status", "--recursive")
    if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
        raise SystemExit("benchmark submodules must match the committed Git HEAD")
    return revision


def _assert_module_origins(repository: Path) -> None:
    origins = {
        "benchmark harness": Path(__file__).resolve(),
        "aether_context package": Path(aether_context.__file__).resolve(),
        "ContextEngine": Path(inspect.getfile(ContextEngine)).resolve(),
    }
    outside = [name for name, path in origins.items() if repository not in path.parents]
    if outside:
        raise SystemExit(
            "benchmark imported code outside the verified repository: " + ", ".join(outside)
        )


def _recall_dataset(path: Path, max_record_bytes: int) -> tuple[dict, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "name",
        "records",
        "queries",
    }:
        raise SystemExit("recall dataset has unknown or missing top-level fields")
    if raw["schema_version"] != "HostedRecallDatasetV1":
        raise SystemExit("recall dataset schema is unsupported")
    if not isinstance(raw["name"], str) or not 1 <= len(raw["name"]) <= 100:
        raise SystemExit("recall dataset name is invalid")
    if not isinstance(raw["records"], list) or not isinstance(raw["queries"], list):
        raise SystemExit("recall dataset records and queries must be arrays")
    record_ids: set[str] = set()
    for record in raw["records"]:
        if not isinstance(record, dict) or set(record) != {"id", "text"} or not all(
            isinstance(record[field], str) for field in ("id", "text")
        ):
            raise SystemExit("recall dataset record is invalid")
        if (
            not record["id"]
            or not record["text"]
            or record["id"] in record_ids
            or len(record["text"].encode()) > max_record_bytes
        ):
            raise SystemExit("recall dataset record is duplicate or too large")
        record_ids.add(record["id"])
    if len(raw["queries"]) < 20:
        raise SystemExit("recall dataset requires at least twenty project-specific queries")
    query_ids: set[str] = set()
    for query in raw["queries"]:
        if not isinstance(query, dict) or set(query) != {
            "id",
            "query",
            "expected_record_ids",
        }:
            raise SystemExit("recall dataset query is invalid")
        expected = query["expected_record_ids"]
        if (
            not isinstance(query["id"], str)
            or not isinstance(query["query"], str)
            or not query["query"]
            or query["id"] in query_ids
            or not isinstance(expected, list)
            or not expected
            or any(not isinstance(item, str) or item not in record_ids for item in expected)
        ):
            raise SystemExit("recall dataset query reference is invalid")
        query_ids.add(query["id"])
    canonical(raw)
    return raw, digest(raw)


def _public_keys(path: Path) -> dict[str, Ed25519PublicKey]:
    return {
        key: Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
    }


def run(args: argparse.Namespace) -> dict:
    repository = args.repository.resolve()
    _assert_module_origins(repository)
    protected_outputs = [args.work_directory, args.output]
    if args.receipt_bound_profile_output:
        protected_outputs.append(args.receipt_bound_profile_output)
    if any(repository == path.resolve() or repository in path.resolve().parents for path in protected_outputs):
        raise SystemExit("benchmark work and output paths must be outside the source repository")
    source_revision = _git_source(repository)
    target_payload = json.loads(args.profile.read_text(encoding="utf-8"))
    if target_payload.get("name") == "hosted-v1" and not target_payload.get(
        "benchmark_receipt"
    ):
        target_payload["benchmark_receipt"] = "0" * 64
    target = ContextProfileV1.model_validate(target_payload)
    if target.index_version != sqlite3.sqlite_version:
        raise SystemExit("profile index_version does not match this benchmark runtime")
    # Admission requires evidence, so measurement uses the same limits and
    # implementation under the explicitly non-production canary profile name.
    measured = target.model_copy(
        update={"name": "bounded-canary/v1", "benchmark_receipt": None}
    )
    dataset, dataset_digest = _recall_dataset(
        args.recall_dataset, measured.max_record_bytes
    )
    args.work_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    database = args.work_directory / "measured.sqlite3"
    replacement_database = args.work_directory / "replacement.sqlite3"
    if database.exists() or replacement_database.exists():
        raise SystemExit("benchmark work directory must not contain prior databases")

    authority = ReceiptSigner("benchmark-gateway", Ed25519PrivateKey.generate())
    proof = ReceiptSigner("benchmark-proof", Ed25519PrivateKey.generate())
    service = ReceiptSigner("benchmark-contextd", Ed25519PrivateKey.generate())
    wrapper = EnvelopeCipher(os.urandom(32), domain=b"benchmark-cycle-wrapper/v1")
    provider = FileCycleKeyProvider(args.work_directory / "cycle-keys", wrapper)
    clock = [int(time.time())]

    def make(path: Path) -> ContextEngine:
        return ContextEngine(
            path,
            measured,
            EnvelopeCipher(os.urandom(32)),
            service,
            {authority.key_id: authority.key.public_key()},
            {proof.key_id: proof.key.public_key()},
            clock=lambda: clock[0],
            cycle_keys=provider,
        )

    started_at = int(time.time())
    engine = make(database)
    reservation = engine.reserve(
        authority.sign(
            {
                "operation": "reserve",
                "owner_id": "benchmark-owner",
                "project_id": "benchmark-project",
                "idempotency_key": "hosted-scale",
                "request_digest": digest("hosted-scale"),
                "profile_digest": digest(measured),
                "expires_at": clock[0] + 3600,
            }
        )
    )
    cycle = reservation["payload"]["context_cycle_id"]
    namespace = NamespaceV1(
        owner_id="benchmark-owner",
        project_id="benchmark-project",
        objective_id="benchmark-objective",
        context_cycle_id=cycle,
    )
    authority_payload = {"objective": "measure hosted context", "paths": ["bench"]}
    project_snapshot = {
        "project_id": "benchmark-project",
        "graph_id": "benchmark-graph",
        "policy_digest": digest("benchmark-policy"),
        "nodes": [],
    }
    binding = ContextBindingV1(
        namespace=namespace,
        context_bucket_id=reservation["payload"]["context_bucket_id"],
        repository_id="benchmark-repository",
        repo_main_sha=source_revision,
        project_graph_id="benchmark-graph",
        graph_revision=1,
        graph_checksum=digest(project_snapshot),
        plan_digest=digest("benchmark-plan"),
        execution_profile_digest=digest("benchmark-execution-profile"),
        shared_ir_digest=digest("benchmark-ir"),
        authorization_receipt_ref="benchmark-authority",
        authorization_digest=digest(authority_payload),
        policy_digest=digest("benchmark-policy"),
        redaction_digest=digest("benchmark-redaction"),
        profile_digest=digest(measured),
        captain_binding_id="benchmark-captain",
        created_at=clock[0],
        expires_at=clock[0] + 604800,
    )
    engine.bind(authority.sign(binding.model_dump()), authority_payload, project_snapshot)

    def capability(role: str = "worker", fence: int = 1) -> dict:
        return authority.sign(
            CapabilityV1(
                namespace=namespace,
                principal_id="benchmark-worker",
                lane_id="benchmark-lane",
                task_id="benchmark-task",
                role=role,
                operations=["append", "retrieve", "checkpoint", "status"],
                source_class="repository_verified",
                binding_digest=digest(binding),
                policy_digest=binding.policy_digest,
                profile_digest=digest(measured),
                fence=fence,
                expires_at=clock[0] + 604000,
                max_bytes=measured.max_record_bytes,
                max_tokens=measured.max_capsule_tokens,
            ).model_dump()
        )

    cap = capability()
    target_records = max(math.ceil(measured.max_records * 0.9), len(dataset["records"]))
    target_bytes = math.ceil(measured.max_indexed_bytes * 0.9)
    if target_records > measured.max_records:
        raise SystemExit("recall dataset exceeds the configured record capacity")
    if target_records + args.retrieval_samples + 2 > measured.max_operations_per_cycle:
        raise SystemExit("profile operation quota cannot measure its declared record capacity")
    dataset_bytes = sum(len(record["text"].encode()) for record in dataset["records"])
    remaining_records = target_records - len(dataset["records"])
    if dataset_bytes > measured.max_indexed_bytes:
        raise SystemExit("recall dataset exceeds the configured byte capacity")
    if remaining_records == 0 and dataset_bytes < target_bytes:
        raise SystemExit("recall dataset leaves no records with which to measure byte capacity")
    filler_bytes = (
        math.ceil(max(0, target_bytes - dataset_bytes) / remaining_records)
        if remaining_records
        else 0
    )
    if filler_bytes > measured.max_record_bytes:
        raise SystemExit("profile byte capacity cannot fit within its record limits")
    appended_count = 0
    dataset_record_ids: dict[str, str] = {}

    def append(identifier: str, text: str) -> str:
        nonlocal appended_count
        if appended_count and appended_count % max(1, measured.max_calls_per_minute - 1) == 0:
            clock[0] += 60
        result = engine.append(
            cap,
            AppendRequestV1(
                idempotency_key=f"measure-{identifier}",
                text=text,
                action_ref=f"measure-{identifier}",
            ),
        )
        appended_count += 1
        return result["payload"]["record_id"]

    for index, record in enumerate(dataset["records"]):
        dataset_record_ids[record["id"]] = append(f"dataset-{index}", record["text"])
    for index in range(remaining_records):
        append(f"capacity-{index}", "x" * max(1, filler_bytes))

    sample_count = args.retrieval_samples
    if not 20 <= sample_count <= len(dataset["queries"]):
        raise SystemExit("retrieval samples must select 20 or more available dataset queries")
    selected = dataset["queries"][:sample_count]

    def retrieve(item: tuple[int, dict]) -> tuple[int, bool]:
        index, query = item
        began = time.perf_counter_ns()
        result = engine.retrieve(
            cap,
            RetrieveRequestV1(
                turn_id="dataset-" + query["id"],
                query=query["query"],
                native_window=max(1024, measured.max_capsule_tokens * 8),
                remaining_prompt_budget=measured.max_capsule_tokens,
            ),
        )["payload"]
        elapsed = math.ceil((time.perf_counter_ns() - began) / 1_000_000)
        found_ids = {entry["metadata"]["record_id"] for entry in result["entries"]}
        expected_ids = {
            dataset_record_ids[record_id] for record_id in query["expected_record_ids"]
        }
        found = expected_ids <= found_ids
        return elapsed, found

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        samples = list(executor.map(retrieve, enumerate(selected)))
    latencies = [sample[0] for sample in samples]
    recall = sum(1 for _, found in samples if found)

    checkpoint_started = time.perf_counter_ns()
    checkpoint = engine.checkpoint(capability("durability"), "hosted-scale-checkpoint")
    checkpoint_ms = math.ceil((time.perf_counter_ns() - checkpoint_started) / 1_000_000)
    receipt_payload = checkpoint["receipt"]["payload"]
    grant = authority.sign(
        {
            "operation": "hydrate",
            "namespace": namespace.model_dump(),
            "manifest_checksum": receipt_payload["manifest_checksum"],
            "key_ref": receipt_payload["key_ref"],
            "key_provider": receipt_payload["key_provider"],
            "key_version": receipt_payload["key_version"],
            "fence": 2,
            "expires_at": clock[0] + 3600,
        }
    )
    replacement = make(replacement_database)
    hydrate_started = time.perf_counter_ns()
    hydrate_receipt = replacement.hydrate(
        grant,
        checkpoint["receipt"],
        base64.b64decode(checkpoint["pack"], validate=True),
        {service.key_id: service.key.public_key()},
    )
    hydrate_ms = math.ceil((time.perf_counter_ns() - hydrate_started) / 1_000_000)
    rebuild_started = time.perf_counter_ns()
    rebuild_receipt = replacement.rebuild_index(
        capability("durability", fence=2), "hosted-scale-index-rebuild"
    )
    rebuild_ms = math.ceil((time.perf_counter_ns() - rebuild_started) / 1_000_000)
    isolation = run_namespace_isolation()
    with engine.store.connection() as db:
        indexed = db.execute(
            "SELECT cursor,bytes FROM cycles WHERE id=?", (cycle,)
        ).fetchone()

    finished_at = int(time.time())
    if _git_source(repository) != source_revision:
        raise SystemExit("source revision changed during benchmark")
    benchmark_signer = ReceiptSigner(
        args.key_id, Ed25519PrivateKey.from_private_bytes(args.signing_key.read_bytes())
    )
    stability_envelope = None
    executor_stable = False
    if args.executor_stability_receipt:
        stability_envelope = json.loads(
            args.executor_stability_receipt.read_text(encoding="utf-8")
        )
        stability_keys = _public_keys(args.executor_stability_keys)
        require_disjoint_keys(
            {benchmark_signer.key_id: benchmark_signer.key.public_key()}, stability_keys
        )
        stability = ExecutorStabilityReceiptV1.model_validate(
            verify(stability_envelope, stability_keys)
        )
        if (
            stability.source_revision != source_revision
            or stability.profile_benchmark_digest != profile_benchmark_digest(target)
            or stability.target_concurrency != args.concurrency
            or not stability.issued_at <= finished_at < stability.expires_at
        ):
            raise SystemExit("executor stability receipt does not bind this benchmark")
        executor_stable = True
    payload = HostedScaleReceiptV1(
        profile_benchmark_digest=profile_benchmark_digest(target),
        source_revision=source_revision,
        source_tree_clean=True,
        index_implementation=target.index_implementation,
        index_version=target.index_version,
        embedding_version=target.embedding_version,
        storage_version=target.storage_version,
        configured_max_records=target.max_records,
        configured_max_indexed_bytes=target.max_indexed_bytes,
        indexed_records=indexed["cursor"],
        indexed_bytes=indexed["bytes"],
        reachable_tokens=indexed["bytes"] // 4,
        resident_peak_bytes=_peak_rss_bytes(),
        retrieval_samples=sample_count,
        target_concurrency=args.concurrency,
        retrieval_p50_ms=_percentile(latencies, 50),
        retrieval_p95_ms=_percentile(latencies, 95),
        retrieval_p99_ms=_percentile(latencies, 99),
        checkpoint_ms=checkpoint_ms,
        checkpoint_receipt_digest=digest(checkpoint["receipt"]),
        hydrate_ms=hydrate_ms,
        hydrate_receipt_digest=digest(hydrate_receipt),
        rebuild_ms=rebuild_ms,
        rebuild_receipt_digest=digest(rebuild_receipt),
        recall_dataset_name=dataset["name"],
        recall_dataset_digest=dataset_digest,
        recall_basis_points=recall * 10000 // sample_count,
        isolation_cases=isolation["cases"],
        unauthorized_records=isolation["unauthorized_records"],
        isolation_result_digest=isolation["result_digest"],
        isolation_evidence_class="predicate",
        executor_stable=executor_stable,
        executor_stability_receipt=stability_envelope,
        started_at=started_at,
        finished_at=finished_at,
        issued_at=finished_at,
        expires_at=finished_at + args.validity_seconds,
    )
    envelope = benchmark_signer.sign(payload.model_dump())
    args.output.write_bytes(canonical(envelope) + b"\n")
    if args.receipt_bound_profile_output:
        receipt_bound = target.model_copy(update={"benchmark_receipt": digest(envelope)})
        args.receipt_bound_profile_output.write_bytes(
            canonical(receipt_bound.model_dump()) + b"\n"
        )
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure and sign one hosted context profile")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--recall-dataset", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-bound-profile-output", type=Path)
    parser.add_argument("--executor-stability-receipt", type=Path)
    parser.add_argument("--executor-stability-keys", type=Path)
    parser.add_argument("--retrieval-samples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--validity-seconds", type=int, default=604800)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64:
        parser.error("--concurrency must be between 1 and 64")
    if not 60 <= args.validity_seconds <= 604800:
        parser.error("--validity-seconds must be between 60 and 604800")
    if bool(args.executor_stability_receipt) != bool(args.executor_stability_keys):
        parser.error("executor stability receipt and key map must be supplied together")
    run(args)


if __name__ == "__main__":
    main()
