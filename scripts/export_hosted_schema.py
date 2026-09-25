"""Regenerate or verify the closed hosted wire schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aether_context import contracts


NAMES = [
    "NamespaceV1",
    "ContextRequestV1",
    "ContextReservationReceiptV1",
    "ContextReservationCommandV2",
    "ContextReservationReceiptV2",
    "ContextReservationCommandV3",
    "ContextReservationReceiptV3",
    "ContextReservationReleaseCommandV1",
    "ContextReservationReleaseReceiptV1",
    "ContextReservationAbortCommandV1",
    "ContextReservationAbortReceiptV1",
    "ContextReservationAbortExpiryCommandV1",
    "ContextInvocationNoDispatchSnapshotV1",
    "ContextInvocationNoDispatchReceiptV1",
    "ContextReservationExpiryCommandV1",
    "ContextReservationExpiryReceiptV1",
    "ContextCheckpointDurabilityCommandV1",
    "ContextCheckpointDurabilityReceiptV1",
    "ContextCheckpointDurabilityReceiptV2",
    "ContextProfileV1",
    "ContextBindingV1",
    "CapabilityV1",
    "SignedV1",
    "ContextBindingCommandV2",
    "ContextBindingCommandV3",
    "PromotionCandidateV1",
    "ContextCheckpointReceiptV1",
    "ContextHydrateCommandV2",
    "ContextHealthV1",
    "ContextHealthV2",
    "ContextHealthV3",
    "ContextHydrateReceiptV1",
    "ContextHydrateReceiptV2",
    "ContextFreezeReceiptV1",
    "AppendRequestV1",
    "RetrieveRequestV1",
    "RemoteDeletionReceiptV1",
    "RetentionCommandV1",
    "ContextRetentionReceiptV1",
    "ManagedKeyProviderReceiptV1",
    "KeyDestructionReceiptV1",
    "ContextLocalKeyDestructionV1",
    "ContextDeletionReceiptV1",
    "ContextDeletionReceiptV2",
    "RuntimeBuildManifestV1",
    "RuntimeEnvironmentV1",
    "RuntimeBuildManifestV2",
    "RuntimeEnvironmentV2",
    "RuntimeBuildManifestV3",
    "ExecutorStabilityReceiptV1",
    "DataPlaneIsolationReceiptV1",
    "HostedScaleReceiptV1",
    "ClosureProofV1",
]
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "aether_context/contracts/schema-v1.json"
)


def render_schema() -> str:
    return (
        json.dumps(
            {name: getattr(contracts, name).model_json_schema() for name in NAMES},
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_schema()
    if args.check:
        if SCHEMA_PATH.read_text(encoding="utf-8") != rendered:
            raise SystemExit("hosted schema is stale")
        return
    SCHEMA_PATH.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
