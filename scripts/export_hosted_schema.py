"""Regenerate the closed hosted wire schemas."""

import json
from pathlib import Path
from aether_context import contracts

names = [
    "NamespaceV1",
    "ContextRequestV1",
    "ContextProfileV1",
    "ContextBindingV1",
    "CapabilityV1",
    "SignedV1",
    "PromotionCandidateV1",
    "ContextCheckpointReceiptV1",
    "ContextHydrateReceiptV1",
    "ContextFreezeReceiptV1",
    "AppendRequestV1",
    "RetrieveRequestV1",
    "RemoteDeletionReceiptV1",
    "RetentionCommandV1",
    "ManagedKeyProviderReceiptV1",
    "KeyDestructionReceiptV1",
    "RuntimeBuildManifestV1",
    "ExecutorStabilityReceiptV1",
    "DataPlaneIsolationReceiptV1",
    "HostedScaleReceiptV1",
    "ClosureProofV1",
]
path = Path(__file__).resolve().parents[1] / "aether_context/contracts/schema-v1.json"
path.write_text(
    json.dumps({name: getattr(contracts, name).model_json_schema() for name in names}, indent=2)
    + "\n",
    encoding="utf-8",
)
