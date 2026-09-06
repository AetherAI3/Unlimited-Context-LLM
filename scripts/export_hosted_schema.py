"""Regenerate the closed wire schema without changing signed golden fixtures."""

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
    "AppendRequestV1",
    "RetrieveRequestV1",
    "ClosureProofV1",
]
path = Path(__file__).resolve().parents[1] / "aether_context/contracts/schema-v1.json"
path.write_text(
    json.dumps({name: getattr(contracts, name).model_json_schema() for name in names}, indent=2)
    + "\n",
    encoding="utf-8",
)
