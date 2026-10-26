"""Native-memory protocol: a local model that can emit and consume latent capsules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..contracts import NamespaceV1
    from ..contracts.latent_v1 import ContextHandoffGrantV1, LatentCapsuleV1, LatentUseReceiptV1


@runtime_checkable
class NativeMemoryLLM(Protocol):
    """Sender and receiver halves of a same-model latent handoff.

    ``encode_capsule`` returns the manifest and its tensor bytes (never inlined).
    ``receive`` checks the grant and tensor digest before injecting; it returns the
    generated text (``None`` when refused) and the ``LatentUseReceiptV1``.
    """

    model_id: str

    @property
    def adapter_id(self) -> str: ...

    def encode_capsule(
        self, text: str, *, namespace: "NamespaceV1", capsule_id: str, anchor_digest: str,
        base_revision: str, created_at: int,
    ) -> "tuple[LatentCapsuleV1, bytes]": ...

    def receive(
        self, capsule: "LatentCapsuleV1", grant: "ContextHandoffGrantV1", tensor: bytes, prompt: str, *,
        receipt_id: str, now: int, max_tokens: int = 32, principal_id: Optional[str] = None,
        current_fence: Optional[int] = None,
    ) -> "tuple[Optional[str], LatentUseReceiptV1]": ...
