"""Domain-separated authenticated encryption and public-key service receipts."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import zlib

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization

from .contracts import SignedV1, canonical


class ContextFault(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ReceiptSigner:
    def __init__(self, key_id: str, key: Ed25519PrivateKey):
        self.key_id, self.key = key_id, key

    def sign(self, payload: dict) -> dict:
        raw = self.key.sign(canonical({"key_id": self.key_id, "payload": payload}))
        return {
            "payload": payload,
            "key_id": self.key_id,
            "signature": base64.b64encode(raw).decode("ascii"),
        }


def require_disjoint_keys(*groups: dict[str, Ed25519PublicKey]) -> None:
    """Distinct key IDs do not create independent authority when keys repeat."""
    seen: set[bytes] = set()
    for group in groups:
        current = {
            key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            for key in group.values()
        }
        if current & seen:
            raise ContextFault("context_verification_key_overlap")
        seen |= current


def verify(envelope: dict, keys: dict[str, Ed25519PublicKey]) -> dict:
    try:
        item = SignedV1.model_validate(envelope)
        keys[item.key_id].verify(
            base64.b64decode(item.signature, validate=True),
            canonical({"key_id": item.key_id, "payload": item.payload}),
        )
        return item.payload
    except (KeyError, ValueError, InvalidSignature) as exc:
        raise ContextFault("context_signature_invalid") from exc


class EnvelopeCipher:
    """The key provider supplies a managed/unwrapped key; never store it here."""

    def __init__(self, key: bytes, domain: bytes = b"context-overpool/v1"):
        if len(key) != 32:
            raise ValueError("a 256-bit key is required")
        self.key, self.domain = key, domain

    def derive(self, namespace: dict) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=self.domain + b"\x00" + canonical(namespace),
        ).derive(self.key)

    def token(self, namespace: dict, word: str) -> str:
        return hmac.new(
            self.derive(namespace), b"index\x00" + word.encode(), hashlib.sha256
        ).hexdigest()

    def encrypt(self, value: dict, aad: dict) -> bytes:
        nonce = os.urandom(12)
        return nonce + AESGCM(self.derive(aad)).encrypt(
            nonce, zlib.compress(canonical(value)), canonical(aad)
        )

    def decrypt(self, value: bytes, aad: dict, max_bytes: int) -> dict:
        try:
            compressed = AESGCM(self.derive(aad)).decrypt(value[:12], value[12:], canonical(aad))
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed, max_bytes + 1)
            if len(raw) > max_bytes or not decoder.eof or decoder.unused_data:
                raise ContextFault("context_pack_limit")
            result = json.loads(raw)
            canonical(result)
            if not isinstance(result, dict):
                raise ValueError("object required")
            return result
        except (InvalidTag, ValueError, zlib.error) as exc:
            raise ContextFault("context_integrity_failure") from exc
