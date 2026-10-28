"""Learned latent handoff (W7): soft-prefix capsules between a frozen same-model pair.

Optional. Requires the ``[latent]`` extra (torch + transformers + pydantic). The
NumPy-only core never imports this package, and importing the package itself
does not import torch: ``tensor_codec`` and ``handoff`` need numpy + pydantic
only, while ``projector``, ``hf_soft_prefix`` and ``training`` import torch.

Capsules are lossy (``LatentCapsuleV1.lossless`` is always false). The exact
anchor is the source record whose digest every capsule carries.
"""

from .protocol import NativeMemoryLLM

__all__ = ["NativeMemoryLLM"]
