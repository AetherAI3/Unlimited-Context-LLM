"""Logging seam for aether-context.

Library discipline: a library never configures the root logger and never prints. Each
module gets a named logger under the ``aether_context`` namespace with a ``NullHandler``
attached, so it stays silent unless the host application configures logging. There is no
``print()`` anywhere in the package — use ``get_logger(__name__)`` instead.
"""
from __future__ import annotations

import logging

_ROOT_NAME = "aether_context"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger with a ``NullHandler`` attached.

    The logger is placed under the ``aether_context`` root namespace so a host app can
    configure the whole package with one call (``logging.getLogger("aether_context")``).
    A ``NullHandler`` is attached idempotently so the library is silent by default and
    never emits "No handlers could be found" warnings.
    """
    full_name = name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}"
    logger = logging.getLogger(full_name)
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


__all__ = ["get_logger"]
