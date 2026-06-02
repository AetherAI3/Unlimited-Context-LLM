"""Configuration dataclasses + ~/.aether-context persistence.

``PoolConfig`` governs the on-disk context pool (size = reach, index kind, slice size).
``SessionConfig`` governs a single run (which model, the window fractions that drive
trigger/target/verbatim behavior). Both are plain dataclasses (no pydantic) to keep the
core dependency surface at numpy-only.

Constants:
  * window fractions TRIGGER 0.75 / TARGET 0.50 / VERBATIM 0.30
  * pool reach math ``reach ≈ pool_gb × 233M tokens`` (README table)
  * the 5 GB pool floor (README minimum for a usable reach)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aether_context._log import get_logger
from aether_context.errors import PoolBudgetError, PoolCorrupt

_log = get_logger(__name__)

# --- engine constants --------------------------------------------------------
#: Minimum usable pool size in GB (README floor).
POOL_GB_FLOOR = 5
#: Tokens of reach per GB of pool (README: reach ≈ pool_gb × 233M).
TOKENS_PER_GB = 233_000_000
#: Window fraction at which overflow encoding triggers.
TRIGGER_FRACTION = 0.75
#: Target window occupancy after a paged compaction.
TARGET_FRACTION = 0.50
#: Fraction of the window kept verbatim (never encoded away).
VERBATIM_FRACTION = 0.30

#: Default retrieval embedding dimensionality (the 256-dim retrieval embedding).
DEFAULT_DIM = 256
#: Default tokens per encoded slice.
DEFAULT_SLICE_TOKENS = 512
#: Config file name inside the pool dir.
CONFIG_FILENAME = "config.json"

_VALID_INDEX = ("flat", "hnsw", "tiered")
_VALID_MODE = ("separate", "shared")


def reach_tokens(pool_gb: int) -> int:
    """Token reach for a given pool size: ``pool_gb × TOKENS_PER_GB``."""
    return pool_gb * TOKENS_PER_GB


def default_pool_dir() -> Path:
    """The default pool directory: ``~/.aether-context``."""
    return Path.home() / ".aether-context"


@dataclass
class PoolConfig:
    """On-disk context pool configuration.

    ``pool_gb`` is *reach*, not window. Rejects ``pool_gb < POOL_GB_FLOOR`` with a reason.
    """

    pool_gb: int = POOL_GB_FLOOR
    mode: str = "separate"
    index: str = "flat"
    dim: int = DEFAULT_DIM
    slice_tokens: int = DEFAULT_SLICE_TOKENS
    dir: Path = field(default_factory=default_pool_dir)

    def __post_init__(self) -> None:
        # Path coercion (callers may pass a str).
        if not isinstance(self.dir, Path):
            self.dir = Path(self.dir)
        if self.pool_gb < POOL_GB_FLOOR:
            raise PoolBudgetError(
                f"pool_gb={self.pool_gb} is below the {POOL_GB_FLOOR} GB pool floor "
                f"(reach would be too small to be useful)"
            )
        if self.index not in _VALID_INDEX:
            raise PoolBudgetError(
                f"index={self.index!r} is not one of {_VALID_INDEX}",
                hint="Use index='flat' (numpy, always works), 'hnsw', or 'tiered'.",
            )
        if self.mode not in _VALID_MODE:
            raise PoolBudgetError(
                f"mode={self.mode!r} is not one of {_VALID_MODE}",
                hint="Use mode='separate' (default) or 'shared'.",
            )

    @property
    def reach(self) -> int:
        """Token reach of this pool (``pool_gb × TOKENS_PER_GB``)."""
        return reach_tokens(self.pool_gb)

    def config_path(self) -> Path:
        """Path to the persisted ``config.json`` inside :attr:`dir`."""
        return self.dir / CONFIG_FILENAME

    def save(self) -> Path:
        """Write this config to ``<dir>/config.json`` (creating the dir). Returns the path."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.config_path()
        data = asdict(self)
        data["dir"] = str(self.dir)  # JSON cannot hold a Path
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        _log.debug("saved pool config to %s", path)
        return path

    @classmethod
    def load(cls, dir: Path | str | None = None) -> "PoolConfig":
        """Load config from ``<dir>/config.json``; defaults if the file is absent.

        Raises :class:`~aether_context.errors.PoolCorrupt` if the file exists but is
        malformed.
        """
        base = Path(dir) if dir is not None else default_pool_dir()
        path = base / CONFIG_FILENAME
        if not path.exists():
            return cls(dir=base)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise PoolCorrupt(f"could not read pool config at {path}: {exc}") from exc
        raw.pop("reach", None)  # derived; never persisted but be defensive
        raw["dir"] = base
        try:
            return cls(**raw)
        except TypeError as exc:
            raise PoolCorrupt(f"pool config at {path} has unexpected fields: {exc}") from exc


@dataclass
class SessionConfig:
    """Per-run configuration.

    ``model`` is required (a spec string like ``"ollama/qwen2.5"`` or ``"mock"``, or a
    ``LocalLLM`` object). The window fractions use the engine defaults.
    """

    model: object
    system: str | None = None
    max_tokens: int | None = None
    verbatim_fraction: float = VERBATIM_FRACTION
    trigger_fraction: float = TRIGGER_FRACTION
    target_fraction: float = TARGET_FRACTION

    def __post_init__(self) -> None:
        for name, value in (
            ("verbatim_fraction", self.verbatim_fraction),
            ("trigger_fraction", self.trigger_fraction),
            ("target_fraction", self.target_fraction),
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise PoolBudgetError(
                    f"{name}={value} must be within [0.0, 1.0]",
                    hint="Window fractions are proportions of the model's native window.",
                )


__all__ = [
    "PoolConfig",
    "SessionConfig",
    "reach_tokens",
    "default_pool_dir",
    "POOL_GB_FLOOR",
    "TOKENS_PER_GB",
    "TRIGGER_FRACTION",
    "TARGET_FRACTION",
    "VERBATIM_FRACTION",
    "DEFAULT_DIM",
    "DEFAULT_SLICE_TOKENS",
    "CONFIG_FILENAME",
]
