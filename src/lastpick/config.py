"""Small, explicit TOML configuration. All paths are relative to a project root."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib
import os


@dataclass(frozen=True)
class Paths:
    raw: str = "data/raw/opendota/matches"
    normalized: str = "data/interim/v2/matches"
    output: str = "data/processed/v2"
    ingestion: str = "data/raw/opendota/foundation_state"


@dataclass(frozen=True)
class Filters:
    game_modes: tuple[int, ...] = (22,)
    lobby_types: tuple[int, ...] = (7,)
    min_rank: float | None = 70
    min_ranked_players: int | None = 5
    patches: tuple[int, ...] = ()
    min_patch: int | None = None
    min_duration: int = 900
    require_rank: bool = True


@dataclass(frozen=True)
class RoleSettings:
    method: str = "farm-v2"
    timestamps: tuple[int, ...] = (5, 10, 15)
    legacy_lh_gap: float = 15
    legacy_gold_gap: float = 400
    min_lh_gap: float = 15
    min_gold_gap: float = 400
    min_confidence: float = 0.65
    min_score_gap: float = 0.15
    position_est_weight: float = 0.15
    farm_weight: float = 0.85


@dataclass(frozen=True)
class DatasetSettings:
    picks: tuple[int, ...] = (3, 4, 5)
    exclude_candidate_phase_attempts: bool = True
    hero_universe: tuple[int, ...] = ()
    version: str = "foundation-v2"


@dataclass(frozen=True)
class IngestionSettings:
    api_url: str = "https://api.opendota.com/api"
    discovery: str = "public"
    request_budget: int = 3000
    daily_budget: bool = True
    target_matches: int = 150000
    delay_seconds: float = 1.2
    timeout_seconds: float = 30
    retries: int = 3
    retry_seconds: float = 5


@dataclass(frozen=True)
class TrainingSettings:
    min_candidate_examples: int = 100
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 1024
    learning_rate: float = 0.003
    model_path: str = "models/foundation_carry_recommender.pt"
    results_path: str = "models/foundation_carry_training_results.json"


def project_root() -> Path:
    """Assets stay in the checkout; installed users can set LASTPICK_ROOT."""
    configured = os.environ.get("LASTPICK_ROOT")
    if configured:
        return Path(configured).resolve()
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / "pyproject.toml").exists() and (directory / "models").is_dir():
            return directory
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    source: str = "opendota"
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    filters: Filters = field(default_factory=Filters)
    roles: RoleSettings = field(default_factory=RoleSettings)
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return Config()
    with path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    sections = {"paths": Paths, "filters": Filters, "roles": RoleSettings,
                "dataset": DatasetSettings, "ingestion": IngestionSettings,
                "training": TrainingSettings}
    for name, constructor in sections.items():
        values = data.pop(name, {})
        values = {key: tuple(value) if isinstance(value, list) else value
                  for key, value in values.items()}
        data[name] = constructor(**values)
    config = Config(**data)
    if config.source != "opendota":
        raise ValueError("only the OpenDota adapter is implemented")
    if config.roles.method not in {"farm-v2", "legacy"}:
        raise ValueError("unknown role inference method")
    if not 0 <= config.roles.min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if config.ingestion.retries < 1 or config.ingestion.request_budget < 1:
        raise ValueError("retries and request_budget must be positive")
    return config
