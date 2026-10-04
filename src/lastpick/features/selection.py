"""Model feature selection is separate from historical statistics accumulation."""

from dataclasses import dataclass
from typing import Any

from lastpick.features.carry import FEATURE_FIELDS, feature_vector


@dataclass(frozen=True)
class FeatureSpec:
    fields: tuple[str, ...] = tuple(FEATURE_FIELDS)

    def __post_init__(self) -> None:
        if len(set(self.fields)) != len(self.fields) or not set(self.fields) <= set(FEATURE_FIELDS):
            raise ValueError("feature selection contains unknown or repeated fields")

    def values(
        self, stats: dict[str, Any], candidate: int, allies: list[int], enemies: list[int]
    ) -> tuple[float, ...]:
        full = feature_vector(stats, candidate, allies, enemies)
        return tuple(full[FEATURE_FIELDS.index(name)] for name in self.fields)
