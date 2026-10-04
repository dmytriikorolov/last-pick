"""Chronological splits keep matches and equal timestamps together."""

from dataclasses import dataclass

from lastpick.data.schemas import DraftDecision, Rejection


@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[DraftDecision, ...]
    validation: tuple[DraftDecision, ...]
    test: tuple[DraftDecision, ...]
    validation_cutoff: int
    test_cutoff: int


def temporal_split(
    decisions: list[DraftDecision], train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> TemporalSplit:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("invalid split fractions")
    times: dict[int, int] = {}
    keys = set()
    for decision in decisions:
        if decision.start_time is None:
            raise Rejection("missing_decision_time")
        key = (decision.match_id, decision.side, decision.event_order)
        if key in keys:
            raise Rejection("duplicate_decision")
        keys.add(key)
        if decision.match_id in times and times[decision.match_id] != decision.start_time:
            raise Rejection("inconsistent_match_timestamp")
        times[decision.match_id] = decision.start_time
    ordered = sorted(set(times.values()))
    if len(ordered) < 3:
        raise ValueError("at least three distinct timestamps required")
    validation_index = max(1, min(len(ordered) - 2, int(len(ordered) * train_fraction)))
    test_index = max(
        validation_index + 1,
        min(len(ordered) - 1, int(len(ordered) * (train_fraction + validation_fraction))),
    )
    validation_cutoff, test_cutoff = ordered[validation_index], ordered[test_index]
    groups: list[list[DraftDecision]] = [[], [], []]
    for decision in sorted(
        decisions, key=lambda d: (d.start_time or 0, d.match_id, d.side, d.event_order)
    ):
        assert decision.start_time is not None
        index = (
            0
            if decision.start_time < validation_cutoff
            else 1
            if decision.start_time < test_cutoff
            else 2
        )
        groups[index].append(decision)
    return TemporalSplit(
        tuple(groups[0]), tuple(groups[1]), tuple(groups[2]), validation_cutoff, test_cutoff
    )
