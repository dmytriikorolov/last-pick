"""Deterministic carry inference; confidence is evidence strength, not calibration."""

from statistics import mean
from typing import Protocol

from lastpick.config import RoleSettings
from lastpick.data.schemas import Match, Player, RolePrediction


class RoleInferer(Protocol):
    def predict(self, match: Match, side: int) -> RolePrediction: ...


def abstain(method: str, reason: str, evidence: dict[str, object] | None = None) -> RolePrediction:
    return RolePrediction(None, None, None, 0.0, {}, evidence or {}, method, "2.0", reason)


class LegacyPos1Heuristic:
    """ISP rule, including requiring lane metadata for all five players."""

    def __init__(self, settings: RoleSettings | None = None) -> None:
        self.settings = settings or RoleSettings()

    @staticmethod
    def minute_ten(player: Player, name: str) -> float | None:
        # Reproduce ISP's minute-index convention even when a separate times array exists.
        values = getattr(player, name)
        return values[10] if values is not None and len(values) > 10 else None

    def predict(self, match: Match, side: int) -> RolePrediction:
        method = "legacy-pos1"
        team = [p for p in match.players if p.side == side]
        if len(team) != 5 or any(p.lane_role is None for p in team):
            return abstain(method, "missing_required_data")
        safe = [p for p in team if p.lane_role == 1]
        if len(safe) != 2:
            return abstain(method, "ambiguous_pos1")
        snapshots = [(self.minute_ten(p, "lh_t"), self.minute_ten(p, "gold_t")) for p in safe]
        if any(lh is None or gold is None for lh, gold in snapshots):
            return abstain(method, "missing_required_data")
        safe.sort(key=lambda p: self.minute_ten(p, "lh_t") or 0, reverse=True)
        a, b = safe
        lh_gap = (self.minute_ten(a, "lh_t") or 0) - (self.minute_ten(b, "lh_t") or 0)
        gold_gap = (self.minute_ten(a, "gold_t") or 0) - (self.minute_ten(b, "gold_t") or 0)
        if lh_gap < self.settings.legacy_lh_gap or gold_gap < self.settings.legacy_gold_gap:
            return abstain(method, "ambiguous_pos1")
        return RolePrediction(
            a.slot,
            a.hero_id,
            1,
            1.0,
            {a.slot: 1.0, b.slot: 0.0},
            {"lh_gap_10": lh_gap, "gold_gap_10": gold_gap},
            method,
            "1.0",
        )


class FarmRoleBaseline:
    """Compare safelane farm across available timestamps, with secondary source signals.

    No hero whitelist and no invented supervised labels. Missing unrelated lane
    metadata no longer excludes the team; trilane cases are supported. At least
    two lane candidates and separated LH AND gold evidence are still required.
    """

    def __init__(self, settings: RoleSettings | None = None) -> None:
        self.settings = settings or RoleSettings()

    def predict(self, match: Match, side: int) -> RolePrediction:
        method = "farm-v2"
        team = [p for p in match.players if p.side == side]
        if len(team) != 5:
            return abstain(method, "invalid_team")
        safe = sorted((p for p in team if p.lane_role == 1), key=lambda p: p.slot)
        if len(safe) < 2:
            return abstain(method, "insufficient_lane_evidence")
        points: dict[int, list[float]] = {p.slot: [] for p in safe}
        strong_minutes: dict[int, list[int]] = {p.slot: [] for p in safe}
        evidence: dict[str, object] = {"safelane_slots": [p.slot for p in safe]}
        usable = []
        telemetry_present = False
        for minute in self.settings.timestamps:
            if match.duration is not None and match.duration < minute * 60:
                continue
            values = {
                p.slot: (p.snapshot("lh_t", minute), p.snapshot("gold_t", minute)) for p in safe
            }
            if any(
                lh is None or gold is None or lh < 0 or gold < 0 for lh, gold in values.values()
            ):
                continue
            telemetry_present = True
            if len(set(values.values())) == 1:
                # Identical snapshots contain no separation evidence (often all-zero early data).
                continue
            usable.append(minute)
            for player in safe:
                lh, gold = values[player.slot]
                assert lh is not None and gold is not None
                rivals = [v for slot, v in values.items() if slot != player.slot]
                rival_lh = max(v[0] for v in rivals if v[0] is not None)
                rival_gold = max(v[1] for v in rivals if v[1] is not None)
                lh_gap, gold_gap = lh - rival_lh, gold - rival_gold
                separation = min(
                    lh_gap / self.settings.min_lh_gap, gold_gap / self.settings.min_gold_gap
                )
                points[player.slot].append(
                    max(-1.0, min(1.0, separation / self.settings.separation_scale))
                )
                if separation >= 1:
                    strong_minutes[player.slot].append(minute)
                evidence[f"slot_{player.slot}_minute_{minute}"] = {
                    "lh_gap": lh_gap,
                    "gold_gap": gold_gap,
                    "separation": separation,
                }
        if not usable:
            return abstain(
                method, "ambiguous_pos1" if telemetry_present else "missing_telemetry", evidence
            )
        scores = {}
        for p in safe:
            # position_est is source evidence, never treated as a trusted target.
            position_hint = (
                1.0 if p.position_est == 1 else -1.0 if p.position_est in (4, 5) else 0.0
            )
            scores[p.slot] = (
                self.settings.farm_weight * mean(points[p.slot])
                + self.settings.position_est_weight * position_hint
            )
        ranked = sorted(safe, key=lambda p: (-scores[p.slot], p.slot))
        best, other = ranked[:2]
        score_gap = scores[best.slot] - scores[other.slot]
        strong = strong_minutes[best.slot]
        consistent = sum(v > 0 for v in points[best.slot]) / len(usable)
        confidence = (
            min(
                self.settings.max_confidence,
                self.settings.confidence_base
                + self.settings.confidence_farm_weight * max(0, mean(points[best.slot]))
                + self.settings.confidence_per_timestamp * len(strong),
            )
            * consistent
        )
        evidence.update(
            {
                "usable_minutes": usable,
                "strong_minutes": strong,
                "score_gap": score_gap,
                "confidence_calibrated": False,
            }
        )
        # Optional telemetry is reported without allowing final farm to override early evidence.
        evidence["secondary"] = {
            str(p.slot): {
                "position_est": p.position_est,
                "xp_10": p.snapshot("xp_t", 10),
                "gpm": p.gpm,
                "xpm": p.xpm,
            }
            for p in safe
        }
        reason = None
        if not strong or score_gap < self.settings.min_score_gap:
            reason = "ambiguous_pos1"
        elif confidence < self.settings.min_confidence:
            reason = "below_role_confidence"
        return RolePrediction(
            best.slot if reason is None else None,
            best.hero_id if reason is None else None,
            1 if reason is None else None,
            confidence,
            scores,
            evidence,
            method,
            "2.0",
            reason,
        )


def make_inferer(settings: RoleSettings) -> RoleInferer:
    if settings.method == "legacy":
        return LegacyPos1Heuristic(settings)
    return FarmRoleBaseline(settings)
