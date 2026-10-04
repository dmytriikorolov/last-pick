"""Bridge ISP extracted CSV statistics into the canonical as-of engine."""

from typing import Any

from lastpick.data.schemas import Match, Player


def canonical_history(records: list[dict[str, Any]]) -> list[Match]:
    output = []
    for row in records:
        players = []
        for side, heroes in enumerate((row["radiant"], row["dire"])):
            for index, hero in enumerate(heroes):
                gpm, gold, lh = row["farm"].get(hero, [None, None, None])
                players.append(
                    Player(
                        index + (128 if side else 0),
                        side,
                        hero,
                        gpm=gpm,
                        gold_t=tuple([None] * 10 + [gold]),
                        lh_t=tuple([None] * 10 + [lh]),
                    )
                )
        output.append(
            Match(
                int(row["match_id"]),
                "isp-csv",
                tuple(players),
                (),
                start_time=int(row["start_time"]),
                duration=row["duration"],
                radiant_win=bool(row["radiant_win"]),
            )
        )
    return output
