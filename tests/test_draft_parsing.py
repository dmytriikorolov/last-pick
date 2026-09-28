import json

from scripts.build_dataset import build_match_examples
from scripts.parse_drafts import validate_match


# makes a small match with a known carry and pick order
def make_match(candidate_order, first_team=0, candidate_hero=25,
               mid_hero=None, both_teams=False):
    heroes = [2, 3, 5, 7, 9, 13, 14, 15, 16, 17]
    teams = [(first_team + i) % 2 for i in range(10)]
    team = teams[candidate_order]
    heroes[candidate_order] = candidate_hero
    own_orders = [i for i in range(10) if teams[i] == team and i != candidate_order]
    if mid_hero is not None:
        heroes[own_orders[1]] = mid_hero

    players = []
    for side in (0, 1):
        side_orders = [i for i in range(10) if teams[i] == side]
        carry_order = candidate_order if side == team else side_orders[-1]
        support_order = next(i for i in side_orders if i != carry_order)
        for slot, order in enumerate(side_orders):
            lane_role = None
            if side == team or both_teams:
                lane_role = 1 if order in (carry_order, support_order) else 2 if slot == 2 else 3
            player = {
                "player_slot": slot if side == 0 else 128 + slot,
                "hero_id": heroes[order],
                "lane_role": lane_role,
                "lh_t": [0] * 10 + [65 if order == carry_order else 15],
                "gold_t": [0] * 10 + [4000 if order == carry_order else 2000],
            }
            players.append(player)

    match = {
        "match_id": 123,
        "game_mode": 22,
        "lobby_type": 7,
        "start_time": 1000,
        "patch": 60,
        "radiant_win": True,
        "players": players,
        "picks_bans": [
            {"is_pick": True, "order": i, "team": teams[i], "hero_id": heroes[i]}
            for i in range(10)
        ],
    }
    return match, teams, heroes

# inserts a pick attempt that did not become a final hero
def add_extra_attempt(match, order, hero_id, team):
    match["picks_bans"].insert(order, {
        "is_pick": True, "order": order, "team": team, "hero_id": hero_id,
    })
    for i, pick in enumerate(match["picks_bans"]):
        pick["order"] = i


# checks that the parser finds the final ten picks in the right order
def test_final_picks_and_team_order():
    match, teams, heroes = make_match(8)
    match["picks_bans"].reverse()
    valid, reason, draft = validate_match(match)
    assert valid, reason
    assert [draft[f"pick_{i}_hero_id"] for i in range(10)] == heroes
    assert [draft[f"pick_{i}_team"] for i in range(10)] == teams


# checks that current phase enemy picks stay hidden
def test_visibility_follows_phases():
    for order in range(4, 10):
        match, teams, heroes = make_match(order)
        rows, _, valid = build_match_examples(match)
        assert valid and len(rows) == 1
        row = rows[0]
        team = teams[order]
        allies = [row[f"ally_{i}"] for i in range(1, 5) if row[f"ally_{i}"] != ""]
        enemies = [row[f"enemy_{i}"] for i in range(1, 5) if row[f"enemy_{i}"] != ""]
        enemy_count = 2 if order < 8 else 4
        assert allies == [heroes[i] for i in range(order) if teams[i] == team]
        # enemies picked in the current phase are still hidden
        assert enemies == [hero for hero, side in zip(heroes, teams) if side != team][:enemy_count]
        if order >= 8:
            assert len(allies) == len(enemies) == 4


# checks extra pick attempts and rejects uncertain drafts
def test_hidden_attempts_and_invalid_drafts():
    match, _, heroes = make_match(8)
    add_extra_attempt(match, 0, 99, 0)
    valid, _, draft = validate_match(match)
    assert valid
    assert [draft[f"pick_{i}_hero_id"] for i in range(10)] == heroes
    assert json.loads(draft["extra_pick_events"])[0]["phase"] == 1

    match, _, _ = make_match(8)
    add_extra_attempt(match, 8, 99, 1)
    rows, skipped, valid = build_match_examples(match)
    assert valid and rows == []
    assert skipped["candidate_phase_extra_attempt"] == 1

    match, _, _ = make_match(8)
    add_extra_attempt(match, 4, match["picks_bans"][0]["hero_id"], 0)
    assert not validate_match(match)[0]
    match, _, _ = make_match(8)
    match["picks_bans"].pop()
    assert not validate_match(match)[0]
