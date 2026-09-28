from scripts.build_dataset import build_match_examples
from test_draft_parsing import make_match


# checks the carry choice and win label for both teams
def test_carry_and_winner_labels():
    for winner in (True, False):
        match, _, _ = make_match(8, candidate_hero=19, mid_hero=25, both_teams=True)
        match["radiant_win"] = winner
        rows, _, valid = build_match_examples(match)
        assert valid and len(rows) == 2
        radiant = next(row for row in rows if row["side"] == "radiant")
        dire = next(row for row in rows if row["side"] == "dire")
        # farm identifies tiny as the carry even though lina can also be a carry
        assert radiant["candidate_hero_id"] == 19
        assert radiant["win"] == int(winner)
        assert dire["win"] == int(not winner)
        for row in rows:
            visible = [row[f"{side}_{i}"] for side in ("ally", "enemy") for i in range(1, 5)]
            assert row["known_ally_count"] == row["known_enemy_count"] == 4
            assert "" not in visible
            assert len(set(visible + [row["candidate_hero_id"]])) == 9


# skips games where farm data cannot identify the carry
def test_ambiguous_or_missing_farm_is_skipped():
    for field, value in (("lh_t", [0] * 10 + [20]), ("gold_t", None)):
        match, _, _ = make_match(8)
        carry = next(player for player in match["players"] if player["hero_id"] == 25)
        carry[field] = value
        assert build_match_examples(match)[0] == []


# checks that unusable drafts do not enter the training data
def test_invalid_and_early_drafts_do_not_create_examples():
    for value in (None, {}, []):
        rows, _, valid = build_match_examples(value)
        assert rows == [] and not valid
    match, _, _ = make_match(8)
    match["picks_bans"].pop()
    assert build_match_examples(match)[0] == []
    for order in (0, 2):
        match, _, _ = make_match(order)
        assert build_match_examples(match)[0] == []
