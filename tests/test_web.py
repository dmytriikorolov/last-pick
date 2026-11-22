import json
import random
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from web.app import create_app

# creates the app and loads its model once
app = create_app()
client = app.test_client()


# makes a valid website request with four heroes on each team
def draft():
    return {"allies": [2, 5, 25, 86], "enemies": [14, 26, 74, 99]}


# checks that the page and top five response work
def test_page_and_recommendations():
    assert client.get("/").status_code == 200
    response = client.post("/api/recommend", json=draft())
    assert response.status_code == 200
    rows = response.json["recommendations"]
    assert len(rows) == 5
    for row in rows:
        assert {"hero_id", "hero_name", "model_score"} <= set(row)
        assert row["hero_name"] and 0 <= row["model_score"] <= 1


# checks invalid sizes, repeated heroes and unknown ids
def test_invalid_drafts_are_rejected():
    for field in ("allies", "enemies"):
        for heroes in ([2, 5, 25], [2, 5, 25, 86, 1], [2, 2, 25, 86], [9999, 5, 25, 86]):
            body = draft()
            body[field] = heroes
            assert client.post("/api/recommend", json=body).status_code == 400
    body = draft()
    body["enemies"][0] = body["allies"][0]
    assert client.post("/api/recommend", json=body).status_code == 400


# checks that the website leaves banned heroes out
def test_bans_are_respected():
    body = draft()
    first = client.post("/api/recommend", json=body).json["recommendations"]
    body["banned"] = [row["hero_id"] for row in first]
    response = client.post("/api/recommend", json=body)
    assert response.status_code == 200
    rows = response.json["recommendations"]
    assert len(rows) == 5
    assert not set(body["banned"]) & {row["hero_id"] for row in rows}


# checks that requests reuse the loaded model
def test_requests_do_not_reload_or_train():
    model = app.extensions["recommender"]["model"]
    with (
        patch("web.app.load_model", side_effect=AssertionError("reloaded")),
        patch.object(model, "train", side_effect=AssertionError("trained")),
    ):
        assert client.post("/api/recommend", json=draft()).status_code == 200


def test_carry_pool_filters_before_ranking():
    body = draft()
    excluded = {
        row["hero_id"] for row in client.post("/api/recommend", json=body).json["recommendations"]
    }
    available = sorted(
        set(app.extensions["recommender"]["pool"])
        - excluded
        - set(body["allies"] + body["enemies"])
    )
    body["candidate_pool"] = available[:2]
    response = client.post("/api/recommend", json=body)
    assert response.status_code == 200
    assert {r["hero_id"] for r in response.json["recommendations"]} == set(available[:2])
    body["candidate_pool"] = []
    assert client.post("/api/recommend", json=body).json["recommendations"] == []
    for invalid in (None, [True], [99999], [available[0], available[0]]):
        body["candidate_pool"] = invalid
        assert client.post("/api/recommend", json=body).status_code == 400


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js is required for browser model parity"
)
def test_static_model_matches_flask_with_carry_pool():
    rng = random.Random(42)
    recommender = app.extensions["recommender"]
    drafts = []
    for index in range(12):
        heroes = rng.sample(recommender["hero_ids"], 11)
        body = {"allies": heroes[:4], "enemies": heroes[4:8], "banned": heroes[8:]}
        if index % 2:
            body["candidate_pool"] = rng.sample(recommender["pool"], 8)
        drafts.append(body)
    static = subprocess.run(
        [
            "node",
            "-e",
            """
            const {rank} = require('./site/public/static/model.js');
            const model = require('./site/public/static/model.json');
            const drafts = JSON.parse(require('fs').readFileSync(0, 'utf8'));
            console.log(JSON.stringify(drafts.map(draft => rank(model, draft))));
            """,
        ],
        input=json.dumps(drafts),
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    for body, actual in zip(drafts, json.loads(static.stdout), strict=True):
        expected = client.post("/api/recommend", json=body).json["recommendations"]
        assert [row["hero_id"] for row in actual] == [row["hero_id"] for row in expected]
        assert [row["model_score"] for row in actual] == pytest.approx(
            [row["model_score"] for row in expected], abs=1e-6
        )
