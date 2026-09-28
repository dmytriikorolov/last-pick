from unittest.mock import patch

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
    with patch("web.app.load_model", side_effect=AssertionError("reloaded")), \
            patch.object(model, "train", side_effect=AssertionError("trained")):
        assert client.post("/api/recommend", json=draft()).status_code == 200
