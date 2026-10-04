"""Run from the repository root with: python -m web.app."""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from lastpick.recommender.carry import DEFAULT_MODEL, load_model, rank_candidates
from scripts import _bootstrap as _bootstrap

ROOT = Path(__file__).resolve().parent.parent


def create_app():
    app = Flask(__name__)
    model, hero_ids, pool = load_model(DEFAULT_MODEL)
    known_ids = set(hero_ids)
    snapshot = json.loads((ROOT / "data/constants/dotaconstants_heroes.json").read_text())
    heroes = sorted(
        [
            {
                "id": row["id"],
                "name": row["localized_name"],
                "attribute": row["primary_attr"],
                "image": f"/static/heroes/{row['id']}.png",
            }
            for row in snapshot.values()
        ],
        key=lambda hero: hero["name"],
    )
    names = {hero["id"]: hero["name"] for hero in heroes}
    app.extensions["recommender"] = {"model": model, "hero_ids": hero_ids, "pool": pool}

    @app.get("/")
    def index():
        return render_template("index.html", heroes=heroes)

    @app.post("/api/recommend")
    def recommend():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="Send a JSON object."), 400
        groups = {}
        for field in ("allies", "enemies", "banned"):
            values = body.get(field, [] if field == "banned" else None)
            if not isinstance(values, list):
                return jsonify(error=f"{field} must be an array of hero IDs."), 400
            if field != "banned" and len(values) != 4:
                return jsonify(error=f"{field} must contain exactly 4 heroes."), 400
            if any(type(hero) is not int or hero not in known_ids for hero in values):
                return jsonify(error=f"{field} contains an unknown or invalid hero ID."), 400
            if len(set(values)) != len(values):
                return jsonify(error=f"{field} contains duplicate heroes."), 400
            groups[field] = values
        visible = groups["allies"] + groups["enemies"]
        if len(set(visible)) != 8:
            return jsonify(error="The same hero cannot appear on both teams."), 400
        if set(visible) & set(groups["banned"]):
            return jsonify(error="A drafted hero cannot also be banned."), 400
        ranking = rank_candidates(
            model,
            hero_ids,
            pool,
            groups["allies"],
            groups["enemies"],
            top_n=5,
            banned=groups["banned"],
        )
        return jsonify(
            recommendations=[
                {"hero_id": hero, "hero_name": names[hero], "model_score": score}
                for hero, score in ranking
            ]
        )

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
