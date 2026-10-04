"""Export the existing UI and final checkpoint as a standalone static site."""

import argparse
import json
import shutil
from pathlib import Path

from lastpick.config import project_root
from lastpick.features.carry import FEATURE_FIELDS
from lastpick.recommender.carry import DEFAULT_MODEL, load_model

ROOT = project_root()


def export_site(output):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    model, hero_ids, pool = load_model(DEFAULT_MODEL)
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
    history = model.historical_bundle
    if history:
        history = dict(
            history, indices=[FEATURE_FIELDS.index(field) for field in history["fields"]]
        )
    layers = [
        {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
        for layer in model.network
        if hasattr(layer, "weight")
    ]
    exported = {
        "hero_ids": hero_ids,
        "candidate_pool": pool,
        "use_static": model.use_static,
        "embedding": model.embedding.weight.detach().tolist(),
        "hero_features": model.hero_features.tolist(),
        "layers": layers,
        "history": history,
        "names": {hero["id"]: hero["name"] for hero in heroes},
    }
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "web/static", output / "static", dirs_exist_ok=True)
    (output / "static/model.json").write_text(json.dumps(exported, separators=(",", ":")))
    env = Environment(
        loader=FileSystemLoader(ROOT / "web/templates"), autoescape=select_autoescape()
    )
    html = env.get_template("index.html").render(
        heroes=heroes, url_for=lambda endpoint, filename: f"/static/{filename}"
    )
    html = html.replace(
        '<script defer src="/static/app.js"></script>',
        '<script defer src="/static/model.js"></script>\n'
        '  <script defer src="/static/app.js"></script>',
    )
    (output / "index.html").write_text(html)
    print(f"Exported site to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    export_site(parser.parse_args().output)
