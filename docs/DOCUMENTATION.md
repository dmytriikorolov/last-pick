# 1. What the project does

This project is a Dota 2 carry recommendation system. The user selects four allies and four enemies, and the model ranks possible position-1 (pos1) heroes for that draft. It returns the top five available carries with model scores, excluding picked and banned heroes. The neural network is trained from historical OpenDota matches.

# 2. Main project flow

```text
OpenDota raw matches
  data/raw/opendota/matches/*.json
    ->
Draft parsing and carry identification
  scripts/parse_drafts.py, scripts/build_dataset.py
    ->
Processed datasets, match statistics and quality checks
  scripts/build_full_draft_dataset.py
  scripts/build_match_data.py, scripts/build_training_quality.py
    ->
Historical feature generation from earlier training matches
  scripts/carry_historical_features.py
    ->
Model training with hero details from data/constants/heroes.csv
  scripts/carry_model.py, scripts/train_carry_recommender.py
    ->
models/final_carry_recommender.pt
    ->
Recommendation functions and web app
  scripts/recommend_carry.py, web/app.py
    ->
Top carry recommendations
```

# 3. Important files

**Data processing**

- `scripts/parse_drafts.py` reconstructs final picks from saved matches.
- `scripts/build_dataset.py` identifies the carry and records the picks visible at their decision.
- `scripts/build_full_draft_dataset.py` creates complete five-versus-five draft rows.
- `scripts/build_match_data.py` extracts match statistics, player statistics and events.
- `scripts/build_training_quality.py` marks examples with missing data or other quality problems.
- `scripts/build_hero_metadata.py` creates the hero details used by the model.

**Model / training**

- `scripts/carry_historical_features.py` builds hero, ally-synergy and enemy-matchup inputs from earlier training matches.
- `scripts/carry_model.py` defines the neural network and its draft inputs.
- `scripts/train_carry_recommender.py` trains one network and saves its best validation checkpoint.
- `scripts/train_baseline.py` trains logistic regression as a simple reference.

**Recommendation**

- `scripts/recommend_carry.py` loads the saved NN and scores available carries.
- `scripts/check_carry_rankings.py` checks how rankings react to different drafts.

**Web app**

- `web/app.py` serves the page and handles recommendation requests.
- `web/templates/index.html` defines the page layout.
- `web/static/app.js` handles hero selection and displays recommendations.
- `web/static/style.css` controls the page appearance.

# 4. Important datasets

Unless stated otherwise, these CSVs are in `data/processed/`.

- `drafts.csv`: one row is a reconstructed match draft, used to build complete lineups.
- `training_examples.csv`: one row is a carry pick with its visible draft and result, used to train the recommender.
- `full_draft_examples.csv`: one row is a match's ten final heroes and result, used for historical hero relationships.
- `match_features.csv`: one row is a match's statistics, used for quality checks and historical duration groups.
- `player_match_features.csv`: one row is a player in a match, used for quality checks and historical farming averages.
- `training_quality.csv`: one row describes a carry example's quality, used to exclude unsuitable examples.
- `data/constants/heroes.csv`: one row describes a hero's name, attribute, attack type and roles, used for fixed model inputs.

# 5. Model

Logistic Regression -> baseline

Neural Network -> contextual recommender

Historical Feature NN -> final model

The final NN combines visible hero picks, learned hero interactions, fixed hero details and all 20 historical features. Those historical features describe win rates, farming, ally synergy, enemy matchups and their sample support; they use training history rather than the predicted match's future statistics.

- Final model: `models/final_carry_recommender.pt`
- Trainable parameters: **13,313**
- Test accuracy: **0.5345**
- Test AUC: **0.5458**

Earlier comparisons showed slightly better classification metrics for LR, but its candidate ranking does not react to the draft, while the NN's ranking does. The current trainer trains only the final NN; results are saved in `models/final_carry_training_results.json`.

# 6. Running the project

From the project root, start the web app with the current local setup:

```sh
PYTHONPATH=/private/tmp/isp-web-deps arch -arm64 .venv/bin/python -m web.app
```

Open `http://127.0.0.1:5000`. The extra Python path points to the local Flask installation.

```text
Select 4 allies -> select 4 enemies -> press Recommend
    -> receive ranked carry suggestions
```

# 7. Current limitations / future work

- Improve the model and collect more varied data; current prediction performance is modest.
- Add a preferred hero pool for the user.
- Add meta and popularity adjustments.
- Explain recommendations in simple terms.
- Add a database later if the project needs it.
