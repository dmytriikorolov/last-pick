# ISP v1 baseline

The completed, credited ISP submission is commit `7dd54e2`, tagged `isp-v1`.
Check out that tag in a separate checkout to reproduce the original implementation.
Foundation v2 retains its checkpoint, results, hero constants, portraits and UI.

## Dataset and objective

OpenDota ranked All Pick (mode 22, lobby 7), principally high rank matches,
was cached in `data/raw/opendota/matches`. Final played teams recover ten
successful picks from overfull logs. Failed attempts in the candidate's phase
exclude that decision. The carry is the more farmed of exactly two safelane
players, with at least 15 LH and 400 gold separation at minute 10.
Team picks 3, 4 and 5 have visibility 2/2, 3/2 and 4/4 respectively;
enemy picks in the current simultaneous phase stay hidden.

`training_examples.csv` labels the historical chosen carry with its team's
win/loss, never labels unpicked alternatives. Complete drafts and player CSVs
provide historical statistics. Time splits keep equal timestamps together;
five training blocks use earlier training matches, and validation/test use a
frozen training history. These files and raw data are gitignored: the tag alone
does not reproduce the dataset without the saved local data/source snapshots.

## Saved model and reference results

`models/final_carry_recommender.pt` is an embedding MLP with candidate/visible
team means, multiplicative interactions, 14 static hero attributes and 20
historical features. It has 13,313 trained parameters. Its 44 candidates have
at least 100 retained training examples; recommendation scores legal candidates
by estimated win probability. The saved result JSON is authoritative:

| Measure | Value |
| --- | ---: |
| Training rows | 73,558 |
| Validation rows | 12,275 |
| Test rows | 22,205 |
| Test accuracy | 0.5345192524 |
| Test ROC-AUC | 0.5457609609 |
| Test log loss | 0.6899946928 |
| Test Brier | 0.2484227717 |
| Best epoch | 2 |
| Validation log loss | 0.6915258765 |

## Limits and preservation

Role labels are heuristic; contexts are mixed; bans were discarded from v1
CSVs; source rank metadata was not consistently retained. Aggregate CSVs are
descriptive and unsafe as training inputs without temporal filtering. The v1
trainer mistakenly used equal weights after removing zero-weight rows. Its
saved model reflects that behavior; corrected future training cannot reproduce
these exact metrics. Outcomes do not identify counterfactual recommendation
quality, and metrics are modest. Do not replace this checkpoint or results to
claim an improvement without a separate, reproducible experiment.

Before Foundation changes, the existing suite passed: **19 tests**. No training
or data regeneration was performed during baseline preservation.
