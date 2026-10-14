# Last Pick

*Your turn to pick.*

A Dota 2 carry recommendation system built using machine learning and historical match data.

The project started as my Individual Software Project (ISP) at Charles University, Faculty of Mathematics and Physics. The original goal was to build a model that could evaluate possible carry picks based on the heroes already selected by both teams.

I've continued working on it since completing the ISP, mostly focusing on improving the data pipeline, code structure, and evaluation methods.

## How it works

Given a partially completed draft, the system ranks available carry heroes by their estimated probability of winning.

The process consists of three main steps:

1. **Data collection and processing.** Match data is collected from OpenDota and processed to reconstruct drafts, identify carry players, and extract relevant information.
2. **Training.** A neural network learns from historical matches, using hero embeddings, draft context, hero attributes, and historical statistics.
3. **Recommendations.** The trained model evaluates available carry heroes for a given draft. A web interface allows users to explore the recommendations.

The model is trained on actual picks and their match outcomes. Since we don't know what would have happened if a different hero had been selected, the predicted probabilities should be interpreted with that limitation in mind.

## Results

The first version of the project was completed as an ISP. It includes a trained neural network, a data processing pipeline, and a working web interface.

The model was evaluated on a held-out test set:

| Metric | Result |
|---|---:|
| Training samples | 73,558 |
| Validation samples | 12,275 |
| Test samples | 22,205 |
| Accuracy | 53.45% |
| ROC-AUC | 0.5458 |
| Log loss | 0.6900 |

The results are fairly modest. Draft information alone doesn't capture everything that determines the outcome of a match, and there are limitations in how roles and draft decisions are reconstructed from the available data.

The original implementation and its results are preserved under the [`isp-v1`](../../tree/isp-v1) tag.

## Development

After finishing the ISP, I started revisiting parts of the implementation that made further experimentation difficult.

The current version includes changes to the project structure, match processing, role inference, and dataset generation. I've also been working on making evaluation more reproducible and avoiding information leakage from future matches.

These changes are mostly to the underlying infrastructure. The web interface still uses the original ISP model, and I haven't trained a replacement model on the updated pipeline yet.

## Running locally

Requires Python 3.11 or newer.

Clone the repository and install the dependencies:

```bash
python -m pip install -e '.[dev,web]'
```

Start the web interface:

```bash
python -m web.app
```

Run the tests:

```bash
python -m pytest -q
```

The original trained model is included in the repository. The historical datasets used for training are not, so reproducing the original experiment requires obtaining and processing the match data separately.

## Documentation

More detailed notes on the implementation and experiments are available in the `docs/` directory:

- [Technical documentation](docs/DOCUMENTATION.md)
- [Data pipeline and architecture](docs/FOUNDATION_V2.md)
- [Original ISP experiment](docs/ISP_V1.md)

## What's next

There are several things I'd like to explore further, particularly stronger statistical baselines, better matchup and synergy features, and accounting for changes between game patches.

For now, the focus is on making experiments easier to run and compare before spending more time on model complexity.

---

This is a personal educational project and is not affiliated with Valve.
