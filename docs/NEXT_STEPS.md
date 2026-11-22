# repo review and next steps

Reviewed on 2026-11-22 at `027c96f4`. LastPick has a working ISP demo and a
separate research pipeline. The next question is whether the models still beat
simple baselines a week later or after a patch.

## what the repo has

OpenDota match details go through one adapter into canonical match records. The
dataset builder reconstructs what each team could see before its pick and uses
early farm to infer the carry. The benchmark reads the resulting 4v4 decisions
and builds its historical features from completed training matches. Flask and
the static site serve the saved ISP model. [Pipeline](FOUNDATION_V2.md) ·
[Benchmark](MODELING_V2.md) · [Demo](../web/README.md)

Graphify found 651 nodes and 1,802 links. The main code paths I followed were
`Collector` and its restart tests, `benchmark()` through `HistoricalFeatures`
to `AsOfHistory`, and Flask through `rank_candidates()`. I checked the source
and local dataset for the findings below. [Graph commands](DOCUMENTATION.md#code-map)

## what is missing

### labels need their own time cutoff

The benchmark splits on match start. Historical features already require
completion before their cutoff, but fitting, model selection and calibration
receive every row in their assigned partitions. I joined the saved 4v4 decisions
to canonical match durations and counted matches with `start_time + duration`
at or after the next partition's start.

| Boundary | Matches still in progress or finishing at the cutoff | 4v4 rows |
| --- | ---: | ---: |
| Main training to validation | 137 | 165 |
| Main validation to test | 508 | 637 |
| June training to validation | 314 | 368 |

For example, the primary sigmoid calibrator can use outcomes that would still
be unknown when the test window starts. This check measures how many rows cross
the cutoff. The effect on the reported scores needs a rerun.
[Benchmark code](../src/lastpick/experiments/benchmark.py) ·
[Historical features](../src/lastpick/experiments/features.py)

### the dataset is large but covers seven UTC dates

The 44,329 decisions fall on June 4–7 and August 22–24, with a 76.55-day gap.
OpenDota calls all of them patch 60 (7.41), but Valve released 7.41d on June 5 and
7.41e on July 30. The windows therefore span balance changes as well as a gap
in collection. The June-to-August test cannot tell us which change drove the
drop. Valve released 7.42 on November 18, and OpenDota maps it to ID 61. That
gives us a new patch to collect and check. [OpenDota patch IDs](https://raw.githubusercontent.com/odota/dotaconstants/master/build/patch.json) ·
[Valve 7.41d](https://www.dota2.com/news/updates/9016/) ·
[Valve 7.41e](https://www.dota2.com/summerscrub2026) ·
[Valve 7.42](https://www.dota2.com/newsentry/689773669992367212)

The collector resumes a backward crawl using one cursor. That works for
backfill. To keep picking up fresh matches, it needs to revisit the newest
page and stop at previously covered matches, or use bounded date windows with
Explorer if a one-page check confirms that route gives useful coverage. The
provider has a `start_time` column and index on `public_matches`. Weekly counts
should show where we got usable drafts, where telemetry was missing, and where
we collected nothing. Live detail coverage for 7.42 still needs to be measured.
[Collector](../src/lastpick/ingestion/collector.py) ·
[OpenDota schema](https://github.com/odota/core/blob/master/sql/create_tables.sql)

### the probability result and the ranking result answer different questions

June-fitted LightGBM's August log loss rose from 0.68919 to 0.69011 after
calibration. In the main benchmark, context LR put Bloodseeker first in 99.49%
of test drafts while raw CatBoost put Phantom Lancer first in 92.85%. Keep
measuring how often draft changes move the rankings alongside log loss, Brier
score and calibration on each later window. Calibration can degrade when the
data change, even if it worked on the window used to fit it.
[Benchmark results](MODELING_V2.md) ·
[Ovadia et al. on calibration under shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

We see the match result for the hero that was picked. We also lack reliable
ban timing and exact candidate sets in this build. Report observed-pick
agreement and model pool coverage as such. Estimating whether a different
pick would win more often needs suitable feedback and evidence about how picks
were made. [Swaminathan and Joachims on logged feedback](https://proceedings.mlr.press/v37/swaminathan15.html)

### carry labels still need a human check

The 400-case review pack has blank human labels. Start with 30 cases across
high confidence, medium confidence and abstentions. Have two reviewers record
the carry or mark the case unclear before seeing the heuristic's answer, where
independent evidence is available. Report reviewer agreement, heuristic
agreement on reviewed picks, and how often it abstains. The balanced pack
overrepresents some cases, so any estimate for the whole dataset needs sampling
weights or a separate representative sample. [Review pack](../reports/role_review_sample.csv)

### the demo could be checked more easily

I ran 50 seeded drafts through Flask and the committed static site model in
Node. All top-five orders matched and the largest score difference was about
`6.1e-8`. Add that small parity check when the export or inference code next
changes. The demo could also say which model it serves and show the supported
carry pool so a missing hero is easier to understand. [Demo setup](../web/README.md)

## plan

1. **Fix the benchmark cutoffs first.** Reuse the canonical durations. Require
   completed outcomes before training, tuning and calibration can use them at
   their respective boundaries. Keep both sides of a match and equal start
   timestamps together. Test the exact cutoff, unknown completion and a match
   still in progress. Report excluded counts and rerun the existing splits with
   the same seed and model settings. Keep the previous results for comparison.
2. **Start collecting across weeks in parallel.** Use one bounded OpenDota
   page to check 7.42 draft visibility, rank and farm coverage, and requests per
   usable decision. Record UTC coverage, patch and skip reasons. Keep fresh
   collection and backfill cursors distinct and retain raw payloads. Include
   Valve release boundaries in reports when a source patch ID covers several
   balance updates.
3. **Compare the same small model set on several later windows.** Start with a
   global prior, a hero/side prior, context LR and one existing tree model. Test
   an expanding training window against a recent window, with raw and
   calibrated scores. Use complete weeks where coverage allows. Report log
   loss, Brier, calibration, pool coverage and top-pick concentration by
   window. Show paired uncertainty by match and leave a later window untouched
   for the final comparison. [Rolling-origin evaluation](https://otexts.robjhyndman.com/fpp3/toolbox.html#time-series-cross-validation)
4. **Run the role review alongside the data work.** Use the pilot to agree on
   the rubric, then work through the review pack where evidence is available.
   Report how much of the sample could be reviewed before deciding whether a
   learned role label is worth pursuing.

I would revisit richer matchup features and a demo model update after the
existing models show a repeatable gain on later windows and rank more than a
handful of heroes for sensible draft changes. A larger network, live drafts and
personalization can wait for that result.

## update

The benchmark now checks completion before using training and validation
outcomes. Both corrected runs are in [Modeling v2](MODELING_V2.md). The weekly
benchmark found zero eligible full weeks. A blinded 30-case carry review pack
is ready for reviewers.

Two small 7.42 pilots ran without a key. Public discovery used 50 requests and
returned 49 match details, with zero accepted by the dataset rules. Filtered
Explorer discovery used 25 requests and returned 24 details, with 18 accepted
matches and nine 4v4 decisions. Twelve of the accepted matches had both
minute-10 farm measures for all ten players. The pilots used separate caches.
[Anonymous collection commands](DOCUMENTATION.md#anonymous-collection-and-weekly-checks)

My Dota adds an activity calendar, recent matches and carry favorites. The
draft tool adds saved drafts, share links, tap-friendly bans, and an optional
favorite carry filter. The website still serves the ISP model.
[Demo setup](../web/README.md)

## how to check whether the suggestions help

1. **Test features players can explain.** Make a patch-dated table of hero
   abilities such as reliable lockdown, hard dispel, illusion clear and tower
   damage. Compare a regularized model with separate candidate–ally and
   candidate–enemy terms against the hero-and-side prior. Use only facts
   available at draft time. Keep player comfort as a user-controlled filter
   until enough completed games support a player-hero estimate.
   [Valve hero abilities](https://www.dota2.com/hero/DrowRanger?l=english) ·
   [Valve patch notes](https://www.dota2.com/patches/69) ·
   [Deep Sets](https://arxiv.org/abs/1703.06114)
2. **Test whether the shortlist is useful.** Show experienced players 30–50
   blinded drafts with one shortlist from a simple baseline and one from
   LastPick. Ask for a preference, a reason, or an abstention.
   [DraftRec study](https://arxiv.org/abs/2204.12750)
3. **Test what happens when players see the advice.** If enough users opt in,
   record the legal heroes, visible draft, declared carry pool, suggestions
   shown, selected hero, and later result. Randomly show LastPick or a simple
   baseline. Compare results by assigned group, including players who ignore
   the advice. Record assignment probabilities and missing results. This can
   estimate whether showing suggestions helps. The outcome for an unpicked
   hero remains unknown.
   [Logged recommendation evaluation](https://arxiv.org/abs/1003.5956) ·
   [Doubly robust evaluation](https://icml.cc/2011/papers/554_icmlpaper.pdf)
