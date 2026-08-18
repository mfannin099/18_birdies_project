# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Personal data science project analyzing the user's golf scores exported from the 18Birdies app. It has three loosely-connected parts that don't share code:

1. **`score_analysis_app/`** — the active, primary deliverable. A Streamlit dashboard (`app.py`) that a user uploads their 18Birdies JSON export into and gets score trends, distributions, and per-hole breakdowns.
2. **`classification_model/`** — a from-scratch golf-round simulator (`golf_sim.py`, using SimPy) that generates synthetic score data, plus a FastAPI service (`main.py`) that serves a `DecisionTreeClassifier` trained on that data to predict golfer skill level (low/mid/high handicap) from `hole_yardage`, `hole_par`, `player_score`.
3. **`data_ingestion/` + `notebooks/`** — one-off/exploratory scripts and notebooks (Bayesian regression with PyMC, EDA) that produced the artifacts the other two parts consume (`sim_golf_model.pkl`, `sim_golf_scores_for_ml.csv`).

There are two separate `utils.py` files with unrelated contents — don't assume they're the same module:
- `score_analysis_app/utils.py` — data cleaning/plotting for the Streamlit app (Polars-based).
- `utils.py` (repo root) — helper functions and the `Golfer`/`Course`/`play_round` simulation classes used by `classification_model/golf_sim.py`.

## Package management

The repo uses **uv**, with a single root-level `pyproject.toml` covering the whole project (not per-subfolder). `score_analysis_app/requirements.txt` and `classification_model` still have implicit pip-era dependencies that aren't yet reflected in `pyproject.toml` beyond what `score_analysis_app/app.py` needs — add packages with `uv add <package>` as new code needs them rather than hand-editing `pyproject.toml`.

```bash
uv sync                 # install/update the environment from pyproject.toml / uv.lock
uv add <package>         # add a new dependency
uv run <command>          # run any command inside the project's venv
```

## Running the Streamlit app

```bash
uv run streamlit run score_analysis_app/app.py
```

This is the primary entry point for development. The app expects a JSON file uploaded through its file picker (exported from 18Birdies' account data download) — there's no sample fixture in the repo, so manual testing requires a real or hand-crafted export matching the shape consumed by `clean_data()` in `score_analysis_app/utils.py` (`myData.activityData.rounds` and `myData.clubData.playedClubs`).

A `Dockerfile` exists under `score_analysis_app/` for containerized deployment; it currently installs from `requirements.txt` via pip rather than uv.

## Running the classification API (classification_model)

```bash
uv run uvicorn classification_model.main:app --reload
```

Loads a pickled model from `notebooks/sim_golf_model.pkl` (path is relative — run from repo root). Exposes `GET /predict?hole_yardage=&hole_par=&player_score=` returning a predicted handicap bucket.

## Architecture notes for `score_analysis_app`

- **Data flow**: raw 18Birdies JSON → `clean_data()` (`utils.py`) parses `rounds` and `playedClubs` from the JSON, joins them, derives `course_par` and an `18 Holes Played Flg` (from total strokes vs. a threshold, default 70), renames columns to display-friendly names, and drops Par-3 courses reported as 9-hole rounds. Everything downstream in `app.py` operates on this cleaned Polars DataFrame.
- **Polars vs. pandas**: data manipulation stays in Polars for as long as possible; conversion `.to_pandas()` happens only at the point of feeding Altair charts or `statsmodels`, since those libraries expect pandas/NumPy.
- **`app.py` is organized as three tabs** (Score Over Time, Score Distribution, Score Per Hole), each a self-contained block operating on `df_sorted` / `full_rounds_df` computed earlier in the file — there's no separation into functions per tab, so changes to one section can silently depend on state computed above it (e.g., `full_rounds_df` with the `z`-score and `Round Index` columns added is reused across tab 2 and tab 3).
- Regression/statistics helpers (`fit_linear_regression`, `make_random_walk`) live in `score_analysis_app/utils.py` and are called both directly and indirectly (via `plot_time_series`) from `app.py`.
