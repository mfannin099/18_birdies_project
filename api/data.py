# Generates synthetic round data shaped like score_analysis_app.utils.clean_data()'s
# output columns (Aces/Albatrosses/Eagles/Birdies/Pars/Bogeys/Double Bogey or Worse
# + Course Par), for training a Total Strokes prediction model. There's no real
# curated 18Birdies dataset to train on, so this approximates one: each scoring
# category is assigned a strokes-relative-to-par value, per-hole category
# probabilities are varied by a synthetic skill level, and Total Strokes is
# built up from Course Par plus those category counts plus noise.
import numpy as np
import pandas as pd

RELATIVE_STROKES = {
    "Aces": -3.0,
    "Albatrosses": -3.0,
    "Eagles": -2.0,
    "Birdies": -1.0,
    "Pars": 0.0,
    "Bogeys": 1.0,
    "Double Bogey or Worse": 2.2,
}
CATEGORIES = list(RELATIVE_STROKES.keys())

# Per-hole probability of landing in each category above, roughly calibrated
# so "low" handicap golfers post far more pars/birdies and far fewer blow-up
# holes than "high" handicap golfers.
SKILL_PROFILES = {
    "low":  [0.0010, 0.0010, 0.0200, 0.1200, 0.5500, 0.2500, 0.0580],
    "mid":  [0.0005, 0.0005, 0.0050, 0.0500, 0.4000, 0.3500, 0.1940],
    "high": [0.0002, 0.0002, 0.0010, 0.0200, 0.2200, 0.3500, 0.4086],
}


def generate_training_data(n_rounds: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skills = list(SKILL_PROFILES.keys())
    rows = []

    for _ in range(n_rounds):
        skill = rng.choice(skills)
        holes = int(rng.choice([9, 18], p=[0.3, 0.7]))
        course_par = round(float(rng.normal((holes / 18) * 72, 2)))

        counts = rng.multinomial(holes, SKILL_PROFILES[skill])
        counts_dict = dict(zip(CATEGORIES, counts))

        noise = rng.normal(0, 1.5)
        total_strokes = course_par + sum(
            counts_dict[c] * RELATIVE_STROKES[c] for c in CATEGORIES
        ) + noise
        total_strokes = max(holes, round(total_strokes))

        rows.append({**counts_dict, "Course Par": course_par, "Total Strokes": total_strokes})

    return pd.DataFrame(rows)
