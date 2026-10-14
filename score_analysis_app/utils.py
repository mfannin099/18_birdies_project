import polars as pl
import json
import pandas as pd
from pandas import json_normalize
import altair as alt
import streamlit as st
import numpy as np
import statsmodels.api as sm

cols_to_del = ['holeStrokes', # This is an array of ints of what you scored on each hole
        'stats_fairwayLefts','stats_fairwayMiddles', 'stats_fairwayRights', 'stats_fairwayShorts',
        'stats_fairwayLongs', 'stats_fairwayHoleCount', 'stats_gir',
        'stats_girLefts', 'stats_girRights', 'stats_girShorts',
        'stats_girLongs', 'stats_girNoChances', 'stats_girHoleCount',
        'stats_strokeGainOverall', 'stats_strokeGainTeeToGreen',
        'stats_putts', 'stats_puttHoleCount', 'stats_recommendedStats',
        'id', 'clubId_id']

# Fields clean_data() reads off each round/club record, keyed by the exact
# path the 18Birdies export uses. Checked up front so a changed/incomplete
# export fails with one clear message instead of a Polars traceback deep
# inside the join/rename pipeline.
ROUND_REQUIRED_FIELDS = ['timestamp', 'score', 'strokes', 'clubId']
ROUND_STATS_REQUIRED_FIELDS = [
    'aces', 'doubleEagleOrBetter', 'eagles', 'birdies', 'pars',
    'bogeys', 'doubleBogeyOrWorse',
]
CLUB_REQUIRED_FIELDS = ['clubId', 'name']


class SchemaValidationError(ValueError):
    """Raised when an uploaded 18Birdies export is missing fields clean_data() relies on."""


def validate_18birdies_schema(data):
    try:
        rounds = data['myData']['activityData']['rounds']
        played_clubs = data['myData']['clubData']['playedClubs']
    except (KeyError, TypeError) as exc:
        raise SchemaValidationError(
            "Expected top-level keys myData.activityData.rounds and "
            "myData.clubData.playedClubs were not found. Is this a 18Birdies "
            "account data export?"
        ) from exc

    if not rounds:
        raise SchemaValidationError("myData.activityData.rounds is empty — no rounds to analyze.")

    for i, round_record in enumerate(rounds):
        missing = [f for f in ROUND_REQUIRED_FIELDS if f not in round_record]
        if missing:
            raise SchemaValidationError(f"Round {i} is missing field(s): {missing}")
        if 'id' not in round_record.get('clubId', {}):
            raise SchemaValidationError(f"Round {i}'s clubId is missing 'id'.")
        stats_missing = [f for f in ROUND_STATS_REQUIRED_FIELDS if f not in round_record.get('stats', {})]
        if stats_missing:
            raise SchemaValidationError(f"Round {i}'s stats is missing field(s): {stats_missing}")

    for i, club_record in enumerate(played_clubs):
        missing = [f for f in CLUB_REQUIRED_FIELDS if f not in club_record]
        if missing:
            raise SchemaValidationError(f"playedClubs entry {i} is missing field(s): {missing}")


def clean_data(json_file, holes_for_18_calc_val = 70):

    ##Opening the json
    data = json.load(json_file)
    validate_18birdies_schema(data)

    ##Grabbing the rounds data
    rounds_data = json_normalize(data['myData']['activityData']['rounds'],
                                sep='_')

    rounds_data_pl = pl.DataFrame(rounds_data)

    ##Grabbing the courses played at
    course_data = json_normalize(data['myData']['clubData']['playedClubs'],
                             sep='_')

    club_data_pl = pl.DataFrame(course_data)

    ##joining the data
    final_df_pl = rounds_data_pl.join(club_data_pl, left_on='clubId_id', right_on='clubId', how='left')


    ##beginning to clean the columns 
    ##cleaning the timestamp 
    final_df_pl = final_df_pl.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).dt.strftime("%Y-%m-%d").alias("timestamp")
    )

    ##calculating par for the course (whether its 9 or 18 holes)
    final_df_pl = final_df_pl.with_columns(
        (pl.col("strokes") - pl.col("score").abs()).alias("course_par")
    )

    ##calculating the number of holes played based on the score of the round 
    final_df_pl = final_df_pl.with_columns(
        pl.when(pl.col("strokes") < holes_for_18_calc_val)
        .then(0)
        .otherwise(1)
        .alias("is_18_holes")
    )

    #Using is_18_holes == 0 and course_par < 30 to drop par 3 course scores
    final_df_pl = final_df_pl.filter(~((pl.col("is_18_holes") == 0) & (pl.col("course_par") <= 30)))
                                     
    # Dropping unused columns
    final_df_pl = final_df_pl.drop(cols_to_del)

    # Renaming columns to make easier to display
    final_df_pl = final_df_pl.rename({"timestamp": "Round Played", "score" : "Strokes Over Par", "strokes":"Total Strokes",
                                      "stats_aces": "Aces", "stats_doubleEagleOrBetter" : "Albatrosses", 'stats_eagles':"Eagles",
                                      "stats_birdies": "Birdies", "stats_pars": "Pars", "stats_bogeys":"Bogeys",
                                      "stats_doubleBogeyOrWorse": "Double Bogey or Worse", 'name': 'Course',
                                      'course_par': "Course Par", "is_18_holes": "18 Holes Played Flg"
                                      
                                      })


    return final_df_pl


def plot_time_series(df, holes_played_flg, holes_played_val):
    # Filter based on holes played
    filtered_df = df.filter(pl.col("18 Holes Played Flg") == holes_played_flg)

    # Convert to Pandas for Altair
    plot_df = filtered_df.to_pandas().sort_values("Round Played")

    # Compute rolling 10-round average
    plot_df["Rolling Avg (10 Rounds)"] = (
        plot_df["Total Strokes"]
        .rolling(window=10, min_periods=1)
        .mean()
    )

    # Compute Random Walk
    plot_df = make_random_walk(plot_df)

    # Compute Linear Regression
    plot_df, model = fit_linear_regression(plot_df)

    # Melt dataframe into long format
    plot_df_melted = plot_df.melt(
        id_vars=["Round Played"],
        value_vars=["Total Strokes", "Rolling Avg (10 Rounds)", "Random Walk", "Linear Regression"],
        var_name="Metric",
        value_name="Value"
    )

    # --- Streamlit UI filter ---
    metric_options = ["Total Strokes", "Rolling Avg (10 Rounds)", "Random Walk", "Linear Regression"]
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        options=metric_options,
        default=metric_options,
        key=f"metric_selector_{holes_played_val}" 
    )

    # Filter based on user selection
    plot_df_melted = plot_df_melted[plot_df_melted["Metric"].isin(selected_metrics)]

    if plot_df_melted.empty:
        st.warning("Please select at least one metric to display.")
        return

    # Color scale
    color_scale = alt.Scale(
        domain=["Total Strokes", "Rolling Avg (10 Rounds)", "Random Walk", "Linear Regression"],
        range=["#1f77b4", "orange", "green", "red"]
    )

    # Legend toggle selection
    legend_selection = alt.selection_point(fields=["Metric"], bind="legend")

    # Chart
    chart = (
        alt.Chart(plot_df_melted)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))
        .encode(
            x=alt.X("Round Played:O", title="Round Played",
                axis=alt.Axis(
                    labelAngle=-45,      # rotate labels diagonally
                    labelFontSize=12,    # smaller font size
                    titleFontSize=13,
                    labelAlign="right"   # align text to right
                )
            ),
            y=alt.Y("Value:Q", title="Total Strokes"),
            color=alt.Color("Metric:N", scale=color_scale, title="Metric"),
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("Round Played", title="Round"),
                alt.Tooltip("Metric", title="Metric"),
                alt.Tooltip("Value", title="Strokes", format=".1f"),
            ],
        )
        .add_params(legend_selection)
        .properties(
            title=f"Total Strokes per Round ({holes_played_val} Holes Only)",
            width=700,
            height=500,
        )
        .configure_title(fontSize=24, anchor="start", fontWeight="bold")
        .configure_view(strokeWidth=0)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def make_random_walk(df):

    len_of_walk = len(df)
    avg = df['Total Strokes'].mean()
    start_value = df['Total Strokes'].head(1).item()
    std_dev = (df['Total Strokes'].std()) * .4

    # Generate random steps
    steps = np.random.normal(loc=0, scale=std_dev, size=len_of_walk-1)
   
    # Prepend the start_value
    random_walk = np.concatenate([[start_value], start_value + np.cumsum(steps)])

    # Add as a new column in the DataFrame
    df['Random Walk'] = random_walk

    return df

def fit_linear_regression(df):

    X_numeric = np.arange(len(df))
    X = sm.add_constant(X_numeric)
    y = df["Total Strokes"].to_numpy()

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    df['Linear Regression'] = predictions
    
    return df, model


def calculate_handicap_index(df, n_recent=20, n_best=8, min_rounds=8):
    """Estimate a WHS-style Handicap Index: average of the lowest `n_best`
    differentials among the `n_recent` most recent 18-hole rounds.

    The export has no Course Rating / Slope, so the differential is
    approximated as Total Strokes - Course Par (slope 113, PCC 0). The result
    is therefore an estimate, not an official index.

    Returns (index, counting_rounds, n_rounds_considered). `index` is None
    when fewer than `min_rounds` 18-hole rounds exist; `counting_rounds` is a
    Polars DataFrame of the rounds that went into the average.
    """
    recent = (
        df.filter(pl.col("18 Holes Played Flg") == 1)
        .sort("Round Played", descending=True)
        .head(n_recent)
        .with_columns((pl.col("Total Strokes") - pl.col("Course Par")).alias("Differential"))
    )
    n_considered = recent.height
    if n_considered < min_rounds:
        return None, recent.clear(), n_considered

    counting = recent.sort("Differential").head(n_best)
    index = round(counting["Differential"].mean(), 1)
    return index, counting.select(["Round Played", "Course", "Total Strokes", "Course Par", "Differential"]), n_considered


def monthly_scoring_summary(df, holes_played_flg=1):
    """Average score per calendar month (Jan-Dec, all years pooled) and per
    year-month. Expects `df` to have a Datetime `Round Played` plus the
    `Month`/`Year` columns app.py adds. Returns (by_calendar_month, by_year_month).
    """
    rounds = df.filter(pl.col("18 Holes Played Flg") == holes_played_flg)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    by_calendar_month = (
        rounds.group_by("Month")
        .agg(
            pl.col("Total Strokes").mean().round(1).alias("Average Score"),
            pl.len().alias("Rounds"),
        )
        .sort("Month")
        .with_columns(pl.col("Month").map_elements(lambda m: month_names[m - 1], return_dtype=pl.String).alias("Month Name"))
    )

    by_year_month = (
        rounds.with_columns(pl.col("Round Played").dt.strftime("%Y-%m").alias("Year-Month"))
        .group_by("Year-Month")
        .agg(
            pl.col("Total Strokes").mean().round(1).alias("Average Score"),
            pl.len().alias("Rounds"),
        )
        .sort("Year-Month")
    )
    return by_calendar_month, by_year_month


def personal_bests(df, streak_len=3):
    """Best 18-hole round, best 9-hole round, best consecutive `streak_len`-round
    stretch of 18-hole rounds (lowest average), and most birdies in a round.
    Each entry is a dict, or None when there isn't enough data.
    """
    def best_round(flg):
        rounds = df.filter(pl.col("18 Holes Played Flg") == flg)
        if rounds.is_empty():
            return None
        row = rounds.sort(["Total Strokes", "Round Played"]).row(0, named=True)
        return {"strokes": row["Total Strokes"], "date": str(row["Round Played"])[:10], "course": row["Course"]}

    full = df.filter(pl.col("18 Holes Played Flg") == 1).sort("Round Played")
    stretch = None
    if full.height >= streak_len:
        rolled = full.with_columns(
            pl.col("Total Strokes").rolling_mean(window_size=streak_len).alias("_avg")
        )
        end = rolled["_avg"].arg_min()
        window = full.slice(end - streak_len + 1, streak_len)
        stretch = {
            "avg": round(rolled["_avg"][end], 1),
            "start": str(window["Round Played"][0])[:10],
            "end": str(window["Round Played"][-1])[:10],
            "scores": window["Total Strokes"].to_list(),
        }

    most_birdies = None
    if not df.is_empty():
        row = df.sort(["Birdies", "Round Played"], descending=[True, False]).row(0, named=True)
        most_birdies = {"birdies": row["Birdies"], "date": str(row["Round Played"])[:10], "course": row["Course"]}

    return {
        "best_18": best_round(1),
        "best_9": best_round(0),
        "best_stretch": stretch,
        "stretch_len": streak_len,
        "most_birdies": most_birdies,
    }


def create_score_df(df_sorted):

    # Calculating Measures for Calculations
    total_strokes = df_sorted.select(pl.col("Total Strokes").sum()).item()
    total_aces = df_sorted.select(pl.col("Aces").sum()).item()
    total_alba = df_sorted.select(pl.col("Albatrosses").sum()).item()
    total_eagles = df_sorted.select(pl.col("Eagles").sum()).item()
    total_birides = df_sorted.select(pl.col("Birdies").sum()).item()
    total_pars = df_sorted.select(pl.col("Pars").sum()).item()
    total_bogeys = df_sorted.select(pl.col("Bogeys").sum()).item()
    total_bogeys_double = df_sorted.select(pl.col("Double Bogey or Worse").sum()).item()


    scoring_df = pl.DataFrame({
    "Score Type": ["Aces", "Albatrosses", "Eagles", "Birdies", "Pars", "Bogeys", "Double Bogey or Worse"],
    "Count": [
        total_aces,
        total_alba,
        total_eagles,
        total_birides,
        total_pars,
        total_bogeys,
        total_bogeys_double
    ]
})

    return scoring_df







