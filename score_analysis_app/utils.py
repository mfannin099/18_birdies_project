import polars as pl
import json
import pandas as pd
from pandas import json_normalize
import altair as alt
import streamlit as st
import numpy as np

cols_to_del = ['holeStrokes', # This is an array of ints of what you scored on each hole
        'stats_fairwayLefts','stats_fairwayMiddles', 'stats_fairwayRights', 'stats_fairwayShorts',
        'stats_fairwayLongs', 'stats_fairwayHoleCount', 'stats_gir',
        'stats_girLefts', 'stats_girRights', 'stats_girShorts',
        'stats_girLongs', 'stats_girNoChances', 'stats_girHoleCount',
        'stats_strokeGainOverall', 'stats_strokeGainTeeToGreen',
        'id', 'clubId_id']

def clean_data(json_file, holes_for_18_calc_val = 70):

    ##Opening the json
    data = json.load(json_file)

    ##Grabbing the rounds data
    rounds_data = json_normalize(data['myData']['activityData']['rounds'], 
                                sep='_', 
                                record_prefix='round_')

    rounds_data_pl = pl.DataFrame(rounds_data)

    ##Grabbing the courses played at
    course_data = json_normalize(data['myData']['clubData']['playedClubs'], 
                             sep='_', 
                             record_prefix='round_')
    
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
    final_df_pl = final_df_pl.drop(columns=cols_to_del)

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

    # Melt dataframe into long format
    plot_df_melted = plot_df.melt(
        id_vars=["Round Played"],
        value_vars=["Total Strokes", "Rolling Avg (10 Rounds)", "Random Walk"],
        var_name="Metric",
        value_name="Value"
    )

    # --- Streamlit UI filter ---
    metric_options = ["Total Strokes", "Rolling Avg (10 Rounds)", "Random Walk"]
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        options=metric_options,
        default=metric_options,
        key=f"metric_selector_{holes_played_val}"  # 👈 unique key here
    )

    # Filter based on user selection
    plot_df_melted = plot_df_melted[plot_df_melted["Metric"].isin(selected_metrics)]

    if plot_df_melted.empty:
        st.warning("Please select at least one metric to display.")
        return

    # Color scale
    color_scale = alt.Scale(
        domain=["Total Strokes", "Rolling Avg (10 Rounds)", "Random Walk"],
        range=["#1f77b4", "orange", "green"]
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
        .configure_title(fontSize=16, anchor="start", fontWeight="bold")
        .configure_view(strokeWidth=0)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def make_random_walk(df):

    len_of_walk = len(df)
    avg = df['Total Strokes'].mean()
    start_value = df['Total Strokes'].head(1).item()
    std_dev = (df['Total Strokes'].std()) * .8

    # Generate random steps
    steps = np.random.normal(loc=0, scale=std_dev, size=len_of_walk-1)
   
    # Prepend the start_value
    random_walk = np.concatenate([[start_value], start_value + np.cumsum(steps)])

    # Add as a new column in the DataFrame
    df['Random Walk'] = random_walk

    return df





