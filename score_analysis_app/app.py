import streamlit as st
import pandas as pd
import polars as pl
import altair as alt
import numpy as np
import scipy.stats as stats


from utils import clean_data, plot_time_series, fit_linear_regression, create_score_df

st.set_page_config(
    page_title="Golf Performance Dashboard",
    page_icon="⛳",                     
    layout="wide",                
    initial_sidebar_state="expanded" 
)

# Before user uploads files
st.title("18 Birdies Golf Score Analysis" )

uploaded_file = st.file_uploader("Choose a JSON file - Download from 18Birdies", type=["json"])

st.markdown(
    """
    Don't have your file yet?  
    👉 [Download your golf round data from 18Birdies here](https://18birdies.com/download-account-data//)
    """,
    unsafe_allow_html=True
)
st.write("")
st.write("")


# After user has uploaded the file

if uploaded_file is not None:
    df = clean_data(uploaded_file)

    # Disply dataframe to look at rounds
    df_sorted = df.sort("Round Played", descending=False)
    st.dataframe(df_sorted.to_pandas(), use_container_width=True)
    st.caption("Note: Par 3 scores are excluded.")

    # Create month/year for future use
    df = df.with_columns(
    pl.col("Round Played")
    .str.strptime(pl.Datetime, "%Y-%m-%d")
)
    df = df.with_columns([
    pl.col("Round Played").dt.month().alias("Month"),
    pl.col("Round Played").dt.year().alias("Year")
])
    
    st.write("")
    st.write("")

############################################################ BEGIN Metrics

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            label="Average Score (18 Holes)",
            value=f"{df.filter(pl.col('18 Holes Played Flg') == 1).select(pl.mean('Total Strokes')).item():.1f}"
        )

    n=10
    with col2:
        avg_last_n = (
            df_sorted
            .filter(pl.col('18 Holes Played Flg') == 1)   # only 18-hole rounds
            .sort('Round Played', descending=True)        # most recent first
            .head(n)                                     # take top n
            .select(pl.mean('Total Strokes'))             # compute mean
            .item()
        )

        st.metric(
            label=f"Average Score (18 Holes) - Last {n} Rounds",
            value=f"{avg_last_n:.1f}"
        )

    with col3:
        st.metric(
            label = "Average Score (9 Holes)",
            value=f"{df.filter(pl.col('18 Holes Played Flg') == 0).select(pl.mean('Total Strokes')).item():.1f}"
        )

    with col4:
        avg_last_n = (
            df_sorted
            .filter(pl.col('18 Holes Played Flg') == 0)   # only 9-hole rounds
            .sort('Round Played', descending=True)        # most recent first
            .head(n)                                     # take top n
            .select(pl.mean('Total Strokes'))             # compute mean
            .item()
        )

        st.metric(
            label=f"Average Score (9 Holes) - Last {n} Rounds",
            value=f"{avg_last_n:.1f}"
        )

    with col5:
        st.metric(
            label="Rounds Logged",
            value=len(df)
        )
    
    with col6:
        st.metric(
            label="Rounds Logged (18 Holes)",
            value=len(df.filter(pl.col('18 Holes Played Flg') == 1))
        )

############################################################ END Metrics

    st.write("")
    st.write("")
    st.markdown("---")  # simple horizontal line

    tab1,tab2, tab3 = st.tabs(['Score Over Time', 'Score Distubution', "Score Per Hole"])

############################################################ BEGIN Score over time

    with tab1:
        plot_time_series(df_sorted,1,18)
        st.caption("Note: Random Walk is STD * .6")

        st.write("")
        st.write("")

        plot_time_series(df_sorted,0,9)
        st.caption("Note: Random Walk is STD * .6")

############################################################ END Score over time

############################################################ BEGIN Score Distrubution/ Course Breakdown

    with tab2:
        st.write("")
        full_rounds_df = df_sorted.filter(pl.col("18 Holes Played Flg") ==1)

        ## Adding In Mean and Median to chart
        mean_score = full_rounds_df["Total Strokes"].mean()
        median_score = full_rounds_df["Total Strokes"].median()
        std_score = full_rounds_df['Total Strokes'].std()
        n = len(full_rounds_df)

        # High level metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_score, scale=std_score/np.sqrt(n))
            st.metric("95% Confidence Interval for Average Score", f"{ci_low:.1f} – {ci_high:.1f}")

        with col2:
            st.write("")



        with col3:
            pdf = full_rounds_df.to_pandas()
            pdf, model = fit_linear_regression(pdf)

            intercept = model.params[0]
            slope = model.params[1]
            equation = f"y = {intercept:.1f} + {slope:.1f}x"
            st.metric("Equation of Linear Regression Line:", equation )



        st.write("")


        chart_kde = (
            alt.Chart(full_rounds_df.to_pandas())
            .transform_density(
                "Total Strokes",
                as_=["Total Strokes", "Likelihood"],
            )
            .mark_area(opacity=0.5)
            .encode(
                x="Total Strokes:Q",
                y="Likelihood:Q"
            )
            .properties(height=500)
        )



        rule_mean = alt.Chart(pd.DataFrame({"x": [mean_score]})).mark_rule(color="red").encode(x="x:Q")
        rule_median = alt.Chart(pd.DataFrame({"x": [median_score]})).mark_rule(color="blue").encode(x="x:Q")


        st.altair_chart((chart_kde + rule_mean + rule_median), use_container_width=True)
        st.caption("Blue: Median")
        st.caption("Red: Mean")



        st.write("")
        full_rounds_df = full_rounds_df.with_columns(
            ((pl.col("Total Strokes") - mean_score) / std_score).alias("z")
        )

        st.header("Outlier Rounds (2 Standard Deviations)")
        outliers = full_rounds_df.filter(pl.col("z").abs() > 2)
        st.write(outliers)


    ##Course Breakdowns
        st.write("")
        st.write("")
        st.header("Course Breakdown") 

        st.write("Distinct Courses Played: ", len(full_rounds_df['Course'].value_counts()))

        st.write("Most Played Courses:")
        top_courses = (
            full_rounds_df
                .select(pl.col("Course").value_counts())  
                .unnest("Course")                       
                .sort("counts", descending=True)            
                .head(5)
        )

        st.table(
            top_courses
        )

        course_stats_wide = (
            full_rounds_df
                .groupby("Course")
                .agg(
                    pl.col("Total Strokes").mean().alias("avg_score"),
                    pl.count().alias("rounds_played")
                )
        )

        course_stats_long = course_stats_wide.melt(
            id_vars=["Course"],     # keep these columns
            value_vars=["avg_score"], # columns to unpivot
            variable_name="Metric",
            value_name="Score"
        )
        course_stats_long = course_stats_long.sort("Score")

        st.write("Average Score by Course:")
        st.table(course_stats_long)


############################################################ END Score Distrubution / Course Breakdown

############################################################ BEGIN Score per Hole

##TODO: Breakdown of Birdie, par, boegey, etc

    # THis is going to exclude just Par 3 Courses (Already dropped)
    with tab3:
        scoring_df = create_score_df(df_sorted)

        # Convert Polars DF → Pandas (Altair expects pandas)
        scoring_pd = scoring_df.to_pandas()

        # Compute percentages (optional but nice for tooltips)
        scoring_pd["Pct"] = scoring_pd["Count"] / scoring_pd["Count"].sum()

        # Base pie chart
        base = alt.Chart(scoring_pd).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Score Type:N"),
            tooltip=[
                alt.Tooltip("Score Type:N"),
                alt.Tooltip("Count:Q"),
                alt.Tooltip("Pct:Q", format=".1%")
            ]
        )

        # Donut = arc with innerRadius
        donut = base.mark_arc(innerRadius=50).properties(
            title = "Score Type Distrubution (Includes 9 and 18 Holes)"
        )

        st.altair_chart(donut, use_container_width=True)

        st.write("")
        st.write(df_sorted)

        # Strokes over time
        
        long_df = (
            df_sorted
                .select([
                    pl.col("Round Played"),
                    pl.col("Birdies"),
                    pl.col("Pars")
                ])
                .melt(
                    id_vars="Round Played",
                    value_vars=["Birdies", "Pars"],
                    variable_name="Score Type",
                    value_name="Count"
                )
                .to_pandas()
        )

        # Altair line chart
        chart = (
            alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Round Played:T", title="Round Played"),
                    y=alt.Y("Count:Q", title="Number of Scoring Events"),
                    color="Score Type:N",
                    tooltip=["Round Played", "Score Type", "Count"]
                )
                .properties(
                    title="Birdies & Pars Over Time",
                    height=400
                )
        )

        st.altair_chart(chart, use_container_width=True)


############################################################ END Score per Hole




