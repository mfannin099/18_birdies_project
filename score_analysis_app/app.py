import streamlit as st
import pandas as pd
import polars as pl
import altair as alt
import numpy as np


from utils import clean_data, plot_time_series

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
    df_sorted = df.sort("Round Played", descending=True)
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

    #TODO Make Tabs
    tab1,tab2 = st.tabs(['Score Over Time', 'Score Distubution'])

############################################################ BEGIN Score over time

    with tab1:
        plot_time_series(df_sorted,1,18)
        st.caption("Note: Random Walk is STD * .6")

        st.write("")
        st.write("")

        plot_time_series(df_sorted,0,9)
        st.caption("Note: Random Walk is STD * .6")

############################################################ END Score over time

############################################################ BEGIN Score Distrubution

    with tab2:
        st.write("")



############################################################ END Score Distrubution



    

    ##TODO: Distrubution of scores

    ##TODO: Breakdown of Birdie, par, boegey, etc

    ##TODO: Course Breakdowns 


    ## Maybe for somethings double all 9 hole scores..?
