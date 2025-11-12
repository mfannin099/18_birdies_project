import streamlit as st
import pandas as pd
import polars as pl
from utils import clean_data

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
