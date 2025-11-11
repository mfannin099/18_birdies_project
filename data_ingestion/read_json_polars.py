import json
import pandas as pd 
import numpy as np
from pandas import json_normalize
import polars as pl
file = "18Birdies_archive.json"

##Value used to determine if the # of holes is 9 or 18 for flag
holes_for_18_calc_val = 70

##Opening the json
with open(file, 'r') as file_:
    data = json.load(file_)

print("------------------------")
print("------------------------")

##Grabbing the rounds data
rounds_data = json_normalize(data['myData']['activityData']['rounds'], 
                             sep='_', 
                             record_prefix='round_')

rounds_data_pl = pl.DataFrame(rounds_data)

##Grabbing the clubs played at
club_data = json_normalize(data['myData']['clubData']['playedClubs'], 
                             sep='_', 
                             record_prefix='round_')
club_data_pl = pl.DataFrame(club_data)

##joining the data
final_df_pl = rounds_data_pl.join(club_data_pl, left_on='clubId_id', right_on='clubId', how='left')
print(final_df_pl.head(3))

##beginning to clean the columns 

##deleting not needed columns
cols_to_del = ['stats_fairwayLefts',
       'stats_fairwayMiddles', 'stats_fairwayRights', 'stats_fairwayShorts',
       'stats_fairwayLongs', 'stats_fairwayHoleCount', 'stats_gir',
       'stats_girLefts', 'stats_girRights', 'stats_girShorts',
       'stats_girLongs', 'stats_girNoChances', 'stats_girHoleCount',
       'stats_strokeGainOverall', 'stats_strokeGainTeeToGreen',
       'id', 'clubId_id']

final_df_pl = final_df_pl.drop(columns=cols_to_del)


##cleaning the timestamp 
final_df_pl = final_df_pl.with_columns(
    pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp")
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

##create new columns for each score for the holes
# for i in range(1, 19):
#     final_df_pl = final_df_pl.with_columns(
#         pl.col("holeStrokes").apply(
#             lambda x: x[i - 1] if i <= len(x) else None, return_dtype=pl.Int64
#         ).alias(f'hole_{i}')
#    )
print(final_df_pl.shape)
print(final_df_pl.columns)
print(final_df_pl.head())

#final_df.to_csv('golf_18birdies_data.csv', index=False)
