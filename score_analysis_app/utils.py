import polars as pl
import json
import pandas as pd
from pandas import json_normalize

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



