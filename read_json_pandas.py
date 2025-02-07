import json
import pandas as pd 
import numpy as np
from pandas import json_normalize

##Value used to determine if the # of holes is 9 or 18 for flag
holes_for_18_calc_val = 70

##Opening the txt file
with open('18Birdies_archive.json.txt', 'r') as file:
    data = json.load(file)


print("------------------------")
print("------------------------")

##Grabbing the rounds data
rounds_data = json_normalize(data['myData']['activityData']['rounds'], 
                             sep='_', 
                             record_prefix='round_')

print("------------------------")
print("------------------------")

##Grabbing the clubs played at
club_data = json_normalize(data['myData']['clubData']['playedClubs'], 
                             sep='_', 
                             record_prefix='round_')


print(club_data.columns)
print("------------------------")
print("------------------------")

##joining the data
final_df = pd.merge(rounds_data, club_data, left_on='clubId_id', right_on='clubId', how='left')


##beginning to clean the columns 

##deleting not needed columns
cols_to_del = ['stats_fairwayLefts',
       'stats_fairwayMiddles', 'stats_fairwayRights', 'stats_fairwayShorts',
       'stats_fairwayLongs', 'stats_fairwayHoleCount', 'stats_gir',
       'stats_girLefts', 'stats_girRights', 'stats_girShorts',
       'stats_girLongs', 'stats_girNoChances', 'stats_girHoleCount',
       'stats_strokeGainOverall', 'stats_strokeGainTeeToGreen',
       'id', 'clubId_id', 'clubId']

final_df = final_df.drop(columns=cols_to_del)


##cleaning the timestamp 
final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='ms')

##calculating par for the course (whether its 9 or 18 holes)
final_df['course_par'] = final_df['strokes'] - final_df['score'].abs()

##calculating the number of holes played based on the score of the round 
final_df['is_18_holes'] = final_df['strokes'].apply(lambda x: 0 if x < holes_for_18_calc_val else 1)

##create new columns for each score for the holes
for i in range(1, 19):  # Loop through hole numbers 1 to 18
    final_df[f'hole_{i}'] = final_df['holeStrokes'].apply(lambda x: x[i-1] if i <= len(x) else np.nan)
print(final_df.shape)
print(final_df.columns)
print(final_df.head())

#final_df.to_csv('golf_18birdies_data.csv', index=False)


##Get Weather data 
##Get Golf course data 