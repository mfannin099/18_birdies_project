import simpy
import random
import pandas as pd 
# from utils import design_golf_course ##Function
from utils import play_round ## Function
from utils import calculate_shot_distance ## Function
from utils import Golfer
from utils import Course

repeat_courses = 100

##Creating database to store the scores
column_names_database = ['player_name','player_handicap', 'course_name', 'hole_timestamp', 'hole_yardage', 'hole_par', 'player_score', 'hole_number']
column_names_to_append = ['hole_timestamp', 'hole_yardage', 'hole_par', 'player_score']
sim_database = pd.DataFrame(columns=column_names_database)
sim_data = []

##acceptable values: high, mid, low
golfers = [
    Golfer("Bad_player", "high"),
    Golfer("Average_player", "mid"),
    Golfer("Pro_player", "low")
]

courses = [
    Course("Local Course", 18),
    Course("Local Links", 18),
    Course("Cheap Course Farms", 18),
    Course("Expensive Fairways", 18),
    Course("Beach Town Golf Club", 18),
    Course("Pebble Hills", 18),
    Course("Oscar Golf Resort", 18),
]

final_courses = courses * repeat_courses

for player in golfers:
    ##Distnace of shots based on handicap
    shot_range = calculate_shot_distance(player.player_handicap)

    for c in final_courses:
        sim_data = []

        ##Simulate data and create ML Classifer...? 
        env = simpy.Environment()
        env.process(play_round(env, shot_range, c ,sim_data)) ## Function is to run the simulation 
        env.run()

        ##Adding in columns that are expected in database
        sim_to_append = pd.DataFrame(sim_data, columns=column_names_to_append)
        sim_to_append['hole_number'] = sim_to_append.index + 1
        sim_to_append['player_name'] = player.player_name
        sim_to_append['player_handicap'] = player.player_handicap  
        sim_to_append['course_name'] = c.course_name  


        # Append df_to_append to the original DataFrame df
        sim_database = pd.concat([sim_database, sim_to_append]) #ignore_index=True)

print(sim_database.shape)

print("------------------")
print("------------------")
print("------------------")
score_sums = sim_database.groupby(['player_name', 'course_name'])['player_score'].sum()
score_avg = score_sums / repeat_courses
print(score_avg)
