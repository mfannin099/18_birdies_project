import simpy
import random
import pandas as pd 
# from utils import design_golf_course ##Function
from utils import play_round ## Function
from utils import calculate_shot_distance ## Function
from utils import Golfer
from utils import Course

golfer1 = Golfer("Bad_player", "high") ##acceptable values: high, mid, low
##Distnace of shots based on handicap
shot_range = calculate_shot_distance(golfer1.player_handicap)
course1 = Course("Augusta National", 18)
#course1.display_course_info()

##Creating database to store the scores
column_names_database = ['player_name','player_handicap', 'course_name', 'hole_timestamp', 'hole_yardage', 'hole_par', 'player_score', 'hole_number']
column_names_to_append = ['hole_timestamp', 'hole_yardage', 'hole_par', 'player_score']
sim_database = pd.DataFrame(columns=column_names_database)
sim_data = []

##Simulate data and create ML Classifer...? 
env = simpy.Environment()
env.process(play_round(env, shot_range, course1 ,sim_data)) ## Function is to run the simulation 
env.run()

##Adding in columns that are expected in database
sim_to_append = pd.DataFrame(sim_data, columns=column_names_to_append)
sim_to_append['hole_number'] = sim_to_append.index + 1
sim_to_append['player_name'] = golfer1.player_name
sim_to_append['player_handicap'] = golfer1.player_handicap  
sim_to_append['course_name'] = course1.course_name  


# Append df_to_append to the original DataFrame df
sim_database = pd.concat([sim_database, sim_to_append]) #ignore_index=True)

print(sim_database.shape)
print(sim_database)
