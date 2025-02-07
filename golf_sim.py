import simpy
import random
import pandas as pd 
from utils import design_golf_course
from utils import play_round
from utils import calculate_shot_distance

handicap = "low" ##acceptable values: high, mid, low

##Creating database to store the scores
column_names_database = ['hole_timestamp', 'hole_yardage', 'hole_par', 'player_score', 'hole_number']
column_names_to_append = ['hole_timestamp', 'hole_yardage', 'hole_par', 'player_score']
sim_database = pd.DataFrame(columns=column_names_database)
sim_data = []

## function for number of holes... to make random yardages...
## Then assign values for par
yardage, par = design_golf_course(18)

##Distnace of shots based on handicap
shot_range = calculate_shot_distance(handicap)
print('Yardage: ',yardage)
print('Par: ',par)



##Simulate data and create ML Classifer...? 
env = simpy.Environment()
env.process(play_round(env, shot_range, yardage, par, sim_data)) ## Function is to run the simulation 
env.run()

sim_to_append = pd.DataFrame(sim_data, columns=column_names_to_append)
sim_to_append['hole_number'] = sim_to_append.index + 1


# Append df_to_append to the original DataFrame df
sim_database = pd.concat([sim_database, sim_to_append]) #ignore_index=True)


print(sim_database)


