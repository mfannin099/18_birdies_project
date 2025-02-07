import simpy
import random
from utils import design_golf_course
from utils import play_round
from utils import calculate_shot_distance

sim_data = []
handicap = "high" ##acceptable values: high, mid, low

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

print(sim_data)






