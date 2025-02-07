import simpy
import random
from utils import design_golf_course
from utils import play_round

scores = []

## function for number of holes... to make random yardages...
## Then assign values for par
yardage, par = design_golf_course(2)
print('Yardage: ',yardage)
print('Par: ',par)


env = simpy.Environment()
env.process(play_round(env, yardage, par)) ## Function is to run the simulation 
env.run()






