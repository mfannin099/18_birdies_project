import simpy
import random
import pandas as pd 
from utils import design_golf_course ##Function
from utils import play_round ## Function
from utils import calculate_shot_distance ## Function
from utils import Golfer
from utils import Course

golfer1 = Golfer("Bad_player", "high") ##acceptable values: high, mid, low

##Creating database to store the scores
column_names_database = ['player_name','player_handicap','hole_timestamp', 'hole_yardage', 'hole_par', 'player_score', 'hole_number']
column_names_to_append = ['hole_timestamp', 'hole_yardage', 'hole_par', 'player_score']
sim_database = pd.DataFrame(columns=column_names_database)
sim_data = []

## function for number of holes... to make random yardages...
## Then assign values for par
yardage, par = design_golf_course(18)

# ##Distnace of shots based on handicap
# shot_range = calculate_shot_distance(golfer1.player_handicap)
# print('Yardage: ',yardage)
# print('Par: ',par)



# ##Simulate data and create ML Classifer...? 
# env = simpy.Environment()
# env.process(play_round(env, shot_range, yardage, par, sim_data)) ## Function is to run the simulation 
# env.run()

# ##Adding in columns that are expected in database
# sim_to_append = pd.DataFrame(sim_data, columns=column_names_to_append)
# sim_to_append['hole_number'] = sim_to_append.index + 1
# sim_to_append['player_name'] = golfer1.player_name
# sim_to_append['player_handicap'] = golfer1.player_handicap  


# # Append df_to_append to the original DataFrame df
# sim_database = pd.concat([sim_database, sim_to_append]) #ignore_index=True)


# print(sim_database)

class Course:
    def __init__(self, name, num_holes):
        """Initialize a golf course with a name and number of holes."""
        self.name = name
        self.num_holes = num_holes
        self.distances, self.pars = self.design_golf_course()  # Automatically generate course design

    def design_golf_course(self):
        """Generate distances and par values for each hole."""
        distances_l = [random.randint(85, 536) for _ in range(self.num_holes)]

        par_l = []
        for distance in distances_l:
            if distance < 221:
                par_l.append(3)
            elif 221 <= distance < 467:
                par_l.append(4)
            else:
                par_l.append(5)

        return distances_l, par_l

    def display_course_info(self):
        """Prints course details including name, holes, distances, and par values."""
        print(f"Course Name: {self.name}")
        print(f"Number of Holes: {self.num_holes}")
        for i in range(self.num_holes):
            print(f"Hole {i+1}: Distance {self.distances[i]} yards, Par {self.pars[i]}")

# Example Usage
course1 = Course("Augusta National", 9)
course1.display_course_info()
