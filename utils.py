import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

##Function to make Correlation matrix
def create_corr(df):
    #Make the matrix
    matrix = df.corr()
    # Create a heatmap with Seaborn
    sns.heatmap(
        matrix, 
        annot=True,          # Display correlation coefficients
        fmt=".2f",           # Format for the numbers
        cmap="coolwarm",     # Color map for the heatmap
        vmin=-1, vmax=1,     # Set the range of the color scale
        linewidths=0.5,      # Add lines between the cells
        cbar=True            # Display the color bar
    )

    # Add titles and labels
    plt.title('Corr Matrix', fontsize=16)
    plt.xticks(fontsize=10, rotation=45)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=10)

    # Show the plot
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

##Function to display metrics for regression model
def display_metrics(slope,intercept,y,y_pred):
    # Calculate the goodness of fit metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    # Display the metrics
    print(f"Equation of the line: y = {slope[0]:.4f}x + {intercept[0]:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")













##BELOW IS FOR golf_sim.py

class Golfer:
    def __init__(self,player_name,player_handicap):
        self.player_name = player_name
        self.player_handicap = player_handicap

def design_golf_course(holes):

    # Generate a list of random integers between 85 and 536 (inclusive)
    distances_l = [random.randint(85, 536) for i in range(holes)]

    ## With distances create values for par for the hole
    par_l = []
    for i in distances_l:
        if i < 221:
            par_l.append(3)
        elif i >= 221 and i < 467:
            par_l.append(4)
        else:
            par_l.append(5)

    return distances_l, par_l


def calculate_shot_distance(handicap):

    # Define variability multipliers based on handicap
    if handicap == "low":  # Low handicap
        variability_range =[0.9, 1.2]  # Less variability
    elif handicap == "mid":  # Mid handicap
        variability_range =[0.75, 1.1]  # Moderate variability
    else:
        variability_range = [0.6, 1.0]  # More variability

    return variability_range


def play_round(env, shot_range, yardage, par, sim_data):
    total_strokes = 0
    total_par = 0

    ## Iteration over a hole
    for i, (hole_yardage, hole_par) in enumerate(zip(yardage, par), start=1):
        strokes = 0
        total_par = total_par + hole_par
        distance_remaining = hole_yardage
        print(f"Time {env.now:.2f}: Starting hole {i} (Yardage: {hole_yardage}, Par: {hole_par})")

        while distance_remaining > 0:
            strokes = strokes +1
            shot_time = random.uniform(1,3)
            yield env.timeout(shot_time)

            # Simulate shot distance. Here we assume the "average" shot might be roughly
            avg_shot = hole_yardage / hole_par
            shot_distance = random.uniform(shot_range[0] * avg_shot, shot_range[1] * avg_shot)
            distance_remaining = max(0, distance_remaining - shot_distance)
            print(f"Time {env.now:.2f}: Hole {i}, Stroke {strokes}: shot {shot_distance:.2f} yards, remaining {distance_remaining:.2f} yards")

        total_strokes += strokes
        print(f"Time {env.now:.2f}: Finished hole {i} in {strokes} strokes\n")
        sim_data.append((env.now, hole_yardage, hole_par, strokes))

    print(f"Round completed at time {env.now:.2f} with total strokes: {total_strokes} with par: {total_par}")
        