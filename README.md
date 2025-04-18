⛳ 18 Birdies Golf Score Analysis & ML Prediction
A personal data science project to analyze golf scores, explore simulation techniques, and serve a machine learning classifier using FastAPI.




📥 Getting Started
Clone the repository:

git clone https://github.com/mfannin099/18_birdies_project.git
cd 18_birdies_project
Install required packages:
pip install -r requirements.txt

Important Files:

read_json_pandas.py  ...... Script to clean/pull data from 18birdies json file

eda_9_holes.ipynb    ...... Notebook to analysis 9 hole scores

golf_sim.py          ...... Python script to create simulated data for classifier. Outputs: sim_golf_scores_for_ml.csv

sim_classifier.ipynb ...... Notebook that contains code for training and outputting DecisionTreeClassifier. Outputs: sim_golf_model.pkl

main.py              ...... Python script for FastAPI where model is deployed. user inputs hole_yardage, hole_par, palyer_score for a golf course. Outputs Handicap prediction

Learnings/Practice:
Practice JSON parsing
Bayes Statistics & PYMC
Classifier Model
Creating Class
FastAPI & deploying ML model


Main Goals & Objectives:
To analyze my golf score & fit regression line + look for outliers
Train and serve a classifier using FastAPI


Improvements: 
Create script thats soul purpose is to import json from 18birdies & analyze

