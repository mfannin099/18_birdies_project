look at 18 holes (double 9nhole) data ---> done 

For Bayes regression model, try with 100 rounds, 1000 rounds... make random data ---> done

Hook up to Github ---> done

binning ---> done 

SimPy
    Make virtual env and use for SimPy
    Create some sort of simulation for golf scores...?.... Or maybe simulate a round of 18 holes? ----> done

Create Classes for Course and player.... also make code ready for a loop ----> done

Run Simulation 200 times, to create dataset that is 100k rows. save to csv, create classifier for the type of golfer ----> done

Classifier for for simulated data ---> done, not a good model

Do something with Rest API or Fast API
    (There is a free book on O'Riley for Fast API Stuff) ----> WIP

    
Pull data from API....? to practice doing that (NOT IN SCOPE?)




### Virtual Env stuff 
Code to create virtual env (Run this line):
python -m venv name_of_env

Code for starting a virtual enviroment (Run this line):
simPy_env\Scripts\activate
(name_of_env\Scripts\activate)

To stop: 
deactivate


## Github stuff:

Link is for making feature branch: https://gist.github.com/vlandham/3b2b79c40bc7353ae95a

Merge new branch into main: git checkout main, git branch, git merge feature_branch_name, git push origin main


## To run Fast API
uvicorn filename:app --reload
