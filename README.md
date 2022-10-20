# Product Placement
Capstone project 

In this repository you will find the following directories:

* Notebooks
* Thesis
* Scripts
* EnvRequirements

## Notebooks 

Contains two jupyter notebook files, Prototype.ipynb and Shapley_Counterfactuals.ipynb. Inside the first one you will find code for data wrangling, preprocessing, feature engineering and modelling. Inside the second file you will find Dice counterfactuals and Shapley values. To be inline with the thesis project, the first step is in Prototype.ipynb whereas the second and third step of the implementation is in Shapley_Counterfactuals.ipynb.

## Thesis 

Contains the thesis' report in pdf format.

## Scripts 

Conctains six different files each one representing a discrete functionality in python scripts. The files are self explainatory and are listed below: 
* data_wrangling.py
* preprocessing.py
* modelling.py
* model_tuning.py
* shapley_values.py
* counterfactuals.py

Those scripts are transfered from the jupyter files and altered to run in terminal with prints in between tasks for easier readability while running. Make sure your directory has the data folder inside when running those files. The order of which you should run the scripts is as listed above. 

## EnvRequirements

Contains the requirements.txt for all the different environments being used. 

* In PrototypeEnv environment you can run 
data_wrangling.py
preprocessing.py
modelling.py
model_tuning.py
Prototype.ipynb

* In CounterfactualsEnv environment you can run
counterfactuals.py 

* In ShapEnv environment you can run 
shapley_values.py



