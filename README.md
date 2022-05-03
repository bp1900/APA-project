# Machine Learning Project 
The goal is to develop a classification model to predict the patients surival after a heart failure.
If the data is not available, the main program will automatically download and create a folder for it.

## Content
* `main.py`: Runs, in order: `exploration.py`, `modeling.py`, and `results.py`.
* `report.pdf`: A self-contained document of our experiment.
* `requirements.txt`: Configuaration file used in [Getting Started](#Getting-Started)
* `exploration.py`: The program that preprocesses the dataset and serializes it. It also generates the data exploration plots (correlation, features vs target, etc.)
* `modeling.py`: The program that reads the previously serialized data and implements the models. **Warning:** the hyperparameter search is deactivated, it directly fits the best models found. If you want to do the full search, which takes a few days to execute, delete (or comment) the second dictionary called `params`.
* `results.py`: The program that reads the serialized results of the models and prints some of the results.
* `Readme.md`: This file.
Once deployed:
* `data/heart.csv`: A heart failure survival prediction dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records#). **If the folder is not available, it will be created and the data downloaded automatically when executing the main program.**
* `data/preprocess.pickle`: the serialization of the data split into train, test and their corresponding preprocessing.
* `results/fitted_models.pickle`: serialization of the models fitted with the best hyperparameters.
* `resluts/metrics.tex`: a .tex file containing the metrics in latex format.
* `figures/`: a folder containing all the images, some of which ended on the report. Others did not.


### About data
This dataset consists of 13 clinical features:
* **age:** age of the patient (years)
* **anaemia:** decrease of red blood cells or hemoglobin (boolean)
* **high blood pressure:** if the patient has hypertension (boolean)
* **creatinine phosphokinase (CPK):** level of the CPK enzyme in the blood (mcg/L)
* **diabetes:** if the patient has diabetes (boolean)
* **ejection fraction:** percentage of blood leaving the heart at each contraction (percentage)
* **platelets:** platelets in the blood (kiloplatelets/mL)
* **sex:** woman or man (binary)
* **serum creatinine:** level of serum creatinine in the blood (mg/dL)
* **serum sodium:** level of serum sodium in the blood (mEq/L)
* **smoking:** if the patient smokes or not (boolean)
* **time:** follow-up period (days)
* [target] **death event:** if the patient deceased during the follow-up period (boolean) 

## Getting Started
To run this project follow these steps:
* Install the requirements: `pip3 install -r requirements.txt`
* Deploy the project: `python3 main.py`

There is no need for the data to be available, it will be downloaded automatically.

## Authors
* **Benjamí Parellada**

## License
This project is licensed under the MIT License.
