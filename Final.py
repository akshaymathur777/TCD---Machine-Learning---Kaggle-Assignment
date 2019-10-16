# Importing the libraries
import numpy as np
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
submit = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
dataset = dataset.dropna()
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values
X = X.dropna()
submit = submit.fillna(0)

# Replacing 0's and unknown values of gender by unknown
X.Gender.replace(['0', 'unknown'], ['unknown' , 'unknown'], inplace=True)
submit.Gender.replace(['0', 'unknown'], ['unknown' , 'unknown'], inplace=True)

# Replacing values of  0's by No in the university column
X['University Degree'].replace(['0', 0], ['No', 'No'], inplace = True)
submit['University Degree'].replace(['0', 0], ['No', 'No'], inplace = True)

# Dropping the instance column only as it gives best result
X = X.drop(['Instance'] , axis='columns')
submit = submit.drop(['Instance'] , axis='columns')

# Train test split of 80 : 20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Using Categorical Boost Regresor
from catboost import CatBoostRegressor
model=CatBoostRegressor(task_type = 'GPU', iterations = 100000, learning_rate = 0.005)
model.fit(X_train,y_train,cat_features=([1, 3, 5, 6, 8]),eval_set=(X_test, y_test))
model.score(X_test,y_test)

# Getting the predicted values
ans = model.predict(submit)
