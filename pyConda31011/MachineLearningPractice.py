# Steps of Machine Learning
# Data Science Principles
# 1. Decide the objective - Predict Fraud in Credit Card Transactions (Visa, MasterCard,...) - Visa
# 2. Data Collection - Collect Transactions Data for that Visa credit card
# 3. Understand the Data - Explore the Data which is Visualize it
# What is an outlier?
# Maths Score - out of 100 students, 99 students got in the range of 40 - 80, 1 student got 99.
# Skip the outliers
# Impute missing values
# 4. Transform the data - Raw Data - Machine readable data - Data Preprocessing, Data Transformation, normalization, standardization

# 5. Build the Model
# 6. Evaluate the model - metrics available, example accuracy of the model
# 7. Model Serving
# 8. Model Monitoring/Maintenance - constantly feed new data to update the model, Model retraining

# Evaluate the model
# Transactions Data from Jan 2020 - Dec 2023 - 10 million Transactions
# Test the model to evaluate
#  Give some data and find how it is predicting
# 10 million data - split the data to 80% and 20%
# Use 80% data to build the model
# Use the 20% to test the model - evaluate the accuracy of the model
# 20% is 200K records, out of this 200k records, if model can predict 180k correctly, 20k incorrectly
# Accurcay = 180k/200k = 90%
# Error = 20k/200k = 10%
# ANother model, 190k correctly, 10k incorrect, accuracy = 190k/200k = 95%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. objective: predict what the 16th student will get in math science and english
# 2. Data Collection:
testscores = {
    "math_scores": [78, 88, 22, 94, 87, 76, 85, 64, 89, 76, 83, 85, 84, 84, 83],
    "science_scores": [65, 74, 83, 94, 78, 56, None, 64, 73, 83, 57, 96, 74, 75, 76],
    "english_scores": [76, 57, 78, 96, 87, 56, 85, 55, 66, 98, 86, 78, 67, 87, 82]
}

dataframe = pd.DataFrame(testscores)


# 3. Visualize the data:
ax = dataframe.plot.bar(rot=0)
ax = dataframe.plot.line(subplots=True)

# Skip the outliers
# 22 is an outlier in the math_scores
dataframe["math_scores"].describe()
# mean = 78.533333
# replace 22 with mean. 
dataframe["math_scores"]=dataframe["math_scores"].replace(22, 79)

# There is a missing value in the science_scores. 
dataframe.isnull().sum()
dataframe["science_scores"].describe()
# mean = 74.857143
# replace missing value with mean
dataframe["science_scores"]=dataframe["science_scores"].fillna(74)
dataframe["science_scores"].dtype
dataframe=dataframe.astype(int)



# 4. Transform the data - Raw Data - Machine readable data - Data Preprocessing, Data Transformation

dataframe=dataframe.astype(int)

# 5. Build the Model





# Using another set of more complex data now

# 2. Data Collection
data = pd.read_csv("InsuranceSampleDataSet.csv")
data.head()
data.describe()

# 3. Visualize the data

ay = data.plot.bar(rot=0)
ay = data.plot.line(rot=0)
ay = data.plot.scatter(x="age", y="charges")
ay = data.plot.scatter(x="age", y="charges")

colours = {'smoker':'tab:blue'}
ay = plt.scatter(x="age", y="charges", c=colours)
plt.show()

# 4. Transform the data - Raw Data - Machine readable data - Data Preprocessing, Data Transformation

data.shape
data.dtypes

#convert objects into categories

# Evaluating model
# Classification models/Supervised modelling
# 1. Confusion matrix: Accuracy
# Regression Modelling
# Mean squared error, Root mean squared error, Mean absolute error

# Cross Validation Technique

# Model overfitting
# Model underfitting