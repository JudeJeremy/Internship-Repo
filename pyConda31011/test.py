import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

testscores = {
    "math_scores": [78, 88, 92, 94, 87, 76],
    "science_scores": [65, 74, 83, 94, 78, 56],
    "names": ["james", "josh", None, "mark", "mark", "john"],
    "floats": [2.1, 4.2, 5.5, 4.3, 5.8, 8.4],
    "missingvalues": ["apple", "missing", "orange", "banana", "milk", "strawberry"]
}

dataframe = pd.DataFrame(testscores, index=["Student 1", "Student 2", "Student 3","Student 4","Student 5","Student 6"])
dataframe
dataframe.loc[[0,1]]
dataframe.iloc[1]
dataframe.iloc[-2:]
dataframe.iloc[:1,:1]
dataframe.iloc[:1,:1]
dataframe.shape
dataframe.columns
dataframe.dtypes
dataframe["names"].value_counts()
dataframe["names"].unique()
dataframe["math_scores"].describe()
dataframe.isnull().sum()
dataframe.head()
dataframe.tail()
dataframe["missingvalues"]=dataframe["missingvalues"].fillna("water")
dataframe["floats"]=dataframe["floats"].fillna(7.8)
dataframe=dataframe.fillna("Matthew")
#dataframe=dataframe[[0, 0]].fillna(54)
dataframe["math_scores"]=dataframe["math_scores"].fillna(54)
dataframe["names"]=dataframe["names"].str.replace("john", "johnny")
dataframe=dataframe.drop(["floats"], axis=0)


fig, ax = plt.subplots()
fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')
plt.show()


ax = dataframe.plot.line(rot=0, subplots=True)
ax = dataframe.plot.bar(rot=0, subplots=True)
ax = dataframe.plot.bar(rot=0)
ax = dataframe.plot.line(rot=0)

# Steps of Machine Learning
# Data Science Principles
# 1. Decide the objective - Predict Fraud in Credit Card Transactions (Visa, MasterCard,...) - Visa
# 2. Data Collection - Collect Transactions Data for that Visa credit card
# 3. Understand the Data - Explore the Data which is Visualize it
# What is an outlier?
# Maths Score - out of 100 students, 99 students got in the range of 40 - 80, 1 student got 99.
# Skip the outliers
# Impute missing values
# Convert to normalized data
# 4. Transform the data - Raw Data - Machine readable data - Data Preprocessing, Data Transformation
# 5. Build the Model
# 6. Evaluate the model - metrcis available, example accuracy of the model
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



