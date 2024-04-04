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


#convert objects into categories

# Evaluating model
# Classification models/Supervised modelling
# 1. Confusion matrix: Accuracy
# Regression Modelling
# Mean squared error, Root mean squared error, Mean absolute error

# Cross Validation Technique

# Model overfitting
# Model underfitting


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

#supervised regression model using linear regression

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X = np.array([[1], [3], [4], [6], [7], [8], [10], [12], [16], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28]])
X = X.reshape(-1, 1)
Y = np.array([2, 4, 5, 7, 8, 9, 10, 11, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
Y.shape
Y = Y.reshape(-1, 1)

plt.scatter(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficient and Y intercept: \n", regr.coef_, regr.intercept_)
# The mean squared error should be close to 0
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))

# Plot outputs
plt.scatter(X_test, Y_test, color="black")
plt.plot(X_test, Y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#supervised classification using decisiontreeclassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn import svm

from sklearn.linear_model import SGDClassifier


#objective is to recognize what class each wine goes into

# 2. Data Collection
wine = load_wine()
X = wine.data
Y = wine.target

# 3. Understand the Data - Explore the Data which is Visualize it
#plt.scatter(wine)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()

clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf = svm.SVC()

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

clf.fit(X_train, Y_train)


Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

#unsupervised clustering using K Means

#unsupervised modelling
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AffinityPropagation

#we dont need a y value since this is unsupervised and we do not know what is the target
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
additional_data = np.random.randint(0, 11, size=(100, 2))
X_with_additional = np.concatenate((X, additional_data), axis=0)
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X_with_additional)
kmeans.labels_

clustering = AffinityPropagation(random_state=5).fit(X)
clustering.labels_

kmeans.predict([[0, 0], [30, 3], [15, 15], [22, 1], [76, 56], [10, 0], [73, 16], [2, 1]])
clustering.predict([[0, 0], [30, 3], [15, 15], [22, 1], [76, 56], [10, 0], [73, 16], [2, 1]])







