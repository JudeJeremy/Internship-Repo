# MOdelling
# 1. Classification - predict categories
# 2. Regression - predict numbers
# 3. Clustering - group similar categories
#supervised modelling
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
clfd = DecisionTreeClassifier()
rfc = RandomForestClassifier()
iris = load_iris()
X = iris.data #features
y = iris.target #target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model on the training data
clfd.fit(X_train, y_train)
# Make predictions on the test data
y_pred = clfd.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

clfd.score(X_train, y_train) # Training Accuracy
clfd.score(X_test, y_test) # Testing Accuracy

# Training Acc > Testing Acc - Model overfitting
# Training Acc < Testing Acc - Model underfitted


rfc.fit(X_train, y_train)
y_pred2 = rfc.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Accuracy: {accuracy2*100:.2f}%")

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy3 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy3*100:.2f}%")

import numpy as np
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#unsupervised modelling
from sklearn.cluster import KMeans
import numpy as np
#we dont need a y value since this is unsupervised and we do not know what is the target
X = np.array([[1, 2], [1, 4], [1, 0],
               [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])

# Code source: Jaques Grobler
# License: BSD 3 clause

#supervised modelling using linear regression
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficient and Y intercept: \n", regr.coef_, regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()