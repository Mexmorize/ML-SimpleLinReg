# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:32:20 2019

@author: Benjamin B
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into Training and Testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

# Fitting Simple Linear Regression to the Training set
"""From sklearn linear model library, we import the LinearRegression class"""
from sklearn.linear_model import LinearRegression
# Make an object of the LinearRegression class that was imported
regressor = LinearRegression()
# Call fit method from LinearRegression class on regressor object
regressor = regressor.fit(x_train, y_train)

# Predicting the Test set results
"""Create vector of predicted values(Salaries = y) to store our predictions
for the independent test vector. We can then compare y_pred(predictions) to
y_test(actual values) and see how well the regressor fit the simple linear
regression model"""
y_pred = regressor.predict(x_test)

# Visualizing the Training set results
"""Plot the real observation points (real values, x_train and y_train) and the 
simple linear regression line"""
plt.scatter(x_train, y_train, color = 'red')
# Plot regression line
"""x coordinate is x_train, y coordinate is the PREDICTIONS of x_train since 
we are trying to plot the regression line and so the y points would need to be
the predictions of the x value, x_train. Therefore we need the regressor of the
x_train data. THE REASON THIS IS NOT y_pred is because y_pred are the values of
the TEST set but we plotting the regression line for the TRAIN set."""
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
# Add a title to the plot
plt.title('Salary vs. Experience (Training set)')
# X-axis label
plt.xlabel("Years of experience")
# Y-axis label
plt.ylabel("Salary")
# Specify the end of the graph and plot
plt.show()

# Visualize Test set results
plt.scatter(x_test, y_test, color = "red")
"""Don't need to change x_train with x_test since plotting the regression line
on the training or test results should yield the same line equation, just cover
a different area. Therefore, don't need to change x_train to x_test"""
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title('Salary vs. Experience (Test set)')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()