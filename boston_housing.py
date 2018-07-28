# coding: utf-8

# Import libraries necessary for this project
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import supplementary visualizations code visuals.py
import visuals as vs


def show_price_on_feature_variation():
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(features.columns):
        plt.subplot(1, 3, i+1)
        plt.plot(data[col], prices, 'x')
        plt.title('%s x MEDV' % col)
        plt.xlabel(col)
        plt.ylabel('MEDV')
    plt.show()


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)

    regressor = DecisionTreeRegressor()
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_


# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)


# Calculate Statistics
minimum_price = min(prices)
maximum_price = max(prices)
mean_price = prices.mean()
median_price = prices.median()
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price))
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${:.2f}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${:.2f}\n".format(std_price))

# code to show that my theory on question 1 is correct
show_price_on_feature_variation()

# Split the data into 80% training and 20% testing.
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)

# Produce a graph for a decision tree model that has been trained and validated on the training data
# using different maximum depths.
vs.ModelComplexity(X_train, y_train)

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Print the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.\n".format(reg.get_params()['max_depth']))

# Matrix for client data
client_data = [[5, 17, 15],  # Client 1
               [4, 32, 22],  # Client 2
               [8, 3, 12]]   # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

# Run the `fit_model` function ten times with different training and testing sets,
# to see how the prediction for a specific client changes with respect to the data it's trained on.
vs.PredictTrials(features, prices, fit_model, client_data)
