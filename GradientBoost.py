from typing import Any
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
class GBoost:
    def __init__(self, boosts, learning_rate):
        self.boosts = boosts
        self.learning_rate = learning_rate
    def fit(self, X, y, monitor):
        if len(X) != len(y):
            raise Exception("Please ensure X and y are the same length")
        def find_best_split(X, y):
            X = np.array(X)  # Ensure X is a NumPy array
            n_samples, n_features = X.shape
            best_mse = float('inf')
            best_feature_index,best_threshold,best_left_value,best_right_value = None,None,None,None
            for feature_index in range(n_features):
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    left_mask = X[:, feature_index] <= threshold
                    right_mask = X[:, feature_index] > threshold
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    left_value = y[left_mask].mean()
                    right_value = y[right_mask].mean()
                    mse = (
                        np.mean((y[left_mask] - left_value) ** 2) * np.sum(left_mask) +
                        np.mean((y[right_mask] - right_value) ** 2) * np.sum(right_mask)
                    ) / n_samples
                    if mse < best_mse:
                        best_mse = mse
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_left_value = left_value
                        best_right_value = right_value
            return best_feature_index, best_threshold, best_left_value, best_right_value, best_mse
        def initialize_predictions(y):
            return np.mean(y)
        def compute_residuals(y, y_pred):
            return y - y_pred
        def train_weak_learner(X, residuals):
            feature_index, threshold, left_value, right_value, best_mse = find_best_split(X, residuals)
            return feature_index, threshold, left_value, right_value, best_mse
        def predict_with_stump(X, feature_index, threshold, left_value, right_value):
            left_count, right_count = 0, 0
            for value in [i[feature_index] for i in X]:
                if value <= threshold:
                    left_count+=1
                else:
                    right_count+=1
            return np.concatenate((np.full((left_count), left_value) , np.full((right_count), right_value)))
        def update_predictions(y_pred, feature_index, threshold, left_value, right_value, X, learning_rate):
            new_prediction = y_pred + learning_rate * predict_with_stump(X, feature_index, threshold, left_value, right_value)
            return new_prediction
        self.initial_pred = initialize_predictions(y)
        current_pred = np.full(len(y),self.initial_pred)
        error = []
        stumps = []
        for t in range(self.boosts):
            residuals = compute_residuals(y,current_pred)# get residuals from the current predictions
            stump = train_weak_learner(X, residuals)# train the next weak learner on the features and residuals
            feature_index, threshold, left_value, right_value, best_mse = stump 
            current_pred = update_predictions(current_pred, feature_index, threshold, left_value, right_value, X, self.learning_rate)# update the predictions to incorporate the latest weak learner
            stumps.append(stump)
            error.append(best_mse)
        self.classifier = stumps
    def predict(self, X):
        def predict_with_stump(X, feature_index, threshold, left_value, right_value):
            left_count, right_count = 0, 0
            for value in [i[feature_index] for i in X]:
                if value <= threshold:
                    left_count+=1
                else:
                    right_count+=1
            return np.concatenate((np.full((left_count), left_value) , np.full((right_count), right_value)))
        def update_predictions(y_pred, feature_index, threshold, left_value, right_value, X, learning_rate):
            new_prediction = y_pred + learning_rate * predict_with_stump(X, feature_index, threshold, left_value, right_value)
            return new_prediction

        current_pred = np.full(len(X), self.initial_pred)
        for classifier in self.classifier:
            current_pred = update_predictions(current_pred, classifier[0], classifier[1], classifier[2], classifier[3], X, self.learning_rate)
        y_pred = current_pred
        return y_pred
    def get_classifier(self):
        return self.classifier

class GBoostMod(GBoost):
    def __init__(self, boosts, learning_rate, weak_learner_type):
        GBoost.__init__(self,boosts,learning_rate)
        self.weak_learner_type = weak_learner_type
    def fit(self,X,y,monitor):
        if len(X) != len(y):
            raise Exception("Please ensure X and y are the same length")
        def initialize_predictions(y):
            return np.mean(y)
        def compute_residuals(y, y_pred):
            return y - y_pred
        def train_weak_learner(X, residuals, weak_learner_type):
            match weak_learner_type:
                case "nearest_neighbours_1":
                    model = KNeighborsRegressor(n_neighbors=1)
                case "nearest_neighbours_2":
                    model = KNeighborsRegressor(n_neighbors=2)
                case "nearest_neighbours_3":
                    model = KNeighborsRegressor(n_neighbors=3)
                case "nearest_neighbours_4":
                    model = KNeighborsRegressor(n_neighbors=4)
                case "tree_depth_2":
                    model = DecisionTreeRegressor(max_depth=2,max_features=1)
                case "tree_depth_3":
                    model = DecisionTreeRegressor(max_depth=3,max_features=1)
                case "tree_depth_4":
                    model = DecisionTreeRegressor(max_depth=4,max_features=1)
                case "neural_network":
                    model = MLPRegressor(max_iter=100)
                case _:
                    print(f"Did not recognise {weak_learner_type}. Using Decision Stump")
                    model = DecisionTreeRegressor(max_depth=1,max_features=1)
            model.fit(X,residuals)
            return model
        def update_predictions(y_pred, learning_rate, model, X):
            predictions = model.predict(X)
            new_prediction = y_pred + learning_rate * predictions
            mse = mean_squared_error(residuals, predictions)
            return new_prediction, mse
        
        self.initial_pred = initialize_predictions(y)
        current_pred = np.full(len(y),self.initial_pred)
        error = []
        weak_learners = []
        for t in range(self.boosts):
            residuals = compute_residuals(y,current_pred)# get residuals from the current predictions
            model = train_weak_learner(X, residuals, self.weak_learner_type)# train the next weak learner on the features and residuals 
            current_pred, best_mse = update_predictions(current_pred, self.learning_rate, model, X)# update the predictions to incorporate the latest weak learner
            weak_learners.append(model)
            error.append(best_mse)
        self.classifier = weak_learners
    def predict(self, X):
        def update_predictions(y_pred, learning_rate, model, X):
            predictions = model.predict(X)
            new_prediction = y_pred + learning_rate * predictions
            return new_prediction
        current_pred = np.full(len(X), self.initial_pred)
        for model in self.classifier:
            current_pred = update_predictions(current_pred, self.learning_rate, model, X)
        y_pred = current_pred
        return y_pred