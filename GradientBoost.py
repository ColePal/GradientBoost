from typing import Any
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class GBoost:
    def __init__(self, boosts, learning_rate=0.1):
        self.boosts = boosts
        self.learning_rate = learning_rate
    def fit(self, X, y, monitor=False, validation_split_percentage=0.1, seed=1, limit=20):
        if len(X) != len(y):
            raise Exception("Please ensure X and y are the same length")
        if validation_split_percentage > 0:
            X, validation_X, y, validation_y = train_test_split(X,y,test_size = validation_split_percentage, random_state = seed)
        if validation_split_percentage == 0 and limit > 0:
            raise Exception("Please ensure validation split is greater than 0 when assigning limit")
        def find_best_split(X, y):
            X = np.array(X)  # Ensure X is a NumPy array
            n_samples, n_features = X.shape
            best_mse = float('inf')
            best_feature_index,best_threshold,best_left_value,best_right_value = None,None,None,None
            for feature_index in range(n_features):
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    left_mask = X[:, feature_index] <= threshold
                    right_mask = ~left_mask
                    left_count = np.sum(left_mask)
                    right_count = n_samples - left_count

                    if left_count == 0 or right_count == 0:
                        continue
                    left_value = np.mean(y[left_mask])
                    right_value = np.mean(y[right_mask])
                    mse = (
                        np.mean((y[left_mask] - left_value) ** 2) * left_count +
                        np.mean((y[right_mask] - right_value) ** 2) * right_count
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
            return find_best_split(X, residuals)
        def predict_with_stump(X, feature_index, threshold, left_value, right_value):
            return np.where(X[:, feature_index] <= threshold, left_value, right_value)

        def update_predictions(y_pred, feature_index, threshold, left_value, right_value, X, learning_rate):
            predictions = predict_with_stump(X, feature_index, threshold, left_value, right_value)
            new_prediction = y_pred + learning_rate * predictions
            return new_prediction

        self.initial_pred = initialize_predictions(y)
        current_pred = np.full(len(y),self.initial_pred)
        error = []
        stumps = []
        best_mse = float('inf')
        tracker = 0
        best_classifier = None
        for t in range(self.boosts):
            if validation_split_percentage > 0:
                if tracker >= limit:
                    self.classifier=best_classifier
                    print("Early stopping")
                    return
            residuals = compute_residuals(y,current_pred)# get residuals from the current predictions
            stump = train_weak_learner(X, residuals)# train the next weak learner on the features and residuals
            current_pred = update_predictions(current_pred, stump[0], stump[1], stump[2], stump[3], X, self.learning_rate)# update the predictions to incorporate the latest weak learner
            stumps.append(stump)
            error.append(stump[4])
            self.classifier = stumps
            if validation_split_percentage > 0:
                current_mse = mean_squared_error(validation_y, self.predict(validation_X))
                tracker += 1
                if current_mse < best_mse:
                    tracker = 0
                    best_mse = current_mse
                    best_classifier = stumps.copy()
                    #print(f"New target: {best_mse}")
        if validation_split_percentage > 0:
            print(f"Validation MSE score: {mean_squared_error(validation_y, self.predict(validation_X))}")
    def predict(self, X):
        def predict_with_stump(X, feature_index, threshold, left_value, right_value):
            weak_prediction = np.array([left_value if i[feature_index] <= threshold else right_value for i in X])
            return weak_prediction
        
        def update_predictions(y_pred, feature_index, threshold, left_value, right_value, X, learning_rate):
            predictions = predict_with_stump(X, feature_index, threshold, left_value, right_value)
            new_prediction = y_pred + learning_rate * predictions
            return new_prediction

        current_pred = np.full(len(X), self.initial_pred)
        for classifier in self.classifier:
            current_pred = update_predictions(current_pred, classifier[0], classifier[1], classifier[2], classifier[3], X, self.learning_rate)
        y_pred = current_pred
        return y_pred

    def get_classifier(self):
        return self.classifier

class GBoostMod(GBoost):

    def __init__(self, boosts, learning_rate, weak_learner_type="tree_depth_1"):
        GBoost.__init__(self,boosts,learning_rate)
        self.weak_learner_type = weak_learner_type
    def fit(self,X,y,monitor=False, validation_split_percentage=0.1, seed=1, limit=10):
        if len(X) != len(y):
            raise Exception("Please ensure X and y are the same length")
        if validation_split_percentage > 0:
            X, validation_X, y, validation_y = train_test_split(X,y,test_size = validation_split_percentage, random_state = seed)
        def initialize_predictions(y):
            return np.mean(y)
        def compute_residuals(y, y_pred):
            return y - y_pred

        def train_weak_learner(X, residuals):
            recognised = True
            match self.weak_learner_type:
                case "nearest_neighbours_1":
                    model = KNeighborsRegressor(n_neighbors=1)
                case "nearest_neighbours_2":
                    model = KNeighborsRegressor(n_neighbors=2)
                case "nearest_neighbours_3":
                    model = KNeighborsRegressor(n_neighbors=3)
                case "nearest_neighbours_4":
                    model = KNeighborsRegressor(n_neighbors=4)

                case "tree_depth_1":
                    model = DecisionTreeRegressor(max_depth=1,max_features=1)
                case "tree_depth_2":
                    model = DecisionTreeRegressor(max_depth=2,max_features=1)
                case "tree_depth_3":
                    model = DecisionTreeRegressor(max_depth=3,max_features=1)
                case "tree_depth_4":
                    model = DecisionTreeRegressor(max_depth=4,max_features=1)
                case "neural_network":
                    model = MLPRegressor(max_iter=100)
                case _:

                    recognised = False
                    model = DecisionTreeRegressor(max_depth=1,max_features=1)
            model.fit(X,residuals)
            return model, recognised
        def update_predictions(y_pred, model, X):
            predictions = model.predict(X)
            new_prediction = y_pred + self.learning_rate * predictions
            mse = mean_squared_error(residuals, predictions)
            return new_prediction, mse
        
        self.initial_pred = initialize_predictions(y)
        current_pred = np.full(len(y),self.initial_pred)
        error = []
        weak_learners = []
        best_validation_mse = float('inf')
        tracker = 0
        for t in range(self.boosts):
            if tracker >= limit:
                self.classifier=best_classifier
                print("Early stopping")
                return
            residuals = compute_residuals(y,current_pred)# get residuals from the current predictions

            model, recognised = train_weak_learner(X, residuals)# train the next weak learner on the features and residuals 
            current_pred, best_mse = update_predictions(current_pred, model, X)# update the predictions to incorporate the latest weak learner
            weak_learners.append(model)
            error.append(best_mse)
            self.classifier = weak_learners
            current_mse = mean_squared_error(validation_y, self.predict(validation_X))
            tracker += 1
            if current_mse < best_validation_mse:
                tracker = 0
                best_validation_mse = current_mse
                best_classifier = weak_learners.copy()
                #print(f"New target: {best_mse}")
        if not recognised:
            print(f"Did not recognise {self.weak_learner_type}. Using Decision Stump")
        if validation_split_percentage > 0:
            print(f"Validation MSE score: {mean_squared_error(validation_y, self.predict(validation_X))}")
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