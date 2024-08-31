from typing import Any
import numpy as np
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
