"""
To prepare the starter code, copy this file over to decision_tree_starter.py
and go through and handle all the inline TODOs.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
from pydot import graph_from_dot_data
import io

from sklearn.model_selection import RandomizedSearchCV

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.labels = None

    @staticmethod
    def entropy(y):
        if len(y) == 0:
            return 0
        H = 0
        classes = np.unique(y)
        for C in classes:
            class_examples = y[y == C]
            p_C = len(class_examples) / len(y)
            H += -(p_C * np.log2(p_C + eps))
        return H
        
    @staticmethod
    def information_gain(X, y, feature_idx, thresh):
        if len(y) == 0:
            return 0
        
        left_mask = X[:, feature_idx] < thresh
        y_left = y[left_mask]
        y_right = y[~left_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        parent_entropy = DecisionTree.entropy(y)
        after_entropy = (len(y_left)*DecisionTree.entropy(y_left) + len(y_right)*DecisionTree.entropy(y_right)) / (len(y_left) + len(y_right))
        return parent_entropy - after_entropy

    @staticmethod
    def gini_impurity(y):
        if len(y) == 0:
            return 0
        
        G = 0
        classes = np.unique(y)
        for C in classes:
            class_examples = y[y == C]
            p_C = len(class_examples) / len(y)
            G += p_C ** 2
        return 1 - G

    @staticmethod
    def gini_purification(X, y, feature_idx, thresh):
        if len(y) == 0:
            return 0
        
        left_mask = X[:, feature_idx] < thresh
        y_left = y[left_mask]
        y_right = y[~left_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        parent_gini = DecisionTree.gini_impurity(y)
        after_gini = (len(y_left)*DecisionTree.gini_impurity(y_left) + len(y_right)*DecisionTree.gini_impurity(y_right)) / (len(y_left) + len(y_right))
        return parent_gini - after_gini

    def split(self, X, y, feature_idx, thresh):
        """
        Split the dataset into two subsets, given a feature and a threshold.
        Return X_0, y_0, X_1, y_1
        where (X_0, y_0) are the subset of examples whose feature_idx-th feature
        is less than thresh, and (X_1, y_1) are the other examples.
        """
        left_mask = X[:, feature_idx] < thresh
        right_mask = ~left_mask            
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
  
    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_thresh = None
                         
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            if len(thresholds) <= 1:
                continue
            
            for thresh in thresholds:
                #gain = self.information_gain(X, y, feature_idx, thresh)
                gain = self.gini_purification(X, y, feature_idx, thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_thresh = thresh
        return best_feature, best_thresh
                
    def fit(self, X, y, depth=0):
        self.labels = y
        # Leaf
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            self.pred = np.argmax(np.bincount(y))
            return
        
        self.split_idx, self.thresh = self.best_split(X, y)
        
        if self.split_idx is None:
            self.pred = np.argmax(np.bincount(y))
            return
        
        X_0, y_0, X_1, y_1 = self.split(X, y, self.split_idx, self.thresh)
        
        if len(y_0) == 0 or len(y_1) == 0:
            self.pred = np.argmax(np.bincount(y))
            return
                         
        self.left = DecisionTree(self.max_depth, self.features)
        self.right = DecisionTree(self.max_depth, self.features)
        
        self.left.fit(X_0, y_0, depth+1)
        self.right.fit(X_1, y_1, depth+1)

    def predict(self, X):
        if self.pred is not None: # leaf
            return np.array([self.pred] * X.shape[0])
                         
        left_mask = X[:, self.split_idx] < self.thresh
        right_mask = ~left_mask
        
        y_pred = np.zeros(X.shape[0], dtype='int')
        y_pred[left_mask] = self.left.predict(X[left_mask])
        y_pred[right_mask] = self.right.predict(X[right_mask])
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self, deep=True):
        return {'max_depth': self.max_depth, 'feature_labels': self.features}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def _to_graphviz(self, node_id):
        if self.pred is not None:  # Leaf node
            return f'{node_id} [label="Prediction: {self.pred}\nSamples: {len(self.labels) if self.labels is not None else 0}"];\n'
        else:
            if self.split_idx is None:
                return f'{node_id} [label="No split found"];\n'
            graph = f'{node_id} [label="{self.features[self.split_idx]} < {self.thresh:.2f}"];\n'
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            if self.left is not None:
                graph += f'{node_id} -> {left_id};\n'
                graph += self.left._to_graphviz(left_id)
            if self.right is not None:
                graph += f'{node_id} -> {right_id};\n'
                graph += self.right._to_graphviz(right_id)
            return graph

    def to_graphviz(self):
        graph = "digraph Tree {\nnode [shape=box];\n"
        graph += self._to_graphviz(0)
        graph += "}\n"
        return graph
        
    def __repr__(self):
        if self.pred is not None:  # Check if leaf node
            return "%s (%s)" % (self.pred, len(self.labels) if self.labels is not None else 0)
        else:
            if self.split_idx is None:  # Handle case where no split was found
                return "Leaf(%s)" % (self.pred if self.pred is not None else "?")
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                          self.thresh, 
                                          self.left.__repr__() if self.left else "?",
                                          self.right.__repr__() if self.right else "?")


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        for tree in self.decision_trees:
            idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[idx], y[idx])

    def predict(self, X):
        votes = np.array([tree.predict(X) for tree in self.decision_trees])
        return stats.mode(votes, axis=0)[0][0]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    

class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)
        
    def get_params(self, deep=True):
        return {'params': self.params, 'n': self.n, 'm': self.m}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


class BoostedRandomForest(RandomForest): 
    def fit(self, X, y):
        # OPTIONAL
        pass
                
    def predict(self, X):
        # OPTIONAL
        pass

def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) + (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode
    return data, onehot_features


def evaluate(clf):
    print("Cross validation", np.mean(cross_val_score(clf, X, y)))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions, dataset="titanic"):
    assert dataset in ["titanic", "spam"], f"dataset should be either 'titanic' or 'spam'"
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'predictions_{dataset}.csv', index_label='Id')


if __name__ == "__main__":
    dataset = "titanic"
    #dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    # Decision Tree
    print("\n\nDecision Tree")
    dt = DecisionTree(max_depth=5, feature_labels=features)
    dt.fit(X, y)
    
    dt_train_acc = dt.score(X, y)
    cv_acc = np.mean(cross_val_score(dt, X, y, cv=5))
    print(f"Training Accuracy: {dt_train_acc:.4f}, Validation Accuracy: {cv_acc:.4f}")

    # Visualize Decision Tree
    print("\n\nTree Structure")
    # Print using repr
    print(dt.__repr__())
    # Save tree to pdf
    graph_from_dot_data(dt.to_graphviz())[0].write_pdf("%s-basic-tree.pdf" % dataset)

    # Random Forest
    print("\n\nRandom Forest")
    rf = RandomForest(params, n=N, m=np.int_(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)
    
    rf_train_acc = rf.score(X, y)
    print(f"Random Forest Training Accuracy: {rf_train_acc:.4f}")
    
    param_dist = {
        'max_depth': [3, 5, 7, 10, 15, 20, 25, None],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'm': ['sqrt', 'log2', 0.3, 0.5, 0.7],  # max_features
        'n': [50, 100, 200, 300]
    }
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=246810)
    random_search.fit(X, y)
    print("Best Parameters:", random_search.best_params_)
    print("Best Validation score:", random_search.best_score_)
    
    best_rf = random_search.best_estimator_
    best_rf.fit(X, y)
    
    train_acc = best_rf.score(X, y)
    print(f"Randomized Search Random Forest Training Accuracy: {train_acc:.4f}")
    
    # Generate Test Predictions
    print("\n\nGenerate Test Predictions")
    pred = best_rf.predict(Z)
    generate_submission(Z, pred, dataset)