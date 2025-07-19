import json
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def load_configuration(path):
    """Reads configuration from a JSON file."""
    with open(path) as file:
        return json.load(file)

def get_digit_data():
    """Loads and returns the digits dataset."""
    digits = load_digits()
    return digits.data, digits.target

def initialize_and_train(X, y, cfg):
    """Initializes and fits a model according to the config."""
    model = LogisticRegression(
        C=cfg['C'],
        solver=cfg['solver'],
        max_iter=cfg['max_iter'],
        random_state=cfg.get('random_state', 42),
        multi_class='ovr'
    )
    model.fit(X, y)
    return model

def serialize_model(model, path):
    """Saves the trained model to disk."""
    joblib.dump(model, path)

def deserialize_model(path):
    """Loads a model from disk."""
    return joblib.load(path)

def compute_metrics(model, X, y):
    """Computes accuracy and weighted F1 for the model."""
    preds = model.predict(X)
    return accuracy_score(y, preds), f1_score(y, preds, average='weighted')