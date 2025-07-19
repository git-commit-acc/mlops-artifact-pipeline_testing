import json
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

def test_config_loads():
    with open('config/config.json') as f:
        config = json.load(f)
    assert isinstance(config['C'], float)
    assert isinstance(config['solver'], str)
    assert isinstance(config['max_iter'], int)
    
def test_model_creation_and_fitting():
    with open('config/config.json') as f:
        config = json.load(f)
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter']
    )
    model.fit(X, y)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")

def test_accuracy_threshold():
    with open('config/config.json') as f:
        config = json.load(f)
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter']
    )
    model.fit(X, y)
    acc = model.score(X, y)
    assert acc > 0.8
