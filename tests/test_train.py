import os, sys, numpy as np, pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils import (
    load_configuration, get_digit_data, initialize_and_train,
    serialize_model, deserialize_model, compute_metrics
)
from sklearn.linear_model import LogisticRegression

def test_config_parsing():
    cfg = load_configuration('config/config.json')
    assert all(param in cfg for param in ['C', 'solver', 'max_iter'])
    assert cfg['C'] > 0 and cfg['max_iter'] > 0
    assert isinstance(cfg['solver'], str)

def test_data_loading():
    X, y = get_digit_data()
    assert X.shape[0] > 0 and X.shape[1] == 64
    assert len(y) == X.shape[0]
    assert set(np.unique(y)) == set(range(10))

def test_model_training_and_params():
    X, y = get_digit_data()
    cfg = load_configuration('config/config.json')
    model = initialize_and_train(X, y, cfg)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_') and hasattr(model, 'classes_')
    assert model.C == cfg['C']
    assert model.max_iter == cfg['max_iter']

def test_metrics():
    X, y = get_digit_data()
    cfg = load_configuration('config/config.json')
    model = initialize_and_train(X, y, cfg)
    accuracy, f1 = compute_metrics(model, X, y)
    assert accuracy > 0.85 and f1 > 0.85

def test_model_serialization_cycle(tmp_path):
    X, y = get_digit_data()
    cfg = load_configuration('config/config.json')
    model = initialize_and_train(X, y, cfg)
    target = tmp_path / "test_model.pkl"
    serialize_model(model, target)
    loaded = deserialize_model(target)
    np.testing.assert_array_equal(model.predict(X[:5]), loaded.predict(X[:5]))
