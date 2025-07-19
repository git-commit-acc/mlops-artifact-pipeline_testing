import json
import pickle
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def main():
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
    with open('model_train.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Accuracy: {:.4f}".format(model.score(X, y)))

if __name__ == "__main__":
    main()