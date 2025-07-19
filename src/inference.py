import pickle
from sklearn import datasets

def main():
    with open('model_train.pkl', 'rb') as f:
        model = pickle.load(f)
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    preds = model.predict(X)
    acc = model.score(X, y)
    print("Sample predictions:", preds[:10])
    print("Inference accuracy: {:.4f}".format(acc))

if __name__ == "__main__":
    main()
