import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from utils import deserialize_model, get_digit_data, compute_metrics

def main():
    model_file = 'model_trained.pkl'
    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found.")
        exit(1)

    model = deserialize_model(model_file)
    X, y = get_digit_data()
    print(f"Running inference on {X.shape[0]} examples.")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    accuracy, f1 = compute_metrics(model, X, y)
    print(f"Inference Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    print("\nSample predictions:")
    for idx in range(min(10, len(predictions))):
        pred = predictions[idx]
        truth = y[idx]
        conf = np.max(probabilities[idx])
        print(f"Sample {idx + 1}: Predicted={pred} | True={truth} | Confidence={conf:.3f}")

    print("\nPrediction Distribution:")
    classes, counts = np.unique(predictions, return_counts=True)
    for c, count in zip(classes, counts):
        print(f"Label {c}: {count} times")
    print("Inference finished.")

if __name__ == '__main__':
    main()
