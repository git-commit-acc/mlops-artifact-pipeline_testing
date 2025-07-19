import sys, os
sys.path.append(os.path.dirname(__file__))

from utilities import load_configuration, get_digit_data, initialize_and_train, serialize_model, compute_metrics

def main():
    config = load_configuration('../config/config.json')
    X, y = get_digit_data()
    print(f"Loaded {X.shape[0]} records, {X.shape[1]} features each")

    print("Fitting model...")
    model = initialize_and_train(X, y, config)

    accuracy, weighted_f1 = compute_metrics(model, X, y)
    print(f"Training complete. Accuracy: {accuracy:.3f} | F1 (weighted): {weighted_f1:.3f}")

    serialize_model(model, 'model_trained.pkl')
    print("Model saved as model_trained.pkl")

if __name__ == '__main__':
    main()