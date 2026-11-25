import os
import joblib
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    db_path = r"C:\Users\DELL\Desktop\projects\invoice generation\data\inventory.db"
    model_dir = r"C:\Users\DELL\Desktop\projects\models"
    os.makedirs(model_dir, exist_ok=True)

    # Load and prepare data
    df = load_vendor_invoice_data(db_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train candidate models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = [
        (evaluate_model(lr_model, X_test, y_test, "Linear Regression"), lr_model),
        (evaluate_model(dt_model, X_test, y_test, "Decision Tree"), dt_model),
        (evaluate_model(rf_model, X_test, y_test, "Random Forest"), rf_model)
    ]

    # Select best model based on lowest RMSE
    best_result, best_model = min(results, key=lambda x: x[0]["rmse"])

    model_path = os.path.join(model_dir, "predict_freight_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_result['model_name']}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()