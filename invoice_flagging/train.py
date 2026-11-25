import os
import joblib

from data_preprocessing import (
    apply_labels,
    load_invoice_data,
    scale_features,
    split_data,
)
from modeling_evaluation import (
    evaluate_classifier,
    train_random_forest_classifier,
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]
TARGET = "flag_invoice"


def main():
    db_path = os.path.join("data", "inventory.db")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    df = load_invoice_data(db_path)
    df = apply_labels(df)

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=FEATURES,
        target=TARGET,
        test_size=0.2,
        random_state=42,
    )

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train=X_train,
        X_test=X_test,
        scaler_path=scaler_path,
    )

    grid_search = train_random_forest_classifier(X_train_scaled, y_train)
    evaluate_classifier(
        model=grid_search,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name="Random Forest Classifier",
    )

    best_model = grid_search.best_estimator_
    model_path = os.path.join(models_dir, "predict_flag_invoice.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    main()