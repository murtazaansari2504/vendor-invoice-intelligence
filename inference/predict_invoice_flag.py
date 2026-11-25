from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

FEATURE_ORDER = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_artifact(path: str | Path):
    with open(path, "rb") as file:
        return joblib.load(file)


def predict_invoice_flag(
    input_data: dict,
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
) -> pd.DataFrame:
    model = load_artifact(model_path)
    scaler = load_artifact(scaler_path)

    expected_cols = list(getattr(scaler, "feature_names_in_", FEATURE_ORDER))
    input_df = pd.DataFrame(input_data)

    missing_cols = [col for col in expected_cols if col not in input_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required feature columns: {missing_cols}")

    ordered_df = input_df[expected_cols]
    scaled_features = scaler.transform(ordered_df)
    predictions = model.predict(scaled_features)

    output_df = input_df.copy()
    output_df["flag_invoice"] = predictions
    output_df["is_flagged"] = output_df["flag_invoice"].astype(bool)
    return output_df


if __name__ == "__main__":
    sample_input = {
        "invoice_quantity": [150, 25],
        "invoice_dollars": [1200.0, 400.0],
        "Freight": [45.0, 12.0],
        "total_item_quantity": [150, 20],
        "total_item_dollars": [1200.0, 250.0],
    }
    result = predict_invoice_flag(sample_input)
    print(result)