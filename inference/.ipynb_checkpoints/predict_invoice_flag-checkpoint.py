import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_artifact(path: str):
    with open(path, "rb") as file:
        return joblib.load(file)


def predict_invoice_flag(
    input_data: dict,
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
) -> pd.DataFrame:
    model = load_artifact(model_path)
    scaler = load_artifact(scaler_path)

    input_df = pd.DataFrame(input_data)
    scaled_features = scaler.transform(input_df)
    predictions = model.predict(scaled_features)

    input_df["flag_invoice"] = predictions
    input_df["is_flagged"] = input_df["flag_invoice"].astype(bool)
    return input_df


if __name__ == "__main__":
    sample_input = {
        "total_item_quantity": [150, 20],
        "total_item_dollars": [1200.0, 250.0],
        "invoice_quantity": [150, 25],
        "invoice_dollars": [1200.0, 400.0],
        "freight": [45.0, 12.0],
    }
    result = predict_invoice_flag(sample_input)
    print(result)