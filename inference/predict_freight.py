from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "predict_freight_model.pkl"


def load_model(model_path: str | Path = MODEL_PATH):
    with open(model_path, "rb") as file:
        return joblib.load(file)


def predict_freight_cost(
    input_data: dict, model_path: str | Path = MODEL_PATH
) -> pd.DataFrame:
    model = load_model(model_path)
    input_df = pd.DataFrame(input_data)

    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        rename_map = {
            col: target
            for target in expected_cols
            for col in input_df.columns
            if col.lower() == target.lower() and col != target
        }
        if rename_map:
            input_df = input_df.rename(columns=rename_map)
        input_df = input_df[list(expected_cols)].copy()

    predictions = model.predict(input_df)
    input_df["predicted_freight"] = predictions.round(2)
    return input_df


if __name__ == "__main__":
    sample_input = {
        "Dollars": [3000, 200]
    }
    result = predict_freight_cost(sample_input)
    print(result)