import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as file:
        return joblib.load(file)


def predict_freight_cost(input_data: dict, model_path: str = MODEL_PATH) -> pd.DataFrame:
    model = load_model(model_path)
    input_df = pd.DataFrame(input_data)
    predictions = model.predict(input_df)
    input_df["predicted_freight"] = predictions.round(2)
    return input_df


if __name__ == "__main__":
    sample_input = {
        "dollars": [3000, 200]
    }
    result = predict_freight_cost(sample_input)
    print(result)