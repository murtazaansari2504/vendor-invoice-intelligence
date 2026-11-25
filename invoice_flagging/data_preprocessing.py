import os
import sqlite3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_invoice_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def create_invoice_risk_label(row) -> int:
    # Flag if invoice total mismatches item-level total by more than $5
    if pd.notna(row["invoice_dollars"]) and pd.notna(row["total_item_dollars"]):
        if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
            return 1

    # Flag if average receiving delay exceeds 10 days
    if pd.notna(row["avg_receiving_delay"]) and row["avg_receiving_delay"] > 10:
        return 1

    return 0


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def split_data(
    df: pd.DataFrame,
    features: list[str],
    target: str = "flag_invoice",
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df[features]
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, scaler