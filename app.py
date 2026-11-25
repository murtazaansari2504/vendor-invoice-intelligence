import numpy as np
import pandas as pd
import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    
    layout="wide",
)

st.title(" Vendor Invoice Intelligence Portal")
st.markdown(
    "Predict expected **freight costs** and flag **high-risk vendor invoices** "
    "for manual audit and approval."
)
st.divider()

st.sidebar.title("Model Selection")
selected_module = st.sidebar.radio(
    "Choose Prediction Module",
    ["Freight Cost Prediction", "Invoice Flag Prediction"],
)
st.sidebar.markdown(
    "**Business Impact:** Helps finance teams forecast landed costs, negotiate "
    "with vendors, and prioritize high-risk invoices for manual review."
)

if selected_module == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")
    st.markdown(
        "**Objective:** Predict expected freight cost for a vendor invoice to "
        "support budgeting, forecasting, and vendor negotiation."
    )

    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=1.0,
                value=1500.0,
                step=50.0,
            )
        with col2:
            quantity = st.number_input(
                "Order Quantity (Optional / Reference)",
                min_value=1,
                value=100,
                step=1,
            )

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        # Match the feature(s) your saved freight regression model expects
        input_data = {"dollars": [dollars]}
        result_df = predict_freight_cost(input_data)
        pred_value = result_df["predicted_freight"].iloc[0]

        st.success(f"Estimated Freight Cost: **${pred_value:,.2f}**")
        st.metric(
            label="Predicted Freight Cost",
            value=f"${pred_value:,.2f}",
            delta=f"For ${dollars:,.2f} invoice (Qty ref: {quantity})",
        )

elif selected_module == "Invoice Flag Prediction":
    st.subheader("Invoice Risk Flag Prediction")
    st.markdown(
        "**Objective:** Flag vendor invoices with abnormal quantities, amounts, "
        "or freight patterns that require manual approval."
    )

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            total_item_quantity = st.number_input(
                "Total Item Quantity", min_value=0, value=150, step=1
            )
            invoice_quantity = st.number_input(
                "Invoice Quantity", min_value=0, value=150, step=1
            )
        with col2:
            total_item_dollars = st.number_input(
                "Total Item Dollars ($)", min_value=0.0, value=1200.0, step=10.0
            )
            invoice_dollars = st.number_input(
                "Invoice Dollars ($)", min_value=0.0, value=1200.0, step=10.0
            )
        with col3:
            freight = st.number_input(
                "Freight Cost ($)", min_value=0.0, value=45.0, step=1.0
            )

        submit_flag = st.form_submit_button("Predict Invoice Flag")

    if submit_flag:
        input_data = {
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars],
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
        }
        result_df = predict_invoice_flag(input_data)
        is_flagged = bool(result_df["is_flagged"].iloc[0])

        if is_flagged:
            st.warning(" Invoice requires **Manual Approval**.")
        else:
            st.success(" Invoice is **Safe for Auto Approval**.")