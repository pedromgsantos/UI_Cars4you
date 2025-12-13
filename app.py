import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, ".")

from preprocessing_utils import CURRENT_YEAR, generate_user_final_df, process_dict

# Page configuration
st.set_page_config(
    page_title="Cars 4 You - Price Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# File paths (relative to repository root)
MODEL_PATH = "files/random_forest_noCV.pkl"
SCALER_PATH = "preprocessing_results/full_dataset/scaler.pkl"

# Input validation ranges
VALIDATION_RANGES = {
    "year": (1990, 2020),
    "mileage": (0, 500000),
    "tax": (0, 1000),
    "mpg": (0.0, 150.0),
    "engineSize": (0.5, 7.0),
    "previousOwners": (0, 20),
    "hasDamage": (0, 1),
}

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 100%);
        color: #e5e7eb;
    }

    h1, h2, h3 {
        color: #e5e7eb;
        font-weight: 600;
    }

    h1 {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .stNumberInput input, .stSelectbox select, .stTextInput input {
        background: rgba(31, 41, 55, 0.8);
        border: 1px solid rgba(75, 85, 99, 0.5);
        border-radius: 8px;
        color: #e5e7eb;
        padding: 0.5rem;
    }

    .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }

    label {
        color: #9ca3af;
        font-weight: 500;
        font-size: 0.875rem;
    }

    .stRadio > div {
        background: rgba(31, 41, 55, 0.5);
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        width: 100%;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }

    .stAlert {
        background: rgba(31, 41, 55, 0.8);
        border-radius: 8px;
    }

    hr {
        border-color: rgba(75, 85, 99, 0.3);
    }
    </style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def validate_inputs(inputs: dict):
    y0, y1 = VALIDATION_RANGES["year"]
    if not (y0 <= inputs["year"] <= y1):
        return False, f"Year must be between {y0} and {y1}."

    m0, m1 = VALIDATION_RANGES["mileage"]
    if not (m0 <= inputs["mileage"] <= m1):
        return False, f"Mileage must be between {m0:,} and {m1:,}."

    t0, t1 = VALIDATION_RANGES["tax"]
    if not (t0 <= inputs["tax"] <= t1):
        return False, f"Annual tax must be between £{t0} and £{t1}."

    mpg0, mpg1 = VALIDATION_RANGES["mpg"]
    if not (mpg0 <= inputs["mpg"] <= mpg1):
        return False, f"MPG must be between {mpg0} and {mpg1}."

    e0, e1 = VALIDATION_RANGES["engineSize"]
    if not (e0 <= inputs["engineSize"] <= e1):
        return False, f"Engine size must be between {e0}L and {e1}L."

    po0, po1 = VALIDATION_RANGES["previousOwners"]
    if not (po0 <= inputs["previousOwners"] <= po1):
        return False, f"Previous owners must be between {po0} and {po1}."

    if inputs["hasDamage"] not in [0, 1]:
        return False, f"Has damage must be either 0 (No) or 1 (Yes)."

    return True, ""


def predict_price_from_dict(input_dict: dict, model) -> float:
    try:
        df_processed = generate_user_final_df(input_dict)
        pred_scaled = float(model.predict(df_processed)[0])
        
        # Unscale prediction
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        
        price_log_min = scaler.data_min_[10]
        price_log_max = scaler.data_max_[10]
        price_log_range = price_log_max - price_log_min
        price_log_unscaled = pred_scaled * price_log_range + price_log_min
        
        final_price = float(np.exp(price_log_unscaled))
        
        return final_price
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        raise

def main():
    st.title("Cars 4 You - Price Predictor")
    st.markdown("Advanced car price prediction with intelligent preprocessing.")
    st.markdown("---")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    col_input, col_result = st.columns([1.5, 1], gap="large")

    with col_input:
        st.subheader("Vehicle Details")

        st.markdown("#### Identification")
        
        col_brand, col_model = st.columns(2)
        with col_brand:
            brand = st.text_input(
                "Brand",
                value="",
                help="e.g., Mercedes, BMW, Audi",
                placeholder="Enter brand name"
            )
        
        with col_model:
            model_name = st.text_input(
                "Model",
                value="",
                help="e.g., A Class, 3 Series, Golf",
                placeholder="Enter model name"
            )

        year = st.number_input(
            "Year",
            min_value=VALIDATION_RANGES["year"][0],
            max_value=VALIDATION_RANGES["year"][1],
            value=2015,
            step=1,
        )

        st.markdown("#### Specifications")
        col_a, col_b = st.columns(2)

        with col_a:
            mileage = st.number_input(
                "Mileage (miles)",
                min_value=VALIDATION_RANGES["mileage"][0],
                max_value=VALIDATION_RANGES["mileage"][1],
                value=50000,
                step=1000,
            )
            mpg = st.number_input(
                "MPG",
                min_value=float(VALIDATION_RANGES["mpg"][0]),
                max_value=float(VALIDATION_RANGES["mpg"][1]),
                value=45.5,
                step=0.1,
            )
            engine_size = st.number_input(
                "Engine Size (L)",
                min_value=float(VALIDATION_RANGES["engineSize"][0]),
                max_value=float(VALIDATION_RANGES["engineSize"][1]),
                value=2.0,
                step=0.1,
            )

        with col_b:
            tax = st.number_input(
                "Annual Tax (£)",
                min_value=VALIDATION_RANGES["tax"][0],
                max_value=VALIDATION_RANGES["tax"][1],
                value=150,
                step=10,
            )
            transmission = st.text_input(
                "Transmission",
                value="",
                help="e.g., Manual, Automatic, Semi-Auto",
                placeholder="Enter transmission type"
            )
            fuel_type = st.text_input(
                "Fuel Type",
                value="",
                help="e.g., Petrol, Diesel, Hybrid",
                placeholder="Enter fuel type"
            )

        st.markdown("#### Condition")
        col_c, col_d = st.columns(2)

        with col_c:
            previous_owners = st.number_input(
                "Previous Owners",
                min_value=VALIDATION_RANGES["previousOwners"][0],
                max_value=VALIDATION_RANGES["previousOwners"][1],
                value=1,
                step=1,
            )

        with col_d:
            has_damage = st.radio(
                "Has Damage",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0,
            )

        st.markdown("")
        predict_button = st.button("Predict Price", use_container_width=True)

    with col_result:
        st.subheader("Result")

        if predict_button:
            if not brand.strip():
                st.error("Please enter a Brand.")
                st.stop()
            if not model_name.strip():
                st.error("Please enter a Model.")
                st.stop()
            if not transmission.strip():
                st.error("Please enter a Transmission type.")
                st.stop()
            if not fuel_type.strip():
                st.error("Please enter a Fuel Type.")
                st.stop()

            inputs = {
                "Brand": brand.strip(),
                "model": model_name.strip(),
                "year": float(year),
                "mileage": float(mileage),
                "tax": float(tax),
                "mpg": float(mpg),
                "engineSize": float(engine_size),
                "transmission": transmission.strip(),
                "fuelType": fuel_type.strip(),
                "paintQuality%": None,
                "previousOwners": float(previous_owners),
                "hasDamage": float(has_damage),
            }

            ok, msg = validate_inputs(inputs)
            if not ok:
                st.error(msg)
            else:
                try:
                    processed_inputs = process_dict(inputs.copy())
                    predicted_price = predict_price_from_dict(inputs, model)

                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(31, 41, 55, 0.6);
                            border: 1px solid rgba(75, 85, 99, 0.4);
                            border-radius: 16px;
                            padding: 1.75rem;
                            text-align: center;
                            margin-top: 1rem;
                        ">
                            <div style="
                                font-size: 0.9rem;
                                color: #9ca3af;
                                font-weight: 600;
                                letter-spacing: 0.06em;
                                margin-bottom: 0.75rem;
                            ">
                                ESTIMATED PRICE
                            </div>
                            <div style="
                                font-size: 2.7rem;
                                font-weight: 800;
                                line-height: 1.15;
                            ">
                                £{predicted_price:,.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("---")
                    st.markdown("**Vehicle Summary**")
                    car_age = CURRENT_YEAR - int(processed_inputs["year"])
                    st.markdown(
                        f"""
- **Brand:** {processed_inputs["Brand"]}
- **Model:** {processed_inputs["model"]}
- **Year:** {int(processed_inputs["year"])} ({car_age} years old)
- **Mileage:** {int(processed_inputs["mileage"]):,} miles
- **Engine Size:** {processed_inputs["engineSize"]} L
- **Transmission:** {processed_inputs["transmission"]}
- **Fuel Type:** {processed_inputs["fuelType"]}
- **MPG:** {processed_inputs["mpg"]}
- **Annual Tax:** £{int(processed_inputs["tax"])}
- **Previous Owners:** {int(processed_inputs["previousOwners"])}
- **Has Damage:** {"Yes" if processed_inputs["hasDamage"] == 1 else "No"}
"""
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.exception(e)
        else:
            st.info("Fill in the vehicle details and click 'Predict Price' to get an estimate.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>"
        "NOVA IMS Master's Project 2025-26<br>"
        "Cars 4 You - Intelligent Price Prediction"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()