"""
Simplified preprocessing utilities for Cars4You UI

2 IMPORTANT NOTES:
 - THIS IS A SIMPLIFIED VERSION OF THE ORIGINAL NOTEBOOK'S PREPROCESSING CODE, BECAUSE IN THE UI ALL FIELDS ARE MANDATORY AND THERE ARE NO MISSING VALUES TO IMPUTE.
 - THE MODEL WE ARE USING IS NOT THE BEST MODEL WE COULD TRAIN, BUT RATHER A SIMPLER AND LIGHTER(!!!) MODEL THAT COULD FIT IN THE ZIP FILE SIZE LIMIT ON THE SUBMISSION.


DESIGN DECISION - No Missing Value Imputation:
==============================================
This module does NOT include the following functions from the original notebook:
- guess_brand_model(): Fills missing Brand/model using engineSize/transmission/fuelType
- get_model(): Returns model mode based on Brand combinations
- get_brand_from_model(): Reverse lookup Brand from model
- fix_empty_categorical(): Fills missing categorical values using mode
- get_median_with_fallback(): Calculates medians with hierarchical fallbacks
- fix_empty_numerical(): Fills missing numerical values using grouped medians

REASON:
-------
In the UI, ALL fields are MANDATORY and provided by the user. There are no missing 
values to impute. The only exception is paintQuality%, which is set to a fixed 
default value (75.0) since it's filled by mechanics post-evaluation and is not 
visible in the user interface (and not used in the prediction whatsoever).

This simplification reduces code from 604 to 255 lines (-58%) while maintaining 
the exact same preprocessing logic for complete inputs. The removed functions were 
essential for training data preprocessing but are unnecessary for production 
prediction where users must fill all fields.

PREPROCESSING PIPELINE:
================================================================================

STEP 1: Fix Categorical Input
    - Standardize variations using mapping dictionaries
    - Example: "VW" → "Volkswagen", "Merc" → "Mercedes"
    - Mappings loaded from: ./mapping_dicts/*.csv

STEP 2: Handle Outliers
    - Cap extreme values to reasonable ranges
    - year ≤ 2020, mileage ≥ 0, 0 ≤ mpg ≤ 400, etc.
    - paintQuality% auto-filled to 75.0 if None

STEP 3: Correct Data Types
    - Ensure proper types: int (year, previousOwners), float (mileage, tax, etc.)

STEP 4: Feature Engineering
    - Create car_age = CURRENT_YEAR (2020) - year
    - Target encode model using median price_log
    - One-hot encode: transmission (Manual, Semi-Auto) and fuelType (Diesel, Hybrid)

STEP 5: Feature Selection
    - Select 10 features expected by trained model:
      [model_encoded, tax, car_age, mileage, mpg, engineSize,
       transmission_Manual, transmission_Semi-Auto, fuelType_Diesel, fuelType_Hybrid]
    - Note: paintQuality%, previousOwners, year are NOT sent to model

STEP 6: Scaling
    - Apply MinMaxScaler to 6 numeric features (indices 0-5)
    - One-hot encoded features (indices 6-9) remain binary (0/1)

================================================================================
USAGE EXAMPLE:
================================================================================

    from preprocessing_utils import generate_user_final_df, CURRENT_YEAR

    input_dict = {
        "Brand": "BMW",
        "model": "3 Series",
        "year": 2016.0,
        "transmission": "Automatic",
        "mileage": 45000.0,
        "fuelType": "Diesel",
        "tax": 200.0,
        "mpg": 55.4,
        "engineSize": 2.0,
        "paintQuality%": None,  # Auto-filled to 75.0
        "previousOwners": 2.0,
        "hasDamage": 0.0
    }
    
    # Returns scaled DataFrame ready for model prediction
    df_scaled = generate_user_final_df(input_dict)

================================================================================
"""
import numpy as np
import pandas as pd
import pickle

CURRENT_YEAR = 2020

# Load mapping dictionaries for standardizing input variations
brand_mapping = pd.read_csv('./mapping_dicts/brand_mapping.csv')
fuelType_mapping = pd.read_csv('./mapping_dicts/fueltype_mapping.csv')
model_mapping = pd.read_csv('./mapping_dicts/model_mapping.csv')
transmission_mapping = pd.read_csv('./mapping_dicts/transmission_mapping.csv')

brand_map = dict(zip(brand_mapping["Variation"].str.strip(), brand_mapping["AssignedValue"].str.strip()))
fuelType_map = dict(zip(fuelType_mapping["Variation"].str.strip(), fuelType_mapping["AssignedValue"].str.strip()))
model_map = dict(zip(model_mapping["Variation"].str.strip(), model_mapping["AssignedValue"].str.strip()))
transmission_map = dict(zip(transmission_mapping["Variation"].str.strip(), transmission_mapping["AssignedValue"].str.strip()))


# ============================================
# STEP 1: Fix categorical variations
# ============================================
def fix_categorical_input(car_dict):
    """Standardize categorical inputs using mapping dictionaries"""
    for key in ["Brand", "model", "fuelType", "transmission"]:
        value = car_dict.get(key)
        if pd.isna(value) or value is None:
            car_dict[key] = np.nan  
        else:
            car_dict[key] = str(value).strip() 

    if not pd.isna(car_dict.get("Brand")):
        car_dict["Brand"] = brand_map.get(car_dict["Brand"], car_dict["Brand"])
    
    if not pd.isna(car_dict.get("fuelType")):
        car_dict["fuelType"] = fuelType_map.get(car_dict["fuelType"], car_dict["fuelType"])
    
    if not pd.isna(car_dict.get("model")):
        car_dict["model"] = model_map.get(car_dict["model"], car_dict["model"])
    
    if not pd.isna(car_dict.get("transmission")):
        car_dict["transmission"] = transmission_map.get(car_dict["transmission"], car_dict["transmission"])
    
    return car_dict


# ============================================
# STEP 2: Handle outliers (cap extreme values)
# ============================================
def handle_outliers(car_dict):
    """Cap extreme values to reasonable ranges"""
    
    if car_dict.get("year") is not None and not pd.isna(car_dict.get("year")):
        if car_dict["year"] > 2020:
            car_dict["year"] = 2020

    if car_dict.get("mileage") is not None and not pd.isna(car_dict.get("mileage")):
        if car_dict["mileage"] < 0:
            car_dict["mileage"] = 0
        
    if car_dict.get("tax") is not None and not pd.isna(car_dict.get("tax")):
        if car_dict["tax"] < 0:
            car_dict["tax"] = 0

    if car_dict.get("mpg") is not None and not pd.isna(car_dict.get("mpg")):
        if car_dict["mpg"] < 0:
            car_dict["mpg"] = 0
        elif car_dict["mpg"] > 400:
            car_dict["mpg"] = 400

    if car_dict.get("engineSize") is not None and not pd.isna(car_dict.get("engineSize")):
        if car_dict["engineSize"] < 0:
            car_dict["engineSize"] = 0

    # paintQuality%: auto-fill if None (not visible in UI)
    if car_dict.get("paintQuality%") is None or pd.isna(car_dict.get("paintQuality%")):
        car_dict["paintQuality%"] = 75.0  # Default median, this is not used aniway...
    else:
        if car_dict["paintQuality%"] < 0:
            car_dict["paintQuality%"] = 0
        elif car_dict["paintQuality%"] > 100:
            car_dict["paintQuality%"] = 100
    
    if car_dict.get("previousOwners") is not None and not pd.isna(car_dict.get("previousOwners")):
        if car_dict["previousOwners"] < 0:
            car_dict["previousOwners"] = 0
    
    return car_dict


# ============================================
# STEP 3: Correct data types
# ============================================
def correct_types(car_dict):
    """Convert features to correct data types"""
    # Ints
    int_features = ['year', 'previousOwners']
    for feature in int_features:
        if car_dict.get(feature) is not None and not pd.isna(car_dict.get(feature)):
            car_dict[feature] = int(car_dict[feature])
    
    # Floats
    float_features = ['mileage', 'tax', 'mpg', 'engineSize', 'paintQuality%']
    for feature in float_features:
        if car_dict.get(feature) is not None and not pd.isna(car_dict.get(feature)):
            car_dict[feature] = float(car_dict[feature])
    
    # Strings 
    str_features = ['Brand', 'model', 'transmission', 'fuelType']
    for feature in str_features:
        value = car_dict.get(feature)
        if pd.isna(value) or value is None or value == 'nan' or value == 'None':
            car_dict[feature] = np.nan  
        else:
            car_dict[feature] = str(value).strip()
    
    return car_dict


# ============================================
# STEP 4: Encoding
# ============================================
def find_encoding(model):
    """Simple target encoding: returns median price_log for model"""
    with open("./preprocessing_results/full_dataset/encoding_maps.pkl", "rb") as f:
        data = pickle.load(f)
    
    if model in data["model_encoding_map"]:
        return data["model_encoding_map"][model]
    else:
        return data["overall_fallback"]


# ============================================
# STEP 5: Scaling
# ============================================
def scale_df(df, scaler):
    """Scale numeric features using pre-fitted scaler"""
    df_scaled = df.copy()

    # Feature mapping for OLD model scaler (6 numeric features)
    feature_mapping = {
        'model_encoded': 0,
        'tax': 1,
        'car_age': 2,
        'mileage': 3,
        'mpg': 4,
        'engineSize': 5
        # One-hot features (transmission, fuelType) are NOT scaled
    }
    
    for feature, scaler_idx in feature_mapping.items():
        if feature not in df_scaled.columns:
            continue
        
        data_min = scaler.data_min_[scaler_idx]
        data_max = scaler.data_max_[scaler_idx]
        data_range = data_max - data_min
        
        if data_range > 0:
            df_scaled[feature] = (df_scaled[feature] - data_min) / data_range
    
    return df_scaled


# ============================================
# MAIN PROCESSING PIPELINE
# ============================================
def process_dict(input_dict):
    """
    Main preprocessing pipeline for single input.
    All fields expected to be provided by user.
    """
    car = fix_categorical_input(input_dict)
    car = handle_outliers(car)
    car = correct_types(car)
    
    # Create car_age
    car["car_age"] = CURRENT_YEAR - car["year"]   
    
    # Encode model
    car["model_encoded"] = find_encoding(car["model"])

    # One-hot encode transmission
    car["transmission_Manual"] = 1 if car["transmission"].lower() == "manual" else 0
    car['transmission_Semi-Auto'] = 1 if car["transmission"].lower() == "semi-auto" else 0
    
    # One-hot encode fuelType
    car["fuelType_Diesel"] = 1 if car["fuelType"].lower() == "diesel" else 0
    car["fuelType_Hybrid"] = 1 if car["fuelType"].lower() == "hybrid" else 0

    return car


def generate_user_final_df(input_dict):
    """Generate final scaled DataFrame for model prediction"""
    processed_car = process_dict(input_dict)
    df_output = pd.DataFrame([processed_car])
    
    # Select only features the OLD model expects (10 features)
    df_output = df_output[[
        "model_encoded",
        "tax",
        "car_age",
        "mileage",
        "mpg",
        "engineSize",
        "transmission_Manual",
        "transmission_Semi-Auto",
        "fuelType_Diesel",
        "fuelType_Hybrid"
    ]]

    # Load and apply scaler
    with open("./preprocessing_results/full_dataset/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    df_output = scale_df(df_output, scaler)
    return df_output


# ============================================
# PREDICTION FUNCTIONS
# ============================================
def predict_price(input_dict):
    """Predict scaled log price"""
    df_input = generate_user_final_df(input_dict)
    
    with open("./FIRST_MODEL_TEST.pkl", "rb") as f:
        model = pickle.load(f)
    
    predicted_price = model.predict(df_input)
    return predicted_price[0]


def final_price(predicted_price_log):
    """Unscale and convert log price to actual price"""
    with open("./preprocessing_results/full_dataset/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Unscale price_log (index 9)
    price_log_min = scaler.data_min_[9]
    price_log_max = scaler.data_max_[9]
    price_log_range = price_log_max - price_log_min
    
    price_log_unscaled = predicted_price_log * price_log_range + price_log_min
    
    # Apply expm1 (because we used log1p in training)
    final_price_value = np.expm1(price_log_unscaled)
    
    return final_price_value