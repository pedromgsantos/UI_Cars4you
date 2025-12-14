# Core libraries
import numpy as np
import pandas as pd
import pickle

CURRENT_YEAR = 2020

brand_mapping = pd.read_csv('./mapping_dicts/brand_mapping.csv')
fuelType_mapping = pd.read_csv('./mapping_dicts/fueltype_mapping.csv')
model_mapping = pd.read_csv('./mapping_dicts/model_mapping.csv')
transmission_mapping = pd.read_csv('./mapping_dicts/transmission_mapping.csv')

brand_map = dict(zip(brand_mapping["Variation"].str.strip(), brand_mapping["AssignedValue"].str.strip()))
fuelType_map = dict(zip(fuelType_mapping["Variation"].str.strip(), fuelType_mapping["AssignedValue"].str.strip()))
model_map = dict(zip(model_mapping["Variation"].str.strip(), model_mapping["AssignedValue"].str.strip()))
transmission_map = dict(zip(transmission_mapping["Variation"].str.strip(), transmission_mapping["AssignedValue"].str.strip()))


def fix_categorical_input(car_dict):
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


def get_brand_from_model(model_name):
    """
    Given a model name, return its corresponding Brand.
    """
    map_dict = dict(zip(
        model_mapping["AssignedValue"].astype(str).str.strip(),
        model_mapping["Brand"].astype(str).str.strip()
    ))
    model_name = str(model_name).strip()
    return map_dict.get(model_name, None)


def fix_empty_categorical(car_dict):
    """
    Fill missing categorical values with safe fallbacks.
    """
    if car_dict.get("fuelType") == "Other":
        car_dict["fuelType"] = np.nan
    if car_dict.get("transmission") == "Unknown":
        car_dict["transmission"] = np.nan

    # Fix Brand if missing (infer from model mapping)
    if (pd.isna(car_dict.get("Brand")) or car_dict.get("Brand") is None) and pd.notna(car_dict.get("model")):
        inferred = get_brand_from_model(car_dict["model"])
        if inferred is not None:
            car_dict["Brand"] = inferred

    # Fix fuelType if missing
    if pd.isna(car_dict.get("fuelType")) or car_dict.get("fuelType") is None:
        car_dict["fuelType"] = "Petrol"

    # Fix transmission if missing
    if pd.isna(car_dict.get("transmission")) or car_dict.get("transmission") is None:
        car_dict["transmission"] = "Manual"

    return car_dict


# fix categorical_input
# =========================================================================================================
# =========================================================================================================
# fix numerical_input

def fix_empty_numerical(car_dict):
    """
    Fill missing numerical values with simple defaults.
    """
    if car_dict.get("year") is None or pd.isna(car_dict.get("year")):
        car_dict["year"] = 2015

    if car_dict.get("mileage") is None or pd.isna(car_dict.get("mileage")):
        car_dict["mileage"] = 50000.0

    if car_dict.get("engineSize") is None or pd.isna(car_dict.get("engineSize")):
        car_dict["engineSize"] = 2.0

    if car_dict.get("tax") is None or pd.isna(car_dict.get("tax")):
        car_dict["tax"] = 150.0

    if car_dict.get("mpg") is None or pd.isna(car_dict.get("mpg")):
        car_dict["mpg"] = 45.0

    if car_dict.get("previousOwners") is None or pd.isna(car_dict.get("previousOwners")):
        car_dict["previousOwners"] = 1.0

    return car_dict


# fix numerical_input
# =========================================================================================================
# =========================================================================================================
# fix outliers - on input fixing for UI, only manual outliers

def handle_outliers(car_dict):
    """
    Cap extreme values at row level. This function does NOT remove rows.
    Data observed above. NO data leakage.
    """
    if car_dict.get("year") is not None and not pd.isna(car_dict["year"]):
        if car_dict["year"] > 2020:
            car_dict["year"] = 2020  # rules of dataset

    if car_dict.get("mileage") is not None and not pd.isna(car_dict["mileage"]):
        if car_dict["mileage"] < 0:
            car_dict["mileage"] = 0

    if car_dict.get("tax") is not None and not pd.isna(car_dict["tax"]):
        if car_dict["tax"] < 0:
            car_dict["tax"] = 0

    if car_dict.get("mpg") is not None and not pd.isna(car_dict["mpg"]):
        if car_dict["mpg"] < 0:
            car_dict["mpg"] = 0  # impossible value
        elif car_dict["mpg"] > 400:
            car_dict["mpg"] = 400  # conservative cap based on observed data

    if car_dict.get("engineSize") is not None and not pd.isna(car_dict["engineSize"]):
        if car_dict["engineSize"] < 0:
            car_dict["engineSize"] = 0

    if car_dict.get("paintQuality%") is not None and not pd.isna(car_dict["paintQuality%"]):
        if car_dict["paintQuality%"] < 0:
            car_dict["paintQuality%"] = 0
        elif car_dict["paintQuality%"] > 100:
            car_dict["paintQuality%"] = 100  # however, we can t have this feature on the final dataset

    if car_dict.get("previousOwners") is not None and not pd.isna(car_dict["previousOwners"]):
        if car_dict["previousOwners"] < 0:
            car_dict["previousOwners"] = 0

    return car_dict


# fix outliers
# =========================================================================================================
# =========================================================================================================
# correct data types

def correct_types(car_dict):
    """
    Convert features to correct data types.
    Maintains NaN for missing values.
    """
    # Ints
    int_features = ['year', 'previousOwners']
    for feature in int_features:
        if car_dict.get(feature) is not None and not pd.isna(car_dict[feature]):
            car_dict[feature] = int(car_dict[feature])

    # Floats
    float_features = ['mileage', 'tax', 'mpg', 'engineSize', 'paintQuality%']
    for feature in float_features:
        if car_dict.get(feature) is not None and not pd.isna(car_dict[feature]):
            car_dict[feature] = float(car_dict[feature])

    # Strings
    str_features = ['Brand', 'model', 'transmission', 'fuelType']
    for feature in str_features:
        value = car_dict.get(feature)
        if pd.isna(value) or value is None or value == 'nan' or value == 'None':  # error preventing
            car_dict[feature] = np.nan
        else:
            car_dict[feature] = str(value).strip()

    return car_dict


# data types
# =========================================================================================================
# =========================================================================================================
# auxiliar to main

def scale_df(df, scaler):
    """
    Manually scale dataframe using scaler's min/max values.

    Args:
        df: DataFrame to scale
        scaler: Fitted MinMaxScaler object

    Returns:
        Scaled DataFrame
    """
    df_scaled = df.copy()

    feature_mapping = {
        'year': 1,
        'car_age': 9,
        'mpg': 4,
        'engineSize': 5,
        'mileage': 2,
        'tax': 3,
        'model_encoded': 0,
        'paintQuality%': 6,
        'previousOwners': 7
    }

    # Scale each feature
    for feature, scaler_idx in feature_mapping.items():
        if feature not in df_scaled.columns:
            continue
        else:
            data_min = scaler.data_min_[scaler_idx]
            data_max = scaler.data_max_[scaler_idx]
            data_range = data_max - data_min

            if data_range > 0:
                df_scaled[feature] = (df_scaled[feature] - data_min) / data_range

    return df_scaled


# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# INPUT DATA PREPROCESSING

def find_encoding(model, year):
    with open("./preprocessing_results/full_dataset/encoding_maps.pkl", "rb") as f:
        data = pickle.load(f)
    if model in data["model_encoding_map"]:  # if it finds the model
        if year in data["model_encoding_map"][model]:
            return data["model_encoding_map"][model][year]
        else:
            return data["model_encoding_map"][model]["fallback"]
    else:
        return data["overall_fallback"]


# ============= PROCESS (single) DICT =============
def process_dict(input_dict):
    """
    Step 1: Clean test data (same as train/val cleaning)
    """
    car = fix_categorical_input(input_dict)
    car = fix_empty_categorical(car)
    car = fix_empty_numerical(car)
    car = handle_outliers(car)
    car = correct_types(car)

    car["car_age"] = CURRENT_YEAR - car["year"]

    car["model_encoded"] = find_encoding(car["model"], car["year"])

    car["transmission_Manual"] = 1 if car["transmission"].lower() == "manual" else 0
    car['transmission_Semi-Auto'] = 1 if car["transmission"].lower() == "semi-auto" else 0
    # Automatic is implicit (when both are 0)

    # fuelType column: original has multiple categories
    car["fuelType_Diesel"] = 1 if car["fuelType"].lower() == "diesel" else 0
    car["fuelType_Hybrid"] = 1 if car["fuelType"].lower() == "hybrid" else 0

    return car


def generate_user_final_df(dict):
    processed_car = process_dict(dict)

    df_output = pd.DataFrame([processed_car])
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

    # SCALER
    with open("./preprocessing_results/full_dataset/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df_output = scale_df(df_output, scaler)
    return df_output
