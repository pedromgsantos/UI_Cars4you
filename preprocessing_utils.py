# Core libraries
import numpy as np
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

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


def guess_brand_model(car_dict, df):
    """
    Fills missing brand and model based on priorities:
    1. engineSize
    2. transmission
    3. fuelType
    
    Args:
        car_dict: Dictionary with car features (brand, model, engineSize, transmission, fuelType)
        df: DataFrame with all car data
    
    Returns:
        Updated car_dict with filled brand and model (if possible)
    """
    
    # If both brand and model are not missing, return as is
    if pd.notna(car_dict.get("Brand")) or pd.notna(car_dict.get("model")):
        return car_dict
    
    engine = car_dict.get("engineSize")
    trans = car_dict.get("transmission")
    fuel = car_dict.get("fuelType")
    
    result_brand = None
    result_model = None
    
    # 1st priority: engineSize
    if pd.notna(engine):
        matches = df[df["engineSize"] == engine]
        if len(matches) > 0:
            if pd.isna(car_dict.get("Brand")):
                brand_mode = matches["Brand"].mode()
                if len(brand_mode) > 0:
                    result_brand = brand_mode[0]
            
            if pd.isna(car_dict.get("model")):
                model_mode = matches["model"].mode()
                if len(model_mode) > 0:
                    result_model = model_mode[0]
    
    # 2nd priority: transmission (if still missing)
    if (result_brand is None or result_model is None) and pd.notna(trans):
        matches = df[df["transmission"] == trans]
        if len(matches) > 0:
            if result_brand is None and pd.isna(car_dict.get("Brand")):
                brand_mode = matches["Brand"].mode()
                if len(brand_mode) > 0:
                    result_brand = brand_mode[0]
            
            if result_model is None and pd.isna(car_dict.get("model")):
                model_mode = matches["model"].mode()
                if len(model_mode) > 0:
                    result_model = model_mode[0]
    
    # 3rd priority: fuelType (if still missing)
    if (result_brand is None or result_model is None) and pd.notna(fuel):
        matches = df[df["fuelType"] == fuel]
        if len(matches) > 0:
            if result_brand is None and pd.isna(car_dict.get("Brand")):
                brand_mode = matches["Brand"].mode()
                if len(brand_mode) > 0:
                    result_brand = brand_mode[0]
            
            if result_model is None and pd.isna(car_dict.get("model")):
                model_mode = matches["model"].mode()
                if len(model_mode) > 0:
                    result_model = model_mode[0]
    
    # Update the dictionary with found values
    if result_brand is not None:
        car_dict["Brand"] = result_brand
    if result_model is not None:
        car_dict["model"] = result_model
    
    return car_dict

def get_model(brand_name, engineSize, transmission, fuelType, df):
    """
    Returns the model (mode) based on priorities:
    1. brand + engineSize
    2. brand + transmission
    3. brand + fuelType
    4. brand only
    """
    
    # Filter only cars from the same brand
    same_brand = df[df["Brand"] == brand_name]
    
    if len(same_brand) == 0:
        return None
    
    # 1st priority: brand + engineSize
    if pd.notna(engineSize):
        mode = same_brand[same_brand["engineSize"] == engineSize]["model"].mode()
        if len(mode) > 0:
            return mode[0]
    
    # 2nd priority: brand + transmission
    if pd.notna(transmission):
        mode = same_brand[same_brand["transmission"] == transmission]["model"].mode()
        if len(mode) > 0:
            return mode[0]
    
    # 3rd priority: brand + fuelType
    if pd.notna(fuelType):
        mode = same_brand[same_brand["fuelType"] == fuelType]["model"].mode()
        if len(mode) > 0:
            return mode[0]
    
    # 4th priority: brand only
    mode = same_brand["model"].mode()
    if len(mode) > 0:
        return mode[0]
    
    return None # hope never happens

def get_brand_from_model(model_name):
    """
    Given a model name, return its corresponding Brand.
    """
    map_dict = dict(zip(
        model_mapping["AssignedValue"].str.strip(),
        model_mapping["Brand"].str.strip()
    ))
    model_name = str(model_name).strip()
    return map_dict.get(model_name, None)


def fix_empty_categorical(car_dict, df):
    """
    Fill missing categorical values using mode of the model.
    If model is missing, it returns None (to signal row should be dropped).
    """
    if car_dict["fuelType"] == "Other":
        car_dict["fuelType"] = np.nan
    if car_dict["transmission"] == "Unknown":
        car_dict["transmission"] = np.nan

    car_dict = guess_brand_model(car_dict, df) # only changes if both brand and model are missing
        
    # Fix model if missing
    if pd.isna(car_dict["model"]) or car_dict["model"] is None:
        car_dict["model"] = get_model(car_dict["Brand"], car_dict["engineSize"],
                                      car_dict["transmission"], car_dict["fuelType"], df)

    # Fix Brand if missing
    if pd.isna(car_dict["Brand"]) or car_dict["Brand"] is None:
        car_dict["Brand"] = get_brand_from_model(car_dict["model"])
    
    # Fix fuelType if missing
    if pd.isna(car_dict["fuelType"]) or car_dict["fuelType"] is None:
        if not pd.isna(car_dict["model"]) and car_dict["model"] is not None:
            model_clean = str(car_dict["model"]).strip()
            mode_fuelType = df.loc[df["model"].str.strip().str.lower() == model_clean.lower(), "fuelType"].mode()
            car_dict["fuelType"] = mode_fuelType[0] if len(mode_fuelType) > 0 else df["fuelType"].mode()[0]
        else:
            car_dict["fuelType"] = df["fuelType"].mode()[0]
    
    # Fix transmission if missing
    if pd.isna(car_dict["transmission"]) or car_dict["transmission"] is None:
        if not pd.isna(car_dict["model"]) and car_dict["model"] is not None:
            model_clean = str(car_dict["model"]).strip()
            mode_transmission = df.loc[df["model"].str.strip().str.lower() == model_clean.lower(), "transmission"].mode()
            car_dict["transmission"] = mode_transmission[0] if len(mode_transmission) > 0 else df["transmission"].mode()[0]
        else:
            car_dict["transmission"] = df["transmission"].mode()[0]
    
    return car_dict

# fix categorical_input
# =========================================================================================================
# =========================================================================================================
# fix numerical_input

def get_median_with_fallback(df, car_dict, feature, groupby_cols, fallback_groupby):
    """
    Get median value for a feature based on groupby columns.
    Falls back to fallback_groupby if no data found.
    """

    mask = pd.Series([True] * len(df), index=df.index)
    for col in groupby_cols:
        value = car_dict.get(col)
        if not pd.isna(value) and value is not None and value != 'nan':
            if col == 'year':
                # Include current year +/- 1 to allow for less overfit/more broad values
                mask = mask & (df[col].between(value - 1, value + 1))
            else:
                mask = mask & (df[col] == value)
    
    filtered_data = df.loc[mask, feature]
    if len(filtered_data) > 0 and not filtered_data.isna().all():
        median_val = filtered_data.median()
        if not pd.isna(median_val):
            return median_val
    
    if fallback_groupby:
        mask = pd.Series([True] * len(df), index=df.index)
        for col in fallback_groupby:
            value = car_dict.get(col)
            if not pd.isna(value) and value is not None and value != 'nan':
                if col == 'year':
                    # Include current year ±1
                    mask = mask & (df[col].between(value - 1, value + 1))
                else:
                    mask = mask & (df[col] == value)
        
        filtered_data = df.loc[mask, feature]
        if len(filtered_data) > 0 and not filtered_data.isna().all():
            median_val = filtered_data.median()
            if not pd.isna(median_val):
                return median_val
    
    return df[feature].median()


def fix_empty_numerical(car_dict, df):
    """
    Fill missing numerical values using medians based on similar cars.
    """
    # Year - Median of (model), no fallback 
    if car_dict["year"] is None or pd.isna(car_dict["year"]):
        car_dict["year"] = get_median_with_fallback(
            df,
            car_dict,
            feature='year',
            groupby_cols=['model'],
            fallback_groupby=None
    )
        

    # Mileage - Median of (model, year±1), fallback to year±1
    if car_dict["mileage"] is None or pd.isna(car_dict["mileage"]):
        car_dict["mileage"] = get_median_with_fallback(
            df,
            car_dict,
            feature='mileage',
            groupby_cols=['model', 'year'],
            fallback_groupby=['year']
        )

    # Engine Size - Median of (model, fuelType), fallback to model
    if car_dict["engineSize"] is None or pd.isna(car_dict["engineSize"]):
        car_dict["engineSize"] = get_median_with_fallback(
            df,
            car_dict,
            feature='engineSize',
            groupby_cols=['model', 'fuelType'],
            fallback_groupby=['model']
        )

    # Tax - Median of (model, year±1), fallback to year±1
    if car_dict["tax"] is None or pd.isna(car_dict["tax"]):
        car_dict["tax"] = get_median_with_fallback(
            df,
            car_dict,
            feature='tax',
            groupby_cols=['model', 'year'],
            fallback_groupby=['year']
        )

    # MPG - Median of (model, fuelType), fallback to fuelType
    if car_dict["mpg"] is None or pd.isna(car_dict["mpg"]):
        car_dict["mpg"] = get_median_with_fallback(
            df,
            car_dict,
            feature='mpg',
            groupby_cols=['model', 'fuelType'],
            fallback_groupby=['fuelType']
        )  
    # PreviousOwners - Median of (model, year), fallback to model
    if car_dict["previousOwners"] is None or pd.isna(car_dict["previousOwners"]):
        car_dict["previousOwners"] = get_median_with_fallback(
            df,
            car_dict,
            feature='previousOwners',
            groupby_cols=['model', 'year'],
            fallback_groupby=['model']
        )  

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
            car_dict["year"] = 2020 # rules of dataset

    if car_dict.get("mileage") is not None and not pd.isna(car_dict["mileage"]):
        if car_dict["mileage"] < 0:
            car_dict["mileage"] = 0
        
    if car_dict.get("tax") is not None and not pd.isna(car_dict["tax"]):
        if car_dict["tax"] < 0:
            car_dict["tax"] = 0

    if car_dict.get("mpg") is not None and not pd.isna(car_dict["mpg"]):
        if car_dict["mpg"] < 0:
            car_dict["mpg"] = 0 # impossible value
        elif car_dict["mpg"] > 400:
            car_dict["mpg"] = 400 # conservative cap based on observed data

    if car_dict.get("engineSize") is not None and not pd.isna(car_dict["engineSize"]):
        if car_dict["engineSize"] < 0:
            car_dict["engineSize"] = 0

    if car_dict.get("paintQuality%") is not None and not pd.isna(car_dict["paintQuality%"]):
        if car_dict["paintQuality%"] < 0:
            car_dict["paintQuality%"] = 0
        elif car_dict["paintQuality%"] > 100:
            car_dict["paintQuality%"] = 100 # however, we can t have this feature on the final dataset
    
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
        if pd.isna(value) or value is None or value == 'nan' or value == 'None': # error preventing
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

    '''with open('./preprocessing_results/full_dataset/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        print(scaler.get_feature_names_out()) # util, criei o feature mapping abaixo com base nisto'''
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
                #print(f"Scaled {feature} (idx={scaler_idx})") # debug
    
    return df_scaled

# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# INPUT DATA PREPROCESSING


def find_encoding(model, year):
    with open("./preprocessing_results/full_dataset/encoding_maps.pkl", "rb") as f:
        data = pickle.load(f)
    if model in data["model_encoding_map"]: # if it finds the model
        if year in data["model_encoding_map"][model]:
            return data["model_encoding_map"][model][year]
        else:
            return data["model_encoding_map"][model]["fallback"]
    else:
        return data["overall_fallback"]

train_df = pd.read_csv('./data/datatovisualization.csv')

# ============= PROCESS (single) DICT =============
def process_dict(input_dict):
    """
    Step 1: Clean test data (same as train/val cleaning)
    """    
    
    car = fix_categorical_input(input_dict)
    car = fix_empty_categorical(car, train_df)
    car = fix_empty_numerical(car, train_df)
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

#result = process_input(input_dict_example)

"""for key, value in result.items():
    print(f"{key:20}: {value}")"""

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

#result_df = generate_user_final_df(input_dict_example)
#print(result_df)

def predict_price(input_dict):
    """
    Given an input dictionary with car features, process it and predict the price using the trained model.
    
    Args:
        input_dict: Dictionary with car features.
    
    Returns:
        Predicted price.
    """
    # Process input dictionary to get final DataFrame
    df_input = generate_user_final_df(input_dict)
    
    # Load trained model
    with open("./FIRST_MODEL_TEST.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Predict price
    predicted_price = model.predict(df_input)
    
    return predicted_price[0]

input_dict_example = {
    "Brand": "VW",
    "model": "Golf",
    "year": 2016.0,
    "transmission": "Semi-Auto",
    "mileage": 28421.0,
    "fuelType": "Petrol",
    "tax": None,
    "mpg": 11.417267753816397,
    "engineSize": 2.0,
    "paintQuality%": 63.0,
    "previousOwners": 4.0,
    "hasDamage": 0.0
}

def final_price(predicted_price_log):
    """
    Unscale and apply exp to the log price prediction.
    
    Args:
        predicted_price_log: Scaled log price from model prediction.
    
    Returns:
        Final unscaled price.
    """
    # Load scaler
    with open("./preprocessing_results/full_dataset/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Unscale price_log (index 10 in scaler)
    price_log_min = scaler.data_min_[10]
    price_log_max = scaler.data_max_[10]
    price_log_range = price_log_max - price_log_min
    
    price_log_unscaled = predicted_price_log * price_log_range + price_log_min
    
    # Apply exp to get final price
    final_price_value = np.exp(price_log_unscaled)
    
    return final_price_value

"""prediction = predict_price(input_dict_example)
print("predicted target for dict:", prediction)
print("final price:", final_price(prediction))"""
'''
first_row_trainset = {
    "Brand": "VW",
    "model": "Golf",
    "year": 2016.0,
    "transmission": "Semi-Auto",
    "mileage": 28421.0,
    "fuelType": "Petrol",
    "tax": None,
    "mpg": 0,
    "engineSize": 2.0,
    "paintQuality%": 63.0,
    "previousOwners": 4.0,
    "hasDamage": 0.0
}

prediction = predict_price(first_row_trainset)
print("predicted target for dict:", prediction)
print("final price:", final_price(prediction))
'''