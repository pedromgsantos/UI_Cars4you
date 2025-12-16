# Cars 4 You - Streamlit UI

This repository contains the Streamlit user interface for the Cars4You project. It provides an analytics interface where a user can enter vehicle details and obtain a price prediction from a pre-trained regression model.

This implementation corresponds to the "Additional Insights" objective (c): create an analytics interface that returns a prediction when new input data is provided.

## Important Notes!!

**1. Simplified Preprocessing:**
This is a **simplified version** of the original notebook's preprocessing code. In the UI, **ALL fields are MANDATORY** and there are **NO missing values to impute**. As such, functions like `guess_brand_model()`, `fix_empty_categorical()`, and `fix_empty_numerical()` have been removed, reducing the code from 604 to 255 lines (-58%) while maintaining the exact same preprocessing logic for complete inputs.

**2. Model Selection:**
The model used in this UI is **NOT the best model** we could train, but rather a **simpler and lighter model** that fits within the ZIP file size limit for submission. The best-performing models and full training pipeline can be found in the main project repository.

## Scope

**Included:**
- Streamlit form to collect vehicle attributes
- Preprocessing required to transform user input into model-ready format
- Loading a pre-trained model and returning a predicted price (£)

**Not included** (these can be found in the main repository):
- Model training, benchmarking, and optimisation notebooks 
- The full project report and analysis deliverables 

## Inputs and Output

**Inputs (examples):**
- Brand, model, year
- mileage, tax, mpg, engineSize
- transmission, fuelType
- previousOwners, hasDamage

**Output:**
- Predicted vehicle price in £

## Repository Structure

```
UI_Cars4you/
├── app.py                                          # Streamlit application (UI + inference)
├── preprocessing_utils.py                          # Preprocessing utilities (simplified for UI)
├── mapping_dicts/                                  # CSV mapping files for standardization
│   ├── brand_mapping.csv
│   ├── fueltype_mapping.csv
│   ├── model_mapping.csv
│   └── transmission_mapping.csv
├── preprocessing_results/full_dataset/
│   ├── scaler.pkl                                  # MinMaxScaler for feature scaling
│   └── encoding_maps.pkl                           # Target encoding maps for model
├── files/
│   └── model_exported.pkl                          # Trained model file (lightweight version)
└── requirements.txt                                # Python dependencies
```

## Run Locally

### 1) Clone the repository
```bash
git clone https://github.com/pedromgsantos/UI_Cars4you.git
cd UI_Cars4you
```

### 2) Create and activate a virtual environment
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the app
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Online Demo

The application is available online here:
- **Live Demo:** https://uicars4you-mlproject.streamlit.app/

---

## Technical Details

**Preprocessing Pipeline:**
1. Fix categorical input variations (e.g., "VW" → "Volkswagen")
2. Handle outliers (cap extreme values)
3. Correct data types
4. Feature engineering (car_age, target encoding, one-hot encoding)
5. Feature selection (10 features for model)
6. MinMax scaling

**Model Features (10 total):**
- `model_encoded` (target encoded)
- `tax`, `car_age`, `mileage`, `mpg`, `engineSize` (numeric, scaled)
- `transmission_Manual`, `transmission_Semi-Auto` (one-hot)
- `fuelType_Diesel`, `fuelType_Hybrid` (one-hot)

**Note:** `paintQuality%` is auto-filled with a default value (75.0) since it's filled by mechanics post-evaluation and is not visible in the user interface.

---
