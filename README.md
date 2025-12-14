# Cars 4 You - Streamlit UI

This repository contains the Streamlit user interface for the Cars4You project. It provides an analytics interface where a user can enter vehicle details and obtain a price prediction from a pre-trained regression model.

This implementation corresponds to the “Additional Insights” objective (c): create an analytics interface that returns a prediction when new input data is provided.

## Scope

Included:
- Streamlit form to collect vehicle attributes
- Preprocessing required to transform user input into the model-ready format
- Loading a pre-trained model and returning a predicted price (£)

Not included (these can be found on the main repository):
- Model training, benchmarking, and optimisation notebooks 
- The full project report and analysis deliverables 

## Inputs and output

Inputs (examples):
- Brand, model, year
- mileage, tax, mpg, engineSize
- transmission, fuelType
- previousOwners, hasDamage

Output:
- Predicted vehicle price in £

## Repository structure

- `app.py` - Streamlit application (UI + inference)
- `preprocessing_utils.py` - preprocessing utilities used at inference time
- `mapping_dicts/` - CSV mapping files used by preprocessing (case-sensitive on Linux)
- `preprocessing_results/full_dataset/scaler.pkl` - scaler used to unscale the model output
- `files/random_forest_compressed.pkl` - model file (tracked with Git LFS)

## Run locally

1) Clone the repository and pull LFS files:
```bash
git lfs install
git clone https://github.com/pedromgsantos/UI_Cars4you.git
cd UI_Cars4you
git lfs pull
```

2) Create and Activate a virtual Environment
```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Run the app (locally)
```bash
streamlit run app.py
```

## Online demo

The application is available online here:
- https://uicars4you-mlproject.streamlit.app/
