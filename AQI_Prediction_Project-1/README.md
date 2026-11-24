# Air Quality Index (AQI) Prediction Using Environmental Data

## Project overview
This Python machine-learning project estimates Air Quality Index (AQI) from pollutant measurements (PM2.5, PM10, NO2, SO2, CO and O3) and weather conditions (temperature and humidity). It uses a Random Forest regression model and provides an interactive Streamlit dashboard.

## Project structure
- `app.py` — Streamlit application (dashboard, prediction form, dataset explorer, model evaluation)
- `train_model.py` — trains and evaluates the Random Forest model; saves `aqi_model.joblib`
- `data/aqi_environmental_data.csv` — sample environmental dataset
- `requirements.txt` — required Python packages

## Run on Windows (PowerShell)
1. Extract the ZIP file.
2. Open PowerShell in the extracted `AQI_Prediction_Project` folder.
3. Create a virtual environment:
   `python -m venv .venv`
4. Activate it:
   `.\.venv\Scripts\Activate.ps1`
5. Install dependencies:
   `python -m pip install -r requirements.txt`
6. Start the application:
   `python -m streamlit run app.py`

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Train model separately (optional)
Run `python train_model.py`. The script prints MAE, RMSE and R² and saves `aqi_model.joblib`. The Streamlit app also trains the model automatically when it starts, so this step is optional.

## Dataset note
The included CSV is a synthetic sample dataset created for demonstrating the full machine-learning workflow. It is not collected from live sensors and should not be used for official AQI reporting or health decisions. The target AQI is a teaching target derived from pollutant-related values, so results demonstrate the workflow rather than official CPCB methodology.

## Machine-learning workflow
1. Load and inspect environmental data.
2. Use pollutant and weather columns as model features.
3. Split records into training and testing sets.
4. Train a Random Forest Regressor.
5. Evaluate using MAE, RMSE and R².
6. Predict AQI for new measurements and display an indicative category.


## Additional uploaded dataset: country AQI history
The project now also includes `data/country_aqi_history.csv`, copied from the CSV supplied by the user. It contains 27,876 records and four fields: `Date`, `Country`, `Status`, and `AQI Value`.

Open the **Historical Country AQI** page in the Streamlit sidebar to:
- choose countries,
- view AQI trends over time,
- compare average AQI by country,
- filter records by AQI status,
- download the filtered records.

Important: this file does not contain PM2.5, PM10, NO2, SO2, CO, O3, temperature, or humidity columns. Therefore it is used for historical analysis and visualization, not as a replacement for the pollutant-feature model training dataset. The existing prediction model continues to use `data/aqi_environmental_data.csv`.
