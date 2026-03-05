import joblib
import pandas as pd
from modules.feature_engineering import EngineerFeatures
from fastapi import FastAPI

package = joblib.load("readmission_production.joblib")

model = package["model"] # calibrated pipeline
threshold = package["threshold"]
features = package["features"]

app = FastAPI() # creates application object

@app.post("/predict") # route decorator
def predict(patient: dict):
    df = pd.DataFrame([patient]) # convert input into data frame
    df = df[features] # enforce feature order

    proba = model.predict_proba(df)[:,1][0]
    flag = int(proba >= threshold)

    return {
        "readmission_risk": float(proba),
        "high_risk": flag
    }

# start command 

# in bash:
# uvicorn app:app --host 0.0.0.0 --port 10000

# in browser:
# http://localhost:10000/docs
