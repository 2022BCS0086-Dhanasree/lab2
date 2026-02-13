from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

NAME = "Dhanasree Gidijala"
ROLL_NO = "2022BCS0086"

# Load trained model (trained with top 8 features)
model = joblib.load("model.pkl")

# Input schema – ONLY 8 FEATURES
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict(f: WineFeatures):

    # Arrange features in the SAME ORDER used during training
    x = np.array([[ 
        f.fixed_acidity,
        f.volatile_acidity,
        f.citric_acid,
        f.residual_sugar,
        f.chlorides,
        f.pH,
        f.sulphates,
        f.alcohol
    ]], dtype=float)

    pred = model.predict(x)[0]
    wine_quality = int(round(float(pred)))

    return {
        "name": NAME,
        "roll_no": ROLL_NO,
        "wine_quality": wine_quality
    }
