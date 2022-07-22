# $DELETE_BEGIN
import pandas as pd

import joblib
import gcsfs
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spaceship_titanic.trainer import BUCKET_NAME, MODEL_NAME, MODEL_VERSION

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

# Test url local
# http://127.0.0.1:8000/predict?HomePlanet=Europa&CryoSleep=False&Cabin_Deck=T&Cabin_Level=1000&Cabin_Side=S&Destination=TRAPPIST-1e&Age=32&VIP=False&RoomService=100&FoodCourt=5000&ShoppingHall=10&Spa=0&VRDeck=0
# http://127.0.0.1:8000/predict?HomePlanet=Europa&CryoSleep=False&Cabin_Deck=C&Cabin_Level=100&Cabin_Side=S&Destination=TRAPPIST-1e&Age=32&VIP=False&RoomService=10000&FoodCourt=50000&ShoppingHall=10&Spa=10000&VRDeck=10000

@app.get("/")
def index():
    return dict(greeting='hello')

@app.get("/predict")
def predict(HomePlanet,
            CryoSleep,
            Cabin_Deck,
            Cabin_Level,
            Cabin_Side,
            Destination,
            Age,
            VIP,
            RoomService=0,
            FoodCourt=0,
            ShoppingMall=0,
            Spa=0,
            VRDeck=0,
            ):

    # Transforms
    Cabin = '/'.join([Cabin_Deck.upper(), Cabin_Level, Cabin_Side.upper()])

    # Gate-keeping
    allow_homeplanet = ['Europa, Earth, Mars']
    allow_destinations = ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22']
    allow_bool = ['True', 'False']

    # Hardcoded params to build X (features not used in the model)
    PassengerId = '0001_01'
    Name = 'Maham Ofracculy'

    # Build X
    X = pd.DataFrame(dict(
        PassengerId = [PassengerId],
        HomePlanet = [HomePlanet],
        CryoSleep = [bool(CryoSleep)],
        Cabin = [Cabin],
        Destination = [Destination],
        Age = [float(Age)],
        VIP = [bool(VIP)],
        RoomService = [float(RoomService)],
        FoodCourt = [float(FoodCourt)],
        ShoppingMall = [float(ShoppingMall)],
        Spa = [float(Spa)],
        VRDeck = [float(VRDeck)],
        Name = [Name]
    ))

    # Loading model from GCP

    gcs_model_name = 'model-220721-162104.joblib'

    # TODO: put this into a function - there should be a gcp.py module
    # fs = gcsfs.GCSFileSystem()
    # with fs.open(f'{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}/{gcs_model_name}') as file:
    #     pipeline = joblib.load(file)

    rel_path = f'../saved_models/{gcs_model_name}'
    abs_path = os.path.dirname(__file__)

    pipeline = joblib.load(os.path.join(abs_path, rel_path))
    print(abs_path)
    y_pred = bool(pipeline.predict(X)[0])
    return dict(Transported=y_pred) # for some reason it can't return a numpy.bool_, has to be a regular bool

if __name__ == '__main__':
    y_pred = predict('Europa', 'True', 'A', '10', 'P', 'TRAPPIST-1e', '32', 'False')
    print(y_pred['Transported'])
