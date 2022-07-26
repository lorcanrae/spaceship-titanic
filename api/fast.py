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

@app.get("/")
def index():
    return dict(greeting='thanks for looking at my API',
                info={'me': 'Lorcan Rae',
                      'my_github': 'github.com/lorcanrae'
                      },
                data={'data_set': 'kaggle_spaceship_titanic',
                      'kaggle_link': 'https://www.kaggle.com/competitions/spaceship-titanic',
                      'field_format': {'FieldName': 'default median or mode value'},
                      'fields_default': {'HomePlanet': 'Earth',
                                'CryoSleep': False,
                                'Cabin_Deck': 'F',
                                'Cabin_Level': '82',
                                'Caibn_Side': 'S',
                                'Destination': 'TRAPPIST-1e',
                                'Age': 27,
                                'VIP': False,
                                'RoomService': 0,
                                'FoodCourt': 0,
                                'ShoppingMall': 0,
                                'Spa': 0,
                                'VRDeck': 0
                                }
                      }
                )

@app.get("/predict")
def predict(HomePlanet='Earth',
            CryoSleep=False,
            Cabin_Deck='F',
            Cabin_Level='82',
            Cabin_Side='S',
            Destination='TRAPPIST-1e',
            Age=27,
            VIP=False,
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

    # Loading model from GCP - hard pass, no reason to fetch model for each predict

    # TODO: put this into a function - there should be a gcp.py module
    # fs = gcsfs.GCSFileSystem()
    # with fs.open(f'{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}/{gcs_model_name}') as file:
    #     pipeline = joblib.load(file)

    local_model_name = 'model-220721-162104.joblib'
    rel_path = f'../saved_models/{local_model_name}'
    abs_path = os.path.dirname(__file__)

    pipeline = joblib.load(os.path.join(abs_path, rel_path))
    # for some reason it can't return a numpy.bool_, has to be a regular bool
    y_pred = bool(pipeline.predict(X)[0])
    return dict(Transported=y_pred)

# API URL: https://spaceship-titanic-api-zby5e6zv3q-ew.a.run.app
