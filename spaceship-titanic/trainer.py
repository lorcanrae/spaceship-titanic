from google.cloud import storage
import numpy as np
import pandas as pd
import joblib
import datetime

# Model imports


# Pipeline imports


# GCP Cloud Storage Config

BUCKET_NAME = 'kaggle-spaceship-titanic'
BUCKET_TRAIN_DATA_PATH = 'data/train.csv'

MODEL_NAME = 'spaceship-titanic-model'
MODEL_VERSION = 'v1'
DATE_STAMP = datetime.now().strftime('%y%m%d')


# Code

def get_data():
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    X_train = df.drop(columns='Transported')
    y_train = df['Transported']
    return (X_train, y_train)

def assemble_pipe():
    pass

# Upload and save model to GCP

def upload_model_to_gcp(model_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blop(f'models/{DATE_STAMP}/')
    blob.upload_from_filename(model_name)

def save_model(model):
    # save model locally
    time_stamp = datetime.now().strftime("%y%m%d-%H%M")
    model_name = f'{MODEL_NAME}-{MODEL_VERSION}-{time_stamp}.joblib'
    joblib.dump(model, model_name)
    print(f'saved {model_name} locally!')

    # Push to gcp
    upload_model_to_gcp(model_name)
    print(f'uploaded {model} to GCP cloud storage under')
    print(f'=> gs://{BUCKET_NAME}/models/{DATE_STAMP}/{model_name}')


if __name__ == '__main__':
    X_train, y_train = get_data()

    # model = train(X_train, y_train)

    # save_model(model)
