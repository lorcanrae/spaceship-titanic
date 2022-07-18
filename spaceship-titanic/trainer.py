from google.cloud import storage
import numpy as np
import pandas as pd
import joblib
import datetime

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from pipeline import create_preproc, create_models_dict

# Model imports


# Pipeline imports


# GCP Cloud Storage Config

BUCKET_NAME = 'kaggle-spaceship-titanic'
BUCKET_TRAIN_DATA_PATH = 'data/train.csv'

MODEL_NAME = 'spaceship-titanic-model'
MODEL_VERSION = 'v1'
DATE_STAMP = datetime.now().strftime('%y%m%d')


class Trainer:

    def __init__(self):
        self.data = None
        self.X_train = None
        self.y_train = None
        self.model_dict = create_models_dict()
        self.preproc = create_preproc()
        self.best_estimators = None
        self.final_pipe = None

    def load_data(self):
        self.data = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}')
        self.X_train = self.data.drop(columns='Transported')
        self.y_train = self.data['Transported']
        print(f'--- X_train shape: {self.X_train.shape}')

    def gridsearchcv_tune(self, cv=5):
        for model in self.model_dict.values():
            search = GridSearchCV(make_pipeline(self.preproc, model['model']),
                                  model['params'],
                                  cv=cv,
                                  scoring='accuracy',
                                  n_jobs=1)
            search.fit(self.X_train, self.y_train)
            model['best_score'] = search.best_score_
            model['best_params'] = search.best_params_
            model['best_estimator'] = search.best_estimator_
        return self

    def assemble_pipe(self):
        self.best_estimators = [model['best_estimator'] for model in self.models_dict.values()]
        voting_classifier = VotingClassifier(
            estimators=self.best_estimators,
            voting='soft',
            weights=[1, 1, 1]
            )
        self.final_pipe = make_pipeline(self.preproc, voting_classifier)
        return self


# Upload and save model to GCP

def upload_model_to_gcp(model_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'models/{DATE_STAMP}/')
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
    trainer = Trainer()
    trainer.load_data()

    # model = train(X_train, y_train)

    # save_model(model)
