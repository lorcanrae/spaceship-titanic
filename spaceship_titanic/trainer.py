from google.cloud import storage
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline

# Model imports

# Pipeline imports

from spaceship_titanic.pipeline import create_preproc, create_models_dict

# GCP Cloud Storage Config - probably should be a params.py file

BUCKET_NAME = 'kaggle-spaceship-titanic'
BUCKET_TRAIN_DATA_PATH = 'data/train.csv'

MODEL_NAME = 'spaceship-titanic-model'
MODEL_VERSION = 'v1'


class Trainer:

    def __init__(self):
        self.data = None
        self.X_train = None
        self.y_train = None
        self.models_dict = create_models_dict()
        self.preproc = create_preproc()
        self.best_estimators = None
        self.final_pipe = None
        self.trained = False

    def load_data(self):
        '''Load data from GCP Bucket to frame'''
        self.data = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}')
        self.X_train = self.data.drop(columns='Transported')
        self.y_train = self.data['Transported']
        print('--- Data Successfully Loaded ----')
        print(f'--- X_train shape: {self.X_train.shape} ---')
        return self

    def gridsearchcv_tune(self, cv=5):
        '''Perform gridsearch based on models_dict["models"]["params"]'''
        for model in self.models_dict.values():
            search = GridSearchCV(make_pipeline(self.preproc, model['model']),
                                  model['params'],
                                  cv=cv,
                                  scoring='accuracy',
                                  n_jobs=1)
            search.fit(self.X_train, self.y_train)
            model['best_score'] = search.best_score_
            model['best_params'] = search.best_params_
            model['best_estimator'] = search.best_estimator_
            print(f'{model["name"]} gridsearch complete!')
        print('Hyperparameter tuning complete!')
        return self

    def assemble_pipe(self):
        '''Assemble final pipe with preprocess + voting classifier ensemble'''
        print('--- Assembling Final Pipe ---')
        self.best_estimators = [model['best_estimator'] for model in self.models_dict.values()]
        voting_classifier = VotingClassifier(
            estimators=self.best_estimators,
            voting='soft',
            weights=[1, 1, 1]
            )
        self.final_pipe = make_pipeline(self.preproc, voting_classifier)
        print('--- Final Pipe Assembled ---')
        return self

    def cross_val(self, cv=5):
        '''cross validate final pipe'''
        print('--- Cross Validating Final Pipe ---')
        score = cross_validate(self.final_pipe,
                               self.X_train,
                               self.y_train,
                               cv=cv,
                               scoring='accuracy',
                               n_jobs=1)
        print(score['test_score'].mean())
        print('--- Cross Validate complete')
        return score['test_score'].mean()

    def train_model(self):
        '''Train final pipeline with training data set'''
        print('Training model')
        self.final_pipe.fit(self.X_train, self.y_train)
        self.trained = True
        print('Model training complete!')
        return self

    def save_model(self):
        '''Save the model into a .joblib format'''
        if self.trained:
            self.local_model_name = f"model-{datetime.now().strftime('%y%m%d-%H%M%S')}.joblib"
            joblib.dump(self.final_pipe, self.local_model_name)
            print(f'{self.local_model_name} saved locally')
        else:
            print('Train model first!')
        return self


    ### GCP Methods

    def upload_model_to_gcp(self):
        '''Save model to GCP bucket'''
        if self.trained:
            print('Starting upload to GCP')
            client = storage.Client().bucket(BUCKET_NAME)
            gcp_storage_location = \
                f"models/{MODEL_NAME}/{MODEL_VERSION}/{self.local_model_name}"
            blob = client.blob(gcp_storage_location)
            blob.upload_from_filename(self.local_model_name)
            print(f"=> {self.local_model_name} uploaded to GCP Bucket {BUCKET_NAME}")
            print(f"inside {gcp_storage_location}")
        else:
            print('Train model first!')
        return self


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_data().gridsearchcv_tune().assemble_pipe()
    trainer.train_model()
    trainer.save_model().upload_model_to_gcp()
