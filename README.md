# Spaceship Titanic - who will survive!

The challenge is to build a _binary classification model_ to predict who have been transported to an alternate
dimension when the _Spaceship Titanic_ ran into a spacetime anomaly hidden within a dustcloud.
This is an [open competition](https://www.kaggle.com/competitions/spaceship-titanic) hosted by kaggle.com.

In addition to the [kaggle submission](https://www.kaggle.com/code/lorcansamuel/spaceship-titanic-eda-ensemble-using-pipes), I have also:
- Performed an Exploratory Data Analysis.
- Built a pipeline that transforms, imputes, encodes, and scales the data.
- Packaged the codebase.
- Used Google Cloud Platform (GCP) AI Platform and Cloud Storage to train and store the model in the cloud.
- Containerized the package using Docker.
- Built and exposed an [API](https://spaceship-titanic-api-zby5e6zv3q-ew.a.run.app) using GCP Cloud Run and Container Registry.
- Created a [front end](https://lorcanrae-spaceship-titanic-web-0--home-w79no8.streamlitapp.com/) that queries the API.

<p float='left'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg' width='75'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg' width='75'>
  <img src='https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg' width='75'>
  <img src='https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg' width='75'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg' width='75'>
  <img src='https://streamlit.io/images/brand/streamlit-mark-color.png' width='75'>

# API

I created an API using FastAPI and Uvicorn, packaged the container using Docker, pushed
the Docker image to GCP Container Registry, and exposed the API using GCP Cloud Run

# Model

6 Models where initially internally scored on 5 fold cross validation of the training dataset:
- Logistic Regressor
- Support Vector classifier (SVM) with a Linear kernel
- Support Vector classifier with a Radial Basis Function (RBF) kernel
- K Nearest Neighbours (KNN) classifier
- Random Forest
- Gradient Boosted Random Forest (GBR)

KNN, SVM with a RBF kernel, and GBR had the highest accuracy. An ensemble voting
classifier of these three models was used. The model was placed at the end of a pipeline
that transformed, imputed, encoded, and scaled the data.

Hyper paramaters where tuned using a grid search on another 5 fold cross validation
of the dataset.

The model has an accuracy of 80.0009% on the test data set.

This model was trained and tuned in the cloud using GCP AI Platform and Cloud Storage - for my own practice.

# Data

The data is part of an open dataset for the [Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic/data)
hosted on kaggle.com. The train data set contains 8693 rows and the test set contains 4277 rows.
y_test is withheld and requires a submission on kaggle.com for the model to be scored.
Refer to competition for details.
