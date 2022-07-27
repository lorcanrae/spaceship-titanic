# Spaceship Titanic - who will survive!

This is a submission for an open [kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/data).
In addition to submitting a notebook on kaggle, I have also:
- Performed an Exploratory Data Analysis.
- Packaged up the codebase.
- Used GCP's AI Platform and Cloud Storage to train and store the model in the cloud.
- Containerized the package using Docker.
- Built and exposed an [API](https://spaceship-titanic-api-zby5e6zv3q-ew.a.run.app) using GCP Cloud Run.
- Created a [front end](https://lorcanrae-spaceship-titanic-web-0--home-w79no8.streamlitapp.com/) that queries the API.

![](https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg)
![](https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg)
![](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)
![](https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg)
![](https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg)
![](https://streamlit.io/images/brand/streamlit-mark-color.png)

# API

I created an API using FastAPI and Uvicorn, packaged the container using Docker, and
exposed the API using GCP Container Registry and Cloud Run.

# Model

6 Models where initially internally scored on 5 fold cross validation of the training dataset:
- Logistic Regressor
- Support Vector classifier (SVM) with a Linear kernel
- Support Vector classifier with a Radial Basis Function (RBF) kernel
- K Nearest Neighbours (KNN) classifier
- Random Forest
- Gradient Boosted Random Forest (GBR)

KNN, SVM with a RBF kernel, and GBR where scored the highest. An ensemble voting
classifier of these three models was used. The model was placed at the end of a pipeline
that transformed, imputed, and scaled the data.

Hyper paramaters where tuned using a grid search on another 5 fold cross validation
of the dataset.

The model has an accuracy of 80.00009% on the test data set.

This model was trained and tuned in the cloud using GCP AI Platform and Cloud Storage - for my own practice.

# Data

The data is part of an open dataset for the [Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic/data)
hosted on kaggle.com. The train data set contains 8693 rows and the test set contains 4277 rows.
