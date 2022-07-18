from category_encoders import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer, RobustScaler, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

from preprocessing import drop_columns, cabin_transform, feat_cols

def create_preproc_1(X_train):
    drop_col = FunctionTransformer(drop_columns)
    cabin_trans = FunctionTransformer(cabin_transform)
    return make_pipeline(drop_col, cabin_trans)

def create_preproc_2(X_train):
    # Imputers
    median_imp = SimpleImputer(strategy='median')
    mode_imp = SimpleImputer(strategy='most_frequent')

    # Encoders
    ohe_enc = OneHotEncoder(sparse=False)
    ord_enc = OrdinalEncoder()

    # Scalers
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()

    # Four parallel pipes
    trans_num = make_pipeline(median_imp, r_scaler)
    trans_ohe = make_pipeline(mode_imp, ohe_enc)
    trans_ordstr = make_pipeline(mode_imp, ord_enc, mm_scaler)
    trans_ordnum = make_pipeline(mode_imp, mm_scaler)

    preproc_2 = make_column_transformer(
        (trans_num, feat_cols()[0]),
        (trans_ohe, feat_cols()[1]),
        (trans_ordstr, feat_cols()[2]),
        (trans_ordnum, feat_cols()[3])
    )
    return preproc_2

def create_model():
    svc_rbf = SVC(kernel='rbf', probability=True)
    knn_cla = KNeighborsClassifier(n_neighbors=19)
    gbe_ens = GradientBoostingClassifier(
        max_depth=3,
        min_samples_leaf=3,
        min_samples_split=3,
        max_features='sqrt',
        subsample=0.8
        )
    voting_classifier = VotingClassifier(estimators=[svc_rbf, knn_cla, gbe_ens],
                                         voting='soft',
                                         weights=[1, 1, 1])
    return voting_classifier

def assemble_pipe():
    return make_pipeline(create_preproc_1, create_preproc_2, create_model)

if __name__ == '__main__':
    pass
