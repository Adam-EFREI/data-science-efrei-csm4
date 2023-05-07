# TP2 Main
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from typing import Callable


def load_dataset(filename: str, delim: str = ';'):
    try:
        ret = pd.read_csv(filename, delimiter=delim)

    except Exception as e:
        raise Exception(e)

    return ret


def model_trainer(dataset, x_var: str, algorithm: Callable, **kwargs):

    try:

        X_train, X_test, y_train, y_test = train_test_split(dataset.drop(x_var, axis=1),
                                                            dataset[x_var],
                                                            test_size=0.2)

        model = algorithm(**kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = model.score(y_pred, y_test)

    except Exception as e:
        raise e

    return model, accuracy



def tp2(**kwargs):
    required_args = [
        'dataset_filename'
    ]
    for arg in required_args:
        if arg not in kwargs.keys():
            raise Exception("Missing arg : {}".format(arg))

    try:
        ds = load_dataset(kwargs['dataset_filename'])

        print(ds)

        lr_model, lr_accuracy = model_trainer(ds, 'IsSpam', LogisticRegression, max_iter=10000)
        mlp_model, mlp_accuracy = model_trainer(ds, 'IsSpam', MLPClassifier, solver='lbfgs', max_iter=10000)
        svc_model, svc_accuracy = model_trainer(ds, 'IsSpam', svm.SVC)
        sgd_model, sgd_accuracy = model_trainer(ds, 'IsSpam', linear_model.SGDClassifier)
        rfc_model, rfc_accuracy = model_trainer(ds, 'IsSpam', ensemble.RandomForestClassifier)

        print(lr_model, lr_accuracy)
        print(mlp_model, mlp_accuracy)
        print(svc_model, svc_accuracy)
        print(sgd_model, sgd_accuracy)
        print(rfc_model, rfc_accuracy)

    except Exception as e:
        raise Exception("Could not load dataset", e)
