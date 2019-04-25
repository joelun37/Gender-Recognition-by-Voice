# Import libraries and packages
import librosa as lr
import librosa.display as lrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
import re
from sklearn import preprocessing
from glob import glob
import wave
from scipy.io import wavfile
import csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

from Gender_Recognition_Functions import *

n_train = round(len(y_)*(80.00/100.0))

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
skCl = KNeighborsClassifier(10)
skCl.fit(x_[:n_train,:], y_[:n_train])

y_pred = skCl.predict(x_[n_train:,:])
acc = np.mean(y_pred == y_[n_train:])
print("Accuracy : " + str(acc))

from sklearn.metrics import f1_score
f1s = f1_score(y_[n_train:], y_pred, average='macro')  
print("F1 Score : " + str(f1s))


# Linear SVM

n_train = round(len(y_)*(80.00/100.0))

svm_clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("linear_svc", LinearSVC(C = 1, loss = "hinge"))
])

svm_clf.fit(x_[:n_train,:], np.ravel(y_[:n_train]))

y_pred = svm_clf.predict(x_[n_train:,:])

acc = np.mean(y_pred == y_[n_train:])
print("Accuracy : " + str(acc))

from sklearn.metrics import f1_score
f1s = f1_score(y_[n_train:], y_pred, average='macro')  
print("F1 Score : " + str(f1s))


# Nonlinear SVM
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C = 10, loss = "hinge"))
])

polynomial_svm_clf.fit(x_[:n_train,:], np.ravel(y_[:n_train]))

y_pred = polynomial_svm_clf.predict(x_[n_train:,:])

acc = np.mean(y_pred == y_[n_train:])
print("Accuracy : " + str(acc))

from sklearn.metrics import f1_score
f1s = f1_score(y_[n_train:], y_pred, average='macro')  
print("F1 Score : " + str(f1s))

# Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_[:n_train,:], np.ravel(y_[:n_train]))

y_pred = log_reg.predict(x_[n_train:,:])

acc = np.mean(y_pred == y_[n_train:])
print("Accuracy : " + str(acc))

from sklearn.metrics import f1_score
f1s = f1_score(y_[n_train:], y_pred, average='macro')  
print("F1 Score : " + str(f1s))
