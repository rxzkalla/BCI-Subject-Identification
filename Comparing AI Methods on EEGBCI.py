import os
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import numpy as np

import sklearn
import scipy
import numpy
import matplotlib
import pandas

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def run():
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    num_subjects = 10
    runs = [5, 9, 13]  # motor imagery: hands vs feet
    subjects_ica = {}

    # return ICA array for 10 subjects
    for i in range(num_subjects):
        subject = i
        raw_fnames = eegbci.load_data(subject, runs)
        raw = concatenate_raws(
            [read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(raw)
        montage = make_standard_montage(
            'standard_1005')  #arrangemnent of electrodes
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))

        raw.crop(tmax=60.)
        raw.filter(14., 30.)

        picks = pick_types(raw.info, eeg=True)

        ica = ICA(n_components=15, random_state=97)
        ica.apply(raw)

        raw.load_data()
        #ica.plot_sources(raw, show_scrollbars=False)
        #ica.plot_properties(raw, picks=[0, 1])
        #ica.plot_scores()

        icaArray = ica.get_components()
        print(icaArray)
        subjects_ica[i] = icaArray
        #features = ica.get_sources()


  num_train = 7
  num_test = num_subjects - num_train

  X_raw_train = []
  Y_raw_train = []
  X_raw_test = []
  Y_raw_test = []

  for i in range(num_train):
    X_raw_train.append(subjects_ica[i])
    Y_raw_train.append(i)

  for i in range(num_test):
    X_raw_test.append(subjects_ica[i])
    Y_raw_test.append(i)

    # make an array of the analysis models
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

  for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
