import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.metrics
import numpy as np
from collections import Counter
import math
from sklearn.utils import resample
from iblearn.over_sampling import SMOTE


def resample_manual(data, chosen):
    x, y = [], []

    for i in range(len(data)):

        features = np.array([feature for feature in data[i][0]]).flatten()
        # manual resampling method
        if i == chosen - 1:
            for resample in range(2 * len(data) // round(math.sqrt(len(data)))):
                x.append(features / (math.sqrt(len(features)) * np.mean(features)) + (1 / (resample + 1)))
                y.append(1)
        else:
            for resample in range(2):
                x.append(features / (math.sqrt(len(features)) * np.mean(features) + (1 / (resample + 1))))
                y.append(0)

    return x, y


def oversample_minority(data, chosen):
    x, y = [], []
    sub, not_sub = [], []

    for i in range(len(data)):
        features = np.array([feature for feature in data[i][0]]).flatten()

        # put chosen persons data into sub
        if i == chosen - 1:
            sub.append(features)

        # put other subject data in x, and put 0 in y
        else:
            x.append(features)
            not_sub.append(features)
            y.append(0)

    # resample the subject
    oversample = resample(sub, replace=True, n_samples=len(not_sub), random_state=42)
    x.append(oversample)

    # append 1, for all the resampled subject data in x
    for i in range(len(oversample)):
        y.append(1)

    return x, y


# updated smote
# I just put the body of this function into the main Mlp to replace the manual resample method
def smote_resample(data, chosen):
    x, y = [], []
    sx, sx = [], []

    for i in range(len(data)):
        features = np.array([feature for feature in data[i][0]]).flatten()
        # put all data in x, with 1 or 0
        x.append(features)

        # bc the first 6 in data will be the selected subject
        # I also just typed y = [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0] (6 1's, 11 0's corresponding to the amount of correct subject run sets, and non-sub run sets)
        if i == (0 or 1 or 2 or 3 or 4 or 5 or 6):
            y.append(1)
        else:
            y.append(0)

    # resample all data

    sm = SMOTE(random_state=42, sampling_strategy='minority')
    sx, sy = sm.fit_resample(x, y)  # sx, sy should have the same size now.
    return sx, sy