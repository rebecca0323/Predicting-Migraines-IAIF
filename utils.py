import numpy as np
import pandas as pd

def load_and_preprocess_data():
    """ Load data from total_diary_migrain.csv file
        and return the data matrix and labels

    Returns
    -------
        features - np.ndarray 
            Migraine data matrix

        features_list - list
            List of Features in our migraine data matrix
        
        labels - np.ndarray
            Truth labels
    """

    data = pd.read_csv('total_diary_migraine.csv', header=0)
    features = pd.DataFrame(data)

    features['no_headache_day'].fillna('N', inplace=True)

    features['migraine'].fillna(0, inplace=True)
    features['headache_day'] = features['headache_day'].map({'Y':0, 'N':1})

    labels = np.array(features['migraine'])
    features = features.drop(['number', 'patient', 'ID', 'no_headache_day', 'migraine'], axis = 1)
    features_list = list(features.columns)
    features = np.array(features)
    features[np.isnan(features)] = 0

    return (features, features_list, labels)