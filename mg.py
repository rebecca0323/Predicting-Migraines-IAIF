import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot
from sklearn.tree import export_graphviz
import pydot
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv('total_diary_migraine.csv', header=0)
features = pd.DataFrame(data)

features['no_headache_day'].fillna('N', inplace=True)

features['migraine'].fillna(0, inplace=True)
features['headache_day'] = features['headache_day'].map({'Y':0, 'N':1})

labels = np.array(features['migraine'])
features = features.drop(['number', 'patient', 'ID', 'no_headache_day', 'migraine'], axis = 1)
feature_list = list(features.columns)
features = np.array(features)
features[np.isnan(features)] = 0

# Using Skicit-learn to split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

isSMOTE = True
if isSMOTE: 
    os = SMOTE(random_state=0)
    os_data_X,os_data_y=os.fit_sample(X_train, y_train)
    X_train = pd.DataFrame(data=os_data_X)
    y_train = pd.DataFrame(data=os_data_y)
    print("length of oversampled data is ",len(os_data_X))
    print(y_train[0].value_counts())

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = RandomForestClassifier(n_estimators=30, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#Individual variable importance
importances = list(clf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]