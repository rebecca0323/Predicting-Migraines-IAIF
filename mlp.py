import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

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

# Defining model
model = Sequential()
model.add(Dense(15, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs = 100, batch_size = 100, verbose=2)

_, accuracy = model.evaluate(X_train, y_train)
print('Training Accuracy: %.2f' % (accuracy*100))

_, test_accuracy = model.evaluate(X_test, y_test)
print('Testing Accuracy: %.2f' % (test_accuracy*100))


"""
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=model, epochs=100, batch_size=50, verbose=2)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""