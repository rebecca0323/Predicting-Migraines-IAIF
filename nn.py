import numpy as np
import pandas as pd
import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.utils import plot_model
import matplotlib.pyplot as plt

features, features_list, labels = utils.load_and_preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

isSMOTE = False
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
model.add(Dense(20, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

plot_model(model, to_file='model.png')

model.fit(X_train, y_train, epochs = 50, batch_size = 100, verbose=2)

_, accuracy = model.evaluate(X_train, y_train)
print('Training Accuracy: %.3f' % (accuracy*100))

_, test_accuracy = model.evaluate(X_test, y_test)
print('Testing Accuracy: %.3f' % (test_accuracy*100))


y_pred = model.predict_classes(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test, y_pred))

#Confusion matrix tells how many correct and incorrect predictions on training and testing data
confusion_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(confusion_matrix,cmap=plt.cm.Blues)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted No Migraines', 'Predicted Migraines'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual No Migraines', 'Actual Migraines'))
ax.set_ylim(1.5, -0.5)
thresh = confusion_matrix.max() / 2.
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        plt.text(j, i, format(confusion_matrix[i, j]),
                ha="center", va="center",
                color="black" if  confusion_matrix[i, j] == 0 or confusion_matrix[i, j] < thresh else "white") 
plt.savefig('Confusion Matrix')