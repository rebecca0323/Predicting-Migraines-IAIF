import numpy as np
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

features, features_list, labels = utils.load_and_preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

isSMOTE = True
if isSMOTE: 
    os = SMOTE(random_state=0)
    os_data_X,os_data_y=os.fit_sample(X_train, y_train)
    X_train = pd.DataFrame(data=os_data_X)
    y_train = pd.DataFrame(data=os_data_y)
    print("length of oversampled data is ",len(os_data_X))
    print(y_train[0].value_counts())
    y_train = np.array(y_train).ravel()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

isPCA = False
if isPCA:
    pca = PCA(n_components=30) # 34 total features
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_

clf = SVC(kernel='poly')
clf.fit(X_train,y_train)

isRFE = False
if isRFE:
    selector = RFE(clf, n_features_to_select=30)
    selector.fit(X_train, y_train)

    y_pred = selector.predict(X_test)
else:
    y_pred = clf.predict(X_test)

print('Accuracy of SVM classifier on test set: {:.5f}'.format(accuracy_score(y_test,y_pred)))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test, y_pred, digits=5))
"""
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
"""