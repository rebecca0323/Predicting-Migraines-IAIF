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

clf = SVC(kernel='linear')
clf.fit(X_train,y_train)

isRFE = True
if isRFE:
    selector = RFE(clf, n_features_to_select=30)
    selector.fit(X_train, y_train)

    y_pred = selector.predict(X_test)
else:
    y_pred = clf.predict(X_test)

print('Accuracy of SVM classifier on test set: {:.5f}'.format(accuracy_score(y_test,y_pred)))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test, y_pred))