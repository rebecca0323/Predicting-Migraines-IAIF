import pandas as pd
import numpy as np
import utils
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

features, features_list, labels = utils.load_and_preprocess_data()

#Logistic Regression model training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

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

isPCA = False
if isPCA:
    pca = PCA(n_components=34) # 34 total features
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_

logit_model=sm.Logit(y_train, X_train)
result=logit_model.fit()
print(result.summary())

#Logistic Regression model testing
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(X_test, y_test)))

#precision, recall, F-beta score
print(classification_report(y_test, y_pred))

"""
#Confusion matrix tells how many correct and incorrect predictions on training and testing data
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#1193 + 122 correct predictions
#43 + 16 incorrect predictions
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted No Migraines', 'Predicted Migraines'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual No Migraines', 'Actual Migraines'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')
plt.savefig('Confusion Matrix')


#ROC curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
"""