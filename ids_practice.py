import pandas as pd
from time import time
from numpy import array
from utils.codes import *

kdd_data_10percent = pd.read_csv("KDD10.txt", header=None, names=names)

features = kdd_data_10percent[num_features].astype(float)
print(features.describe())

labels = kdd_data_10percent['label'].copy()
labels[labels!='normal.'] = 'attack.'

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
features = pd.DataFrame(minmax.fit_transform(features.values), columns=num_features)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree', leaf_size=500)
# clf = GaussianNB()
t0 = time()
clf.fit(features,labels)
tt = time()-t0
print("Classifier trained in {} seconds".format(round(tt,3)))

kdd_data_corrected = pd.read_csv("corrected.txt", header=None, names=names)
print(kdd_data_corrected['label'].value_counts())
print(kdd_data_corrected.describe())

# kdd_data_corrected['label'][kdd_data_corrected['label']!='normal.'] = 'attack.'
kdd_data_corrected.loc[(kdd_data_corrected['label'] != 'normal.'), 'label'] = 'attack.'
print(kdd_data_corrected['label'].value_counts())

from sklearn.model_selection import train_test_split
kdd_data_corrected[num_features] = kdd_data_corrected[num_features].astype(float)
kdd_data_corrected[num_features] = pd.DataFrame(minmax.fit_transform(kdd_data_corrected[num_features].values), columns=num_features)
print(kdd_data_corrected)
# kdd_data_corrected[num_features].apply(lambda x: MinMaxScaler().fit_transform(x))

features_train, features_test, labels_train, labels_test = train_test_split(
    kdd_data_corrected[num_features], 
    kdd_data_corrected['label'], 
    test_size=0.1, 
    random_state=42
    )

pred = clf.predict(features_test)
print(pred)

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
acc = accuracy_score(pred, labels_test)
cn_mt = confusion_matrix(pred.astype(str), labels_test.astype(str))
cl_rp = classification_report(pred, labels_test)
print(cn_mt)
print(cl_rp)
print(acc)

