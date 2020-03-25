import pandas as pd
from time import time
from numpy import array

names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


kdd_data_10percent = pd.read_csv("KDD10.txt", header=None, names=names)
# print(kdd_data_10percent.describe())

# print(kdd_data_10percent['label'].value_counts())
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
features = kdd_data_10percent[num_features].astype(float)
# print(features.describe())

from sklearn.neighbors import KNeighborsClassifier
labels = kdd_data_10percent['label'].copy()
labels[labels!='normal.'] = 'attack.'
# print(labels.value_counts())

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
features = pd.DataFrame(minmax.fit_transform(features.values), columns=num_features)
# print(features)

clf = KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree', leaf_size=500)
t0 = time()
clf.fit(features,labels)
tt = time()-t0
print("Classifier trained in {} seconds".format(round(tt,3)))

kdd_data_corrected = pd.read_csv("corrected.txt", header=None, names = names)
print(kdd_data_corrected['label'].value_counts())

kdd_data_corrected['label'][kdd_data_corrected['label']!='normal.'] = 'attack.'
print(kdd_data_corrected['label'].value_counts())

from sklearn.model_selection import train_test_split
kdd_data_corrected[num_features] = kdd_data_corrected[num_features].astype(float)
kdd_data_corrected = pd.DataFrame(minmax.fit_transform(kdd_data_corrected[num_features].values), columns=num_features)
# kdd_data_corrected[num_features].apply(lambda x: MinMaxScaler().fit_transform(x))

features_train, features_test, labels_train, labels_test = train_test_split(
    kdd_data_corrected[num_features], 
    kdd_data_corrected['label'], 
    test_size=0.1, 
    random_state=42
    )

t0 = time()
pred = clf.predict(features_test)
tt = time() - t0
print("Predicted in {} seconds".format(round(tt,3)))

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print(acc)
print("R squared is {}.".format(round(acc,4)))