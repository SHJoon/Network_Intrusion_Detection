import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# Change the following variables as desired.
# 'k_value' is any positive integer
k_value = 5

# Assign column names to the dataset
names = ['duration', 'protocol_type', 'service',\
    'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',\
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',\
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',\
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',\
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',\
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',\
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',\
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',\
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',' dst_host_srv_rerror_rate', 'class']

# Read dataset to pandas dataframe
train_dataset = pd.read_csv("KDDTrain+20Percent.txt", names=names)

# Label encoder transforms the dataframe into integers accordinglyp
label_encoder = preprocessing.LabelEncoder()
transform_train = train_dataset.iloc[:, :-2].apply(label_encoder.fit_transform)

X_train_fit = transform_train.iloc[:, :-2].values
y_train_fit = train_dataset.iloc[:, 41].values

test_dataset = pd.read_csv("KDDTest.txt", names=names)

transform_test = test_dataset.iloc[:, :-2].apply(label_encoder.fit_transform)

X_test_fit = transform_test.iloc[:, :-2].values
y_test_fit = test_dataset.iloc[:, 41].values

# Normalize the data for more accurate results
scaler = preprocessing.StandardScaler()
scaler.fit(X_train_fit)
X_train = scaler.transform(X_train_fit)
X_test = scaler.transform(X_test_fit)

# Apply the k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=k_value)
classifier.fit(X_train, y_train_fit)

y_pred = classifier.predict(X_test)
print(y_pred)   

# Print out the results
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_fit.astype(str), y_pred.astype(str)))
print(classification_report(y_test_fit, y_pred))

# Calculating error for K values between 1 and 40
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train_fit)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test_fit))
    print(error[i-1])

# Plot the Error Rate per k value graph
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
