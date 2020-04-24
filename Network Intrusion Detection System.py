import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.codes import names
from sklearn import preprocessing

# Change the following variables as desired.
# 'k_value' is any positive integer
k_value = 5

# Read dataset to pandas dataframe
train_dataset = pd.read_csv("KDD10.txt", names=names)

# Label encoder transforms the dataframe into integers accordinglyp
label_encoder = preprocessing.LabelEncoder()
transform_train = train_dataset.iloc[:, :-1].apply(label_encoder.fit_transform)
train_dataset.loc[(train_dataset['label'] != 'normal.'), 'label'] = 'anomaly.'

X_train_fit = transform_train.values
y_train_fit = train_dataset.iloc[:, 41].values

test_dataset = pd.read_csv("corrected.txt", names=names)

transform_test = test_dataset.iloc[:, :-1].apply(label_encoder.fit_transform)
test_dataset.loc[(test_dataset['label'] != 'normal.'), 'label'] = 'anomaly.'

X_test_fit = transform_test.values
y_test_fit = test_dataset.iloc[:, 41].values

# Normalize the data for more accurate results
minmax = preprocessing.MinMaxScaler()
minmax.fit(X_train_fit)
X_train = minmax.transform(X_train_fit)
X_test = minmax.transform(X_test_fit)

from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
bayes.fit(X_train_fit, y_train_fit)

y_pred = bayes.predict(X_test_fit)

# Print out the results
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test_fit.astype(str), y_pred.astype(str)))
print(classification_report(y_test_fit, y_pred))
b_acc = accuracy_score(y_pred, y_test_fit)
print(f"Bayes Classifier Accuracy: {b_acc}")

# Apply the k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=k_value)
classifier.fit(X_train_fit, y_train_fit)

y_pred = classifier.predict(X_test_fit)

# Print out the results
print(confusion_matrix(y_test_fit.astype(str), y_pred.astype(str)))
print(classification_report(y_test_fit, y_pred))
k_acc = accuracy_score(y_pred, y_test_fit)
print(f"K Nearest Neighbor Classifier Accuracy: {k_acc}")

# Calculating error for K values between 1 and 40
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_fit, y_train_fit)
    pred_i = knn.predict(X_test_fit)
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
