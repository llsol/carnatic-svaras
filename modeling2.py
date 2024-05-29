from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os

svara_data = pd.read_csv('svara_features/svara_features_normalized.csv')

# extract the features and target

# X = svara_data.iloc[:, :-7]
features = svara_data.iloc[:, :-7]

# y = svara_data.iloc[:, -7:]
target_columns = svara_data.iloc[:, -7:]

# combine boolean columns into a single categorical target column 
target = target_columns.idxmax(axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=37, stratify=target)
np.random.seed(37)

# Combine features and target for saving
train_svara_data = pd.concat([X_train, y_train], axis=1)
test_svara_data = pd.concat([X_test, y_test], axis=1)


# create a folder with the train and test svara_data
if not os.path.exists('svara_data'):
    os.makedirs('svara_data')

# Save to CSV
train_svara_data.to_csv('svara_data/train_svara_data.csv', index=False)
test_svara_data.to_csv('svara_data/test_svara_data.csv', index=False)

# Load the training svara_data
train_svara_data = pd.read_csv('svara_data/train_svara_data.csv')

# Load the test svara_data
test_svara_data = pd.read_csv('svara_data/test_svara_data.csv')

# Separate features and target in training svara_data
X_train = train_svara_data.iloc[:, :-1]
y_train = train_svara_data.iloc[:, -1]

# Separate features and target in test svara_data
X_test = test_svara_data.iloc[:, :-1]
y_test = test_svara_data.iloc[:, -1]

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training svara_data
X_train = imputer.fit_transform(X_train)

# Transform the test svara_data
X_test = imputer.transform(X_test)

# Initialize the HistGradientBoostingClassifier
reg = GradientBoostingClassifier(random_state=37)

# Fit the model to the training svara_data
reg.fit(X_train, y_train)

# Predict the clusters for the test svara_data
y_pred = reg.predict(X_test)

y_test = y_test.astype(str)
y_pred = y_pred.astype(str)

cm = confusion_matrix(y_test, y_pred)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", cm)

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))