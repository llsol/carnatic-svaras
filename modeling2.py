from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import os

data = pd.read_csv('features/features_normalized.csv')

# extract the features and target

# X = data.iloc[:, :-7]
features = data.iloc[:, :-7]

# y = data.iloc[:, -7:]
target_columns = data.iloc[:, -7:]

# combine boolean columns into a single categorical target column 
target = target_columns.idxmax(axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=37, stratify=target)
np.random.seed(37)

# Combine features and target for saving
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


# create a folder with the train and test data
if not os.path.exists('data'):
    os.makedirs('data')

# Save to CSV
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# Load the training data
train_data = pd.read_csv('data/train_data.csv')

# Load the test data
test_data = pd.read_csv('data/test_data.csv')

# Separate features and target in training data
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Separate features and target in test data
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train = imputer.fit_transform(X_train)

# Transform the test data
X_test = imputer.transform(X_test)

# Initialize the HistGradientBoostingClassifier
reg = GradientBoostingClassifier(random_state=37)

# Fit the model to the training data
reg.fit(X_train, y_train)

# Predict the clusters for the test data
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