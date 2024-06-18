from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os

transition_data = pd.read_csv('svara_transition/transition_features_two_sides.csv')

# create is_not transition column with the opposite of is_transition
transition_data['is_not_transition'] = transition_data['is_transition'].apply(lambda x: 1 if x == 0 else 0)



# extract the features and target

# X = transition_data.iloc[:, :-2]
features = transition_data.iloc[:, :-2]

# y = transition_data.iloc[:, -2:]
target_columns = transition_data.iloc[:, -2:]

# combine boolean columns into a single categorical target column 
target = target_columns.idxmax(axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=37, stratify=target)
np.random.seed(37)

# Combine features and target for saving
train_transition_data = pd.concat([X_train, y_train], axis=1)
test_transition_data = pd.concat([X_test, y_test], axis=1)


# create a folder with the train and test transition_data
if not os.path.exists('transition_data'):
    os.makedirs('transition_data')

# Save to CSV
train_transition_data.to_csv('transition_data/train_transition_data.csv', index=False)
test_transition_data.to_csv('transition_data/test_transition_data.csv', index=False)

# Load the training transition_data
train_transition_data = pd.read_csv('transition_data/train_transition_data.csv')

# Load the test transition_data
test_transition_data = pd.read_csv('transition_data/test_transition_data.csv')

# Separate features and target in training transition_data
X_train = train_transition_data.iloc[:, :-1]
y_train = train_transition_data.iloc[:, -1]

# Separate features and target in test transition_data
X_test = test_transition_data.iloc[:, :-1]
y_test = test_transition_data.iloc[:, -1]

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training transition_data
X_train = imputer.fit_transform(X_train)

# Transform the test transition_data
X_test = imputer.transform(X_test)

# Initialize the HistGradientBoostingClassifier
reg = GradientBoostingClassifier(random_state=37)

# Fit the model to the training transition_data
reg.fit(X_train, y_train)

# Predict the clusters for the test transition_data
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