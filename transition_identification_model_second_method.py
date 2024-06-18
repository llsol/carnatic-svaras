from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import os

# Load the data
transition2_data = pd.read_csv('svara_transition/svara_transitions_method_2.csv')

# Print the initial columns of the data
print("Initial columns in transition2_data:", transition2_data.columns)

# Rename the 'label' column to 'is_transition'
transition2_data.rename(columns={'label': 'is_transition'}, inplace=True)

# Create a column in the transition2_data for the target, opposite to the last existing boolean column
transition2_data['is_not_transition'] = transition2_data['is_transition'].apply(lambda x: 1 if x == 0 else 0)

# Print the columns after adding is_not_transition
print("Columns after adding is_not_transition:", transition2_data.columns)

# Extract the features and target
features = transition2_data.iloc[:, :-2]  # All columns except the last two
target_columns = transition2_data.iloc[:, -2:]  # The last two columns

# Combine boolean columns into a single categorical target column
target = target_columns.idxmax(axis=1)

# Print the features and target columns
print("Features columns:", features.columns)
print("Target columns:", target_columns.columns)

# Combine features and target into one DataFrame for easy manipulation
data = pd.concat([features, target.rename('target')], axis=1)

# Print the combined data columns
print("Combined data columns:", data.columns)

# Separate the samples with is_transition=1 and is_not_transition=1
transition_samples = data[data['target'] == 'is_transition']
non_transition_samples = data[data['target'] == 'is_not_transition']

# Print the number of transition and non-transition samples
print("Number of transition samples:", len(transition_samples))
print("Number of non-transition samples:", len(non_transition_samples))

# Select a random sample of non_transition_samples with the same size as transition_samples
non_transition_sample = non_transition_samples.sample(n=len(transition_samples), random_state=37)

# Combine the transition samples and the selected non-transition samples
balanced_data = pd.concat([transition_samples, non_transition_sample])

# Separate features and target in the balanced data
features_balanced = balanced_data.iloc[:, :-1]
target_balanced = balanced_data.iloc[:, -1]

# Ensure all feature data is numeric
features_balanced = features_balanced.apply(pd.to_numeric, errors='coerce')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_balanced, target_balanced, test_size=0.2, random_state=37, stratify=target_balanced)
np.random.seed(37)

# Combine features and target for saving
train_transition2_data = pd.concat([X_train, y_train], axis=1)
test_transition2_data = pd.concat([X_test, y_test], axis=1)

# Create a folder to save the train and test data
if not os.path.exists('transition_data'):
    os.makedirs('transition_data')

# Save to CSV
train_transition2_data.to_csv('transition_data/train_transition2_data.csv', index=False)
test_transition2_data.to_csv('transition_data/test_transition2_data.csv', index=False)

# Load the training data
train_transition2_data = pd.read_csv('transition_data/train_transition2_data.csv')

# Load the test data
test_transition2_data = pd.read_csv('transition_data/test_transition2_data.csv')

# Separate features and target in training data
X_train = train_transition2_data.iloc[:, :-1]
y_train = train_transition2_data.iloc[:, -1]

# Separate features and target in test data
X_test = test_transition2_data.iloc[:, :-1]
y_test = test_transition2_data.iloc[:, -1]

# Ensure all feature data is numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train = imputer.fit_transform(X_train)

# Transform the test data
X_test = imputer.transform(X_test)

# Initialize the GradientBoostingClassifier
classifier = GradientBoostingClassifier(random_state=37)

# Fit the model to the training data
classifier.fit(X_train, y_train)

# Predict the target for the test data
y_pred = classifier.predict(X_test)

# Convert target and prediction to string to avoid classification report errors
y_test = y_test.astype(str)
y_pred = y_pred.astype(str)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", cm)

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
