import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_ind

# Load the normalized data
df = pd.read_csv('svara_transition/svara_transitions_normalized.csv')

# Descriptive statistics
print(df.groupby('is_transition').describe())

# Correlation with the transition label
correlations = df.corr()['is_transition'].sort_values(ascending=False)
print(correlations)

# Visualizations
# Histograms and box plots
features = df.columns.drop(['id', 'is_transition'])

for feature in features:
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=feature, hue='is_transition', element='step', stat='density', common_norm=False)
    plt.title(f'Histogram of {feature}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='is_transition', y=feature)
    plt.title(f'Boxplot of {feature}')
    
    plt.show()

# Train a Random Forest Classifier to get feature importances
X = df[features]
y = df['is_transition']

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. {features[indices[f]]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Permutation feature importance
perm_importance = permutation_importance(clf, X, y, n_repeats=10, random_state=42)

sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(12, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importances (train set)")
plt.tight_layout()
plt.show()

# Statistical tests
for feature in features:
    transition_values = df[df['is_transition'] == 1][feature]
    non_transition_values = df[df['is_transition'] == 0][feature]
    t_stat, p_value = ttest_ind(transition_values, non_transition_values)
    print(f"{feature}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
