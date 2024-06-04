import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load the feature differences CSV file
df_feature_diffs = pd.read_csv('svara_transition/feature_differences_with_labels.csv')

# Ensure all columns except 'label' are numeric
for column in df_feature_diffs.columns:
    if column != 'label':
        df_feature_diffs[column] = pd.to_numeric(df_feature_diffs[column], errors='coerce')

# Separate the data into transition and non-transition
transitions = df_feature_diffs[df_feature_diffs['label'] == 'transition']
non_transitions = df_feature_diffs[df_feature_diffs['label'] == 'non_transition']

# Initialize a dictionary to store the results of statistical tests
stats_results = {}

# Perform statistical tests and collect results
for column in df_feature_diffs.columns:
    if column != 'label':
        t_stat, p_value = ttest_ind(transitions[column].dropna(), non_transitions[column].dropna(), equal_var=False)
        stats_results[column] = {'t_stat': t_stat, 'p_value': p_value}

# Convert stats results to a DataFrame for easier viewing
df_stats_results = pd.DataFrame(stats_results).T

# Display statistical test results
print("Statistical Test Results (t-test):")
print(df_stats_results)

# Plotting feature differences
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_feature_diffs.columns):
    if column != 'label':
        plt.subplot(4, 4, i + 1)
        sns.boxplot(x='label', y=column, data=df_feature_diffs)
        plt.title(f'{column} Differences')
        plt.xlabel('')
        plt.ylabel('Difference')
plt.tight_layout()
plt.show()

# Visualizing feature distributions
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_feature_diffs.columns):
    if column != 'label':
        plt.subplot(4, 4, i + 1)
        sns.histplot(transitions[column].dropna(), color='blue', label='Transition', kde=True, stat='density', bins=30)
        sns.histplot(non_transitions[column].dropna(), color='red', label='Non-transition', kde=True, stat='density', bins=30)
        plt.title(f'{column} Distribution')
        plt.xlabel('Difference')
        plt.ylabel('Density')
        plt.legend()
plt.tight_layout()
plt.show()
