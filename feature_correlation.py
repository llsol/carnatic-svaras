import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the CSV file into a DataFrame
csv_file_path = 'svara_features/svara_features_normalized.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Exclude the first column (element IDs) from the features
features_df = df.iloc[:, 1:25]

# Compute the Spearman rank correlation matrix
spearman_corr = features_df.corr(method='spearman')

# Round the correlation matrix to two decimals
spearman_corr = spearman_corr.round(2)

# Print the correlation matrix
print("Spearman Rank Correlation Matrix:")
print(spearman_corr)

# Optionally, visualize the correlation matrix using a heatmap
plt.figure(figsize=(100, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", 
            xticklabels=spearman_corr.columns, yticklabels=spearman_corr.columns)
plt.xticks(rotation=90)
plt.title('Spearman Rank Correlation Matrix')
plt.show()