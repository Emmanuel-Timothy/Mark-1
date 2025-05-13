import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv('data.csv')

# Map 'Lateness_Frequency' to numeric values
data['Lateness_Frequency'] = data['Lateness_Frequency'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
})

# List of reasons to analyze
reasons = ['Reason_Lazyness', 'Reason_Forgetfulness', 'Reason_Extracurriculars', 'Reason_TightDeadlines']

# Calculate the correlation between each reason and Lateness_Frequency
correlations = {}
for reason in reasons:
    correlations[reason] = data[reason].astype(int).corr(data['Lateness_Frequency'])

# Convert the correlations dictionary to a DataFrame for visualization
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f")
plt.title('Correlation Between Reasons and Lateness Frequency')
plt.xlabel('Correlation')
plt.ylabel('Reasons')
plt.tight_layout()

# Show the plot
plt.show()