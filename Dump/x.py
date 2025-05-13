import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data.csv')

# Map 'Lateness_Frequency' to numeric values
data['Lateness_Frequency'] = data['Lateness_Frequency'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
})

# Group by 'External_Factors' and calculate the average Lateness_Frequency
x_factor_analysis = data.groupby('External_Factors')['Lateness_Frequency'].mean().sort_values()

# Create a bar chart
plt.figure(figsize=(12, 8))
x_factor_analysis.plot(kind='bar', color='skyblue')
plt.title('Average Lateness Frequency by External Factors')
plt.xlabel('External Factors')
plt.ylabel('Average Lateness Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()