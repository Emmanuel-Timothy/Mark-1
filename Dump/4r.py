import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data.csv')

# Map 'Lateness_Frequency' to numeric values
data['Lateness_Frequency'] = data['Lateness_Frequency'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
})

# List of reasons to analyze
reasons = ['Reason_Lazyness', 'Reason_Forgetfulness', 'Reason_Extracurriculars', 'Reason_TightDeadlines']

# Calculate the average Lateness_Frequency for each reason
average_lateness = {}
for reason in reasons:
    average_lateness[reason] = data[data[reason] == True]['Lateness_Frequency'].mean()

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(average_lateness.keys(), average_lateness.values(), color='skyblue')
plt.title('Average Lateness Frequency by Reason')
plt.xlabel('Reasons')
plt.ylabel('Average Lateness Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()