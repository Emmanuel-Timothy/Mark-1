import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the CSV file
data = pd.read_csv('data.csv')

# Step 1: Map 'Lateness_Frequency' to numeric values
lateness_mapping = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4
}
data['Lateness_Frequency'] = data['Lateness_Frequency'].map(lateness_mapping)

# Step 2: Group by 'External_Factors' and calculate the average lateness
x_factor_analysis = data.groupby('External_Factors')['Lateness_Frequency'].mean().sort_values()

# Step 3: Create a bar chart
plt.figure(figsize=(12, 8))
x_factor_analysis.plot(kind='bar', color='skyblue')
plt.title('Average Lateness Frequency by External Factors')
plt.xlabel('External Factors')
plt.ylabel('Average Lateness Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Step 4: Correlation with One-Hot Encoded External Factors
external_dummies = pd.get_dummies(data['External_Factors'])

# Combine with lateness frequency column
correlation_data = pd.concat([data['Lateness_Frequency'], external_dummies], axis=1)

# Compute correlations
correlation_matrix = correlation_data.corr()
lateness_correlations = correlation_matrix['Lateness_Frequency'].drop('Lateness_Frequency')

print("\n📊 Correlation between External Factors and Lateness Frequency:")
print(lateness_correlations.sort_values(ascending=False))

# Step 5: ANOVA Test - is there a statistically significant difference between groups?
groups = [group['Lateness_Frequency'].values for name, group in data.groupby('External_Factors')]
f_stat, p_value = stats.f_oneway(*groups)

print("\n🔬 ANOVA Results:")
print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ The differences between groups are statistically significant.")
else:
    print("❌ No significant difference found between groups.")
