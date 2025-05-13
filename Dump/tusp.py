import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data = pd.read_csv('data.csv')

# Convert 'Tutoring_Min' and 'Lateness_Frequency' to numeric, handling non-numeric values
data['Tutoring_Min'] = pd.to_numeric(data['Tutoring_Min'], errors='coerce')
data['Lateness_Frequency'] = data['Lateness_Frequency'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
})

# Drop rows with missing values in the relevant columns
data = data.dropna(subset=['Tutoring_Min', 'Lateness_Frequency'])

# Calculate the line of best fit
x = data['Tutoring_Min']
y = data['Lateness_Frequency']
m, b = np.polyfit(x, y, 1)  # Slope (m) and intercept (b)

# Calculate R²
y_pred = m * x + b
ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum((y - y_pred)**2)
r_squared = 1 - (ss_residual / ss_total)

# Create the scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, color='blue', label='Data points')
plt.plot(x, m * x + b, color='red', label='Best fit line') 
plt.title('Scatterplot of Class Minutes vs Lateness Frequency')
plt.xlabel('Class Minutes')
plt.ylabel('Lateness Frequency')
plt.grid(True)
plt.legend()

# Annotate R² on the plot
plt.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()