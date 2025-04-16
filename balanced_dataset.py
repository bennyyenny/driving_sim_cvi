import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('driving_log.csv', header=None)

# Assign column names
data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

# Plot the original steering angle distribution
plt.figure(figsize=(10, 5))
plt.hist(data['steering'], bins=50, color='skyblue', edgecolor='black')
plt.title('Original Steering Angle Distribution')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()