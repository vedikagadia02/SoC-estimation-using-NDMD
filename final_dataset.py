import pandas as pd
import numpy as np

# Load the dataset from Excel
dataset_path = 'dataset_B0005.xlsx'
df = pd.read_excel(dataset_path)

# Exclude the 'cycle' column
df = df.drop(columns=['cycle'])

# Reshape the data to have dimensions 168 x 179 x 6
reshaped_data = df.values.reshape((168, 179, -1))

# Save the reshaped data as a NumPy array
np.save('reshaped_dataset.npy', reshaped_data)

# Print the shape of the reshaped dataset
print("Shape of the reshaped dataset:", reshaped_data.shape)
