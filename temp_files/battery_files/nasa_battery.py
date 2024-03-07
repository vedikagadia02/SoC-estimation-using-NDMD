import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = np.load("../soc_implementation/dataset/soc_dataset_7.npy")
new_col = np.full((len(data[0]),1), 1)                      # size [179,1]
temp_dataset = data[0]                                      # size [179,6]
print(temp_dataset.shape)
new_temp = np.concatenate((new_col, temp_dataset), axis=1)  # size [179,7]
for i, cycle in enumerate(data[1:]):
    new_col = np.full((len(data[0]),1), i+1)
    new_cycle = np.concatenate((new_col, cycle), axis=1)
    nt = np.concatenate([new_temp,new_cycle])
    new_temp = nt

# data shape [30072,7]
dataset = pd.DataFrame(data=new_temp, columns = ['cycle', 'capacity', 'voltage_measured', 'current_measured', 'temperature_measured', 'voltage_load', 'current_load', 'time'])
print(dataset)

unique_temp_values = dataset['temperature_measured'].nunique()
print(f'The unique values in the temperature_measured column are: {unique_temp_values}')

C = dataset['capacity'][0]
soh = []
for i in range(len(dataset)):
  soh.append([dataset['capacity'][i] / C])
soh = pd.DataFrame(data=soh, columns=['SoH'])
attribs=['capacity', 'voltage_measured', 'current_measured',
         'temperature_measured', 'current_load', 'voltage_load', 'time']
train_dataset = dataset[attribs]
sc = MinMaxScaler(feature_range=(0,1))
train_dataset = sc.fit_transform(train_dataset)
print(train_dataset.shape)
print(soh.shape)

dataset_cycle1 = dataset[dataset['cycle'] == 166]
fig, axs = plt.subplots(3, figsize=(10,15))

axs[0].scatter(dataset_cycle1['time'], dataset_cycle1['current_measured'])
axs[0].set_title('Current vs Time for Cycle 1')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Current Measured')
axs[0].set_yscale('log')

axs[1].scatter(dataset_cycle1['time'], dataset_cycle1['voltage_measured'])
axs[1].set_title('Voltage vs Time for Cycle 1')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Voltage Measured')
axs[2].scatter(dataset_cycle1['time'], dataset_cycle1['temperature_measured'])
axs[2].set_title('Temperature vs Time for Cycle 1')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Temperature Measured')

# Display the plot
plt.tight_layout()

# Save the plot
plt.savefig('scatterplot_cycle1.png')

plt.show()

