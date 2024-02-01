import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(battery, min_rows_per_cycle):
    mat = loadmat('battery_data/' + battery + '.mat')
    print('Total data in dataset: ', len(mat[battery][0, 0]['cycle'][0]))
    counter = 0
    dataset = []
    capacity_data = []

    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                   int(row['time'][0][1]),
                                   int(row['time'][0][2]),
                                   int(row['time'][0][3]),
                                   int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]

            # Find the minimum number of rows for this cycle
            min_rows = min(min_rows_per_cycle, len(data[0][0]['Voltage_measured'][0]))

            for j in range(min_rows):
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]
                dataset.append([counter + 1, ambient_temperature, capacity,
                                voltage_measured, current_measured,
                                temperature_measured, current_load,
                                voltage_load, time])
            capacity_data.append([counter + 1, ambient_temperature, capacity])
            counter = counter + 1

    dataset_df = pd.DataFrame(data=dataset,
                               columns=['cycle', 'ambient_temperature', 'capacity',
                                        'voltage_measured', 'current_measured',
                                        'temperature_measured', 'current_load',
                                        'voltage_load', 'time'])

    # Save the dataset DataFrame to Excel (excluding 'ambient_temperature' and 'current_load' columns)
    dataset_excel_path = 'dataset_' + battery + '.xlsx'
    dataset_df[['cycle', 'capacity', 'voltage_measured',
                'current_measured', 'temperature_measured', 'voltage_load', 'time']].to_excel(dataset_excel_path, index=False)
    print(f'Dataset saved to {dataset_excel_path}')

    # Calculate average capacity for each cycle
    avg_capacity_per_cycle = dataset_df.groupby('cycle')['capacity'].mean().reset_index()

    # Plot average capacity vs cycle
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='cycle', y='capacity', data=avg_capacity_per_cycle, marker='o')
    plt.title('Average Capacity vs Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Average Capacity')
    plt.grid(True)
    plt.show()

    # Print the number of rows for each cycle in the dataset
    cycle_counts = dataset_df['cycle'].value_counts().sort_index()
    print("\nNumber of rows for each cycle:")
    print(cycle_counts)

    # Find the minimum number of rows and corresponding cycle
    min_rows_cycle = cycle_counts.idxmin()
    min_rows_count = cycle_counts.min()
    print(f"\nMinimum number of rows for a cycle: {min_rows_count} (Cycle {min_rows_cycle})")

    return [dataset_df, pd.DataFrame(data=capacity_data,
                                      columns=['cycle', 'ambient_temperature', 'capacity'])]

# Specify the minimum number of rows per cycle you want to consider
min_rows_per_cycle = 179

dataset, capacity = load_data('B0005', min_rows_per_cycle)
print("Shape of dataset_df:", dataset.shape)
