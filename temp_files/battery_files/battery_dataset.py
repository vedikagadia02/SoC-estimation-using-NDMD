import scipy.io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def helperMovingAverage(df):
    temp_df = df.iloc[[0,1,2,-1], ::100]
    new_df = temp_df.transpose()
    new_df.columns = ["Voltage", "Current", "Temperature", "SOC"]

    # Calculate moving averages
    new_df['AverageVoltage'] = new_df['Voltage'].rolling(window=6, min_periods=1).mean()
    new_df['AverageCurrent'] = new_df['Current'].rolling(window=6, min_periods=1).mean()

    # Reorder columns
    new_df = new_df[["Voltage", "Current", "AverageVoltage", "AverageCurrent", "Temperature", "SOC"]]

    return new_df

mat = scipy.io.loadmat('./battery_dataset_10to25C/Train/train_10to25C.mat')
data = mat['X']
trainDataFull = pd.DataFrame(data)
trainDataFull.to_csv('battery_dataset_matlab.csv', index=False)
print(trainDataFull)
print(trainDataFull.shape)

idx0 = list(range(1, 184258))
idx10 = list(range(184258, 337974))
idx25 = list(range(337974, 510531))
idxN10 = list(range(510531, 669956))

trainData0deg = helperMovingAverage(trainDataFull.iloc[:, idx0])
trainData10deg = helperMovingAverage(trainDataFull.iloc[:, idx10])
trainData25deg = helperMovingAverage(trainDataFull.iloc[:, idx25])
trainDataN10deg = helperMovingAverage(trainDataFull.iloc[:, idxN10])

trainData = pd.concat([trainData0deg, trainData10deg, trainData25deg, trainDataN10deg])
print(trainData)
print(trainData.shape)

dataframes = {'0deg': trainData0deg, '10deg': trainData10deg, '25deg': trainData25deg, 'N10deg': trainDataN10deg}
columns = ['Voltage', 'Current', 'Temperature', 'SOC']
sns.set_style("whitegrid")
save_dir = 'dataset_visualisation'
for key, df in dataframes.items():
    df_dir = os.path.join(save_dir, key)
    os.makedirs(df_dir, exist_ok=True)
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i == j:
                continue
            plt.figure(figsize=(10, 8))
            sns.lineplot(x=df[columns[i]], y=df[columns[j]])
            plt.xlabel(columns[i])
            plt.ylabel(columns[j])
            plt.savefig(os.path.join(df_dir, f'{columns[i]}_vs_{columns[j]}.png'))
            plt.close()
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=df.index, y=df[columns[i]])
        plt.xlabel('Index')
        plt.ylabel(columns[i])
        plt.savefig(os.path.join(df_dir, f'{columns[i]}_over_index.png'))
        plt.close()
