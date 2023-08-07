from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from os import environ
environ["QT_DEVICE_PIXEL_RATIO"] = "0"
environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
environ["QT_SCREEN_SCALE_FACTORS"] = "1"
environ["QT_SCALE_FACTOR"] = "1"

bs_all = [
    'B0005',
    'B0006',
    'B0007',
    'B0018',
    'B0025',
    'B0026',
    'B0027',
    'B0028',
    'B0029',
    'B0030',
    'B0031',
    'B0032',
    'B0033',
    'B0034',
    'B0036',
    'B0038',
    'B0039',
    'B0040',
    'B0041',
    'B0042',
    'B0043',
    'B0044',
    'B0045',
    'B0046',
    'B0047',
    'B0048',
    'B0049',
    'B0050',
    'B0051',
    'B0052',
    'B0053',
    'B0054',
    'B0055',
    'B0056',
]

bs = [
    'B0005',
    'B0006',
    'B0007',
    'B0018'
]

ds = []
for b in bs:
    ds.append(loadmat(f'DATA/{b}.mat'))

types = []
times = []
ambient_temperatures = []
datas = []

for i in range(len(ds)):
    x = ds[i][bs[i]]["cycle"][0][0][0]
    ambient_temperatures.append(x['ambient_temperature'])
    types.append(x['type'])
    times.append(x['time'])
    datas.append(x['data'])

for i in range(len(ds)):
    print(f'Battery: {bs[i]}')
    print(f'Cycles: {datas[i].size}')
    print()

# CHARGE ALL CYCLES
params = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_charge', 'Voltage_charge']

for p in params:
    fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
    param = p
    for i in range(len(bs)):
        for j in range(datas[i].size):
            if types[i][j] == 'charge':

                if i % 2 == 0:
                    axs[i // 2, 0].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0])
                    axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
                else:
                    axs[i // 2, 1].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0])
                    axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
    for ax in axs.flat:
        ax.set(ylabel = param, xlabel = 'Time')
    fig.tight_layout(pad = 0.3)
    fig.savefig(f'PLOTS/{p}_all.png')

# CHARGE FIRST AND LAST CYCLES
for p in params:
    
    # Printing first cycles
    
    fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
    param = p
    for i in range(len(bs)):
        for j in range(20):
            if types[i][j] == 'charge':
                if i % 2 == 0:
                    axs[i // 2, 0].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 0].legend()
                else:
                    axs[i // 2, 1].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 1].legend()
    for ax in axs.flat:
        ax.set(ylabel = param, xlabel = 'Time')
    fig.tight_layout(pad = 0.3)
    fig.savefig(f'PLOTS/{p}_first.png')
    # Printing last cycles

    fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
    for i in range(len(bs)):
        for j in range(datas[i].size - 20, datas[i].size):
            if types[i][j] == 'charge':
                if i % 2 == 0:
                    axs[i // 2, 0].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 0].legend()
                else:
                    axs[i // 2, 1].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 1].legend()

    for ax in axs.flat:
        ax.set(ylabel = param, xlabel = 'Time')
    fig.tight_layout(pad = 0.3)
    fig.savefig(f'PLOTS/{p}_last.png')

# DISCHARGE ALL CYCLES
params = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']

for param in params:
    fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
    for i in range(len(bs)):
        for j in range(datas[i].size):
            if types[i][j] == 'discharge':
                if i % 2 == 0:
                    axs[i // 2, 0].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], 'x')
                    axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
                else:
                    axs[i // 2, 1].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0])
                    axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
    for ax in axs.flat:
        ax.set(ylabel = param, xlabel = 'Time')
    fig.tight_layout(pad = 0.3)
    fig.savefig(f'PLOTS/{param}_all.png')
    
fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
for i in range(len(bs)):
    cap = []
    cycle = []
    for j in range(datas[i].size):
        if types[i][j] == 'discharge':
            cap.append(datas[i][j]['Capacity'][0][0][0][0])
            cycle.append(j)
    if i % 2 == 0:
        axs[i // 2, 0].plot(cycle, cap)
        axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
    else:
        axs[i // 2, 1].plot(cycle, cap)
        axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
        
    for ax in axs.flat:
        ax.set(ylabel = 'Capacity', xlabel = 'Cycles')
fig.tight_layout(pad = 0.3)
fig.savefig(f'PLOTS/Capacity_Line.png')

# DISCHARGE FIRST AND LAST CYCLES
for p in params:
    
    # Printing first cycles
    
    fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
    param = p
    for i in range(len(bs)):
        for j in range(20):
            if types[i][j] == 'discharge':
                if i % 2 == 0:
                    axs[i // 2, 0].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 0].legend()
                else:
                    axs[i // 2, 1].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 1].legend()
    for ax in axs.flat:
        ax.set(ylabel = param, xlabel = 'Time')
    fig.tight_layout(pad = 0.3)
    fig.savefig(f'PLOTS/{p}_first.png')
    
    # Printing last cycles

    fig, axs = plt.subplots((len(bs) + 1) // 2, 2)
    for i in range(len(bs)):
        for j in range(datas[i].size - 20, datas[i].size):
            if types[i][j] == 'discharge':
                if i % 2 == 0:
                    axs[i // 2, 0].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 0].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 0].legend()
                else:
                    axs[i // 2, 1].plot(datas[i][j]['Time'][0][0][0], datas[i][j][param][0][0][0], label = f'{j + 1}')
                    axs[i // 2, 1].set_title(f'Battery: {bs[i]}')
                    axs[i // 2, 1].legend()

    for ax in axs.flat:
        ax.set(ylabel = param, xlabel = 'Time')
    fig.tight_layout(pad = 0.3)
    fig.savefig(f'PLOTS/{p}_last.png')

# REGRESSION
from pprint import pprint

Cycles = {}
params = ['Temperature_measured', 'Voltage_measured', 'Voltage_load', 'Time']

for i in range(len(bs)):
    Cycles[bs[i]] = {}
    Cycles[bs[i]]['count'] = 168 # This is true for battery B0005, 06, 07
    for param in params:
        Cycles[bs[i]][param] = []
        for j in range(datas[i].size):
            if types[i][j] == 'discharge':
                Cycles[bs[i]][param].append(datas[i][j][param][0][0][0])
        
    cap = []
    for j in range(datas[i].size):
        if types[i][j] == 'discharge':
            cap.append(datas[i][j]['Capacity'][0][0][0][0])
    Cycles[bs[i]]['Capacity'] = np.array(cap)

# FEATURE EXTRACTION
## CRITICAL TIME POINTS FOR A CYCLE
## We will only these critical points for furthur training

## TEMPERATURE_MEASURED
## => Time at highest temperature

## VOLTAGE_MEASURED
## => Time at lowest Voltage

## VOLTAGE_LOAD
## => First time it drops below 1 volt after 1500 time


def getTemperatureMeasuredCritical(tm, time):
    high = 0
    critical = 0
    for i in range(len(tm)):
        if (tm[i] > high):
            high = tm[i]
            critical = time[i]
    return critical

def getVoltageMeasuredCritical(vm, time):
    low = 1e9
    critical = 0
    for i in range(len(vm)):
        if (vm[i] < low):
            low = vm[i]
            critical = time[i]
    return critical

def getVoltageLoadCritical(vl, time):
    for i in range(len(vl)):
        if (time[i] > 1500 and vl[i] < 1):
            return time[i]
    return -1

# First Cycle
f = getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][0], Cycles[bs[0]]['Time'][0])

# 100th Cycle
m = getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][100], Cycles[bs[0]]['Time'][100])

# Last Cycle
l = getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][167], Cycles[bs[0]]['Time'][167])

print(f'Temperature_Measured Critical points')
print(f'First Cycle:\t{f}')
print(f'100th Cycle:\t{m}')
print(f'Last Cycle:\t{l}')
print()

## Conclusion
## !!BATTERY GET HOT QUICKER as they AGE!!

# First Cycle
f = getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][0], Cycles[bs[0]]['Time'][0])

# 100th Cycle
m = getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][100], Cycles[bs[0]]['Time'][100])

# Last Cycle
l = getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][167], Cycles[bs[0]]['Time'][167])

print(f'Voltage_measured Critical points')
print(f'First Cycle:\t{f}')
print(f'100th Cycle:\t{m}')
print(f'Last Cycle:\t{l}')
print()

## Conclusion
## !!VOLTAGE HOLDS FOR LESS TIME as they AGE!!

# First Cycle
f = getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][0], Cycles[bs[0]]['Time'][0])

# 100th Cycle
m = getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][100], Cycles[bs[0]]['Time'][100])

# Last Cycle
l = getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][167], Cycles[bs[0]]['Time'][167])

print(f'Voltage_load Critical points')
print(f'First Cycle:\t{f}')
print(f'100th Cycle:\t{m}')
print(f'Last Cycle:\t{l}')
print()

## Conclusion
## !!VOLTAGE HOLDS FOR LESS TIME as they AGE!!

temperature_measured = []
voltage_measured = []
voltage_load = []
capacity = Cycles[bs[0]]['Capacity']

for i in range(Cycles[bs[0]]['count']):
    temperature_measured.append(getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][i], Cycles[bs[0]]['Time'][i]))
    voltage_measured.append(getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][i], Cycles[bs[0]]['Time'][i]))
    voltage_load.append(getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][i], Cycles[bs[0]]['Time'][i]))

## Plotting (Critical Points) v/s (Cycles)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(range(1, len(temperature_measured) + 1), temperature_measured)
axs[0, 0].set(ylabel = 'Critical Points for TM', xlabel = 'Cycle')

axs[0, 1].plot(range(1, len(voltage_measured) + 1), voltage_measured)
axs[0, 1].set(ylabel = 'Critical Points for VM', xlabel = 'Cycle')

axs[1, 0].plot(range(1, len(voltage_load) + 1), voltage_load)
axs[1, 0].set(ylabel = 'Critical Points for VL', xlabel = 'Cycle')

axs[1, 1].plot(range(1, len(voltage_measured) + 1), capacity)
axs[1, 1].set(ylabel = 'Capacity', xlabel = 'Cycle')

fig.tight_layout(pad = 0.3)
fig.savefig(f'PLOTS/Critical_Values.png')

# TRAINING REGRESSION MODEL
X = []
for i in range(Cycles[bs[0]]['count']):
    X.append(np.array([temperature_measured[i], voltage_measured[i], voltage_load[i]]))
X = np.array(X)
y = np.array(capacity)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

diff = 0
total = 0
for i in range(len(y_test)):
    diff += abs(y_test[i] - y_pred[i])
    total += y_test[i]
diff /= len(y_test)
total /= len(y_test)
accuracy = ((total - diff) / total) * 100
print(f'Average Difference Between Predicted and Real Capacities: {diff}')
print(f'Accuracy: {accuracy}')

# REGRESSION MODEL 2
bs = [
    'B0005',
    'B0006',
    'B0007',
]

temperature_measured = []
voltage_measured = []
voltage_load = []
capacity = []

for b in bs:
    for c in Cycles[b]['Capacity']:
        capacity.append(c)

for _ in range(len(bs)):
    for i in range(Cycles[bs[_]]['count']):
        temperature_measured.append(getTemperatureMeasuredCritical(Cycles[bs[_]]['Temperature_measured'][i], Cycles[bs[_]]['Time'][i]))
        voltage_measured.append(getVoltageMeasuredCritical(Cycles[bs[_]]['Voltage_measured'][i], Cycles[bs[_]]['Time'][i]))
        voltage_load.append(getVoltageLoadCritical(Cycles[bs[_]]['Voltage_load'][i], Cycles[bs[_]]['Time'][i]))

# TRAINING THE MODEL
X = []
for i in range(len(temperature_measured)):
    X.append(np.array([temperature_measured[i], voltage_measured[i], voltage_load[i]]))
X = np.array(X)
y = np.array(capacity)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
diff = 0
total = 0
for i in range(len(y_test)):
    diff += abs(y_test[i] - y_pred[i])
    total += y_test[i]
diff /= len(y_test)
total /= len(y_test)
accuracy = ((total - diff) / total) * 100
print()
print(f'Average Difference Between Predicted and Real Capacities: {diff}')
print(f'Accuracy: {accuracy}')