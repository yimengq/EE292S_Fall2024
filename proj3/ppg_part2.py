import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy import blackman
from scipy.signal import butter, filtfilt

data = np.load('ppg_2.npz', allow_pickle=True)['arr_0'].item()['adc'][100:]
# data = data - np.mean(data)

fs = 400 # maybe we need to change this
peaks, _ = find_peaks(data, height=0.0, distance=fs/2)  
time_diff = np.diff(peaks) / fs
T = np.mean(time_diff)
bpm = 60 / T
max_hrv = np.max(time_diff) - np.min(time_diff)  
rms_hrv = np.sqrt(np.mean((time_diff - np.mean(time_diff))**2))

print("BPM: ", bpm)
print("Max HRV: ", max_hrv)
print("rms_hrv: ", rms_hrv)

plt.figure()
plt.plot(np.arange(len(data))/400, data, label="ADC")
plt.plot(peaks/400, data[peaks], "x")
plt.xlabel('Time (s)')
plt.ylabel('Adc (V)')
plt.legend(['PPG Signal', 'Peaks'])
plt.title("PPG Signal with Peaks")
plt.savefig('peaks.png')
