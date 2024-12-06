import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy import blackman
from scipy.signal import butter, filtfilt

data = np.load('ppg.npz', allow_pickle=True)['arr_0'].item()['adc'][100:]

fs = 100 # maybe we need to change this
peaks, _ = find_peaks(data, height=0.0, distance=fs / 2)  
time_diff = np.diff(peaks) / fs
T = np.mean(time_diff)
bpm = 60 / T
max_hrv = np.max(time_diff) - np.min(time_diff)  
rms_hrv = np.sqrt(np.mean((time_diff - np.mean(time_diff))**2))

print("BPM: ", bpm)
print("Max HRV: ", max_hrv)
print("rms_hrv: ", rms_hrv)

plt.figure()
plt.plot(data)
plt.plot(peaks, data[peaks], "x")
plt.title("PPG Signal with Peaks")
plt.savefig('peaks.png')
