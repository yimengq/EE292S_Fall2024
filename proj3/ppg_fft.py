import numpy as np
from numpy import blackman
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# Use non-interactive backend
matplotlib.use("Agg")

def blackman_window(_fft_size, signal):
    window = blackman(_fft_size)
    signal_padded = np.zeros(_fft_size)
    signal_padded[:len(signal)] = signal[:min(len(signal), _fft_size)]
    windowed_signal = signal_padded * window
    return windowed_signal

def fft(signal,_fft_size,f_sampling):
    window = blackman(_fft_size)
    signal_padded = np.zeros(_fft_size)
    signal_padded[:len(signal)] = signal[:min(len(signal), _fft_size)]
    windowed_signal = signal_padded * window
    fft_output = np.fft.fft(windowed_signal[:_fft_size])
    fft_freqs = np.fft.fftfreq(_fft_size, 1.0 / f_sampling)
    return fft_freqs[:_fft_size//2], np.abs(fft_output[:_fft_size//2])


adc = np.load('ppg.npz', allow_pickle=True)['arr_0'].item()['adc'][400:]
adc = adc - np.mean(adc)
print(len(adc))
# time = np.load('ppg.npz', allow_pickle=True)['arr_0'].item()['time'][200:]
# time = time - time[0]

# time_diffs = np.diff(time) 
# sampling_rates = 1 / time_diffs  
# avg_sampling_rate = np.mean(sampling_rates)

# print(f"Average Sampling Rate: {avg_sampling_rate:.2f} Hz")

plt.clf()
plt.plot(adc, label="ADC")
plt.xlabel('time (s)')
plt.ylabel('adc (V)')
plt.legend()
plt.savefig("adc.png")

freq, adc_fft = fft(adc, 4096, 400)
# freq, adc_fft = fft(adc, 1024, 100)
freq_range = (freq >= 1.0) & (freq <= 2)  
index = np.argmax(adc_fft[freq_range]) 
strongest_freq = freq[freq_range][index]
T_fft = 1 / strongest_freq
bpm_fft = 60 / T_fft
print("BPM from FFT: ", bpm_fft)
print("Strongest frequency: ", strongest_freq)

plt.clf()
plt.plot(freq, 20 * np.log10(adc_fft+1e-10))
plt.plot(strongest_freq, 20 * np.log10(adc_fft[freq == strongest_freq]), 'ro')
plt.xlim([0,2])
plt.xlabel('freq (Hz)')
plt.savefig('fft.png')
