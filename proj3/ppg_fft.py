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


adc = np.load('ppg.npz', allow_pickle=True)['arr_0'].item()['adc'][1000:]
time = np.load('ppg.npz', allow_pickle=True)['arr_0'].item()['time'][1000:]

time_diffs = np.diff(time) 
sampling_rates = 1 / time_diffs  
avg_sampling_rate = np.mean(sampling_rates)

print(f"Average Sampling Rate: {avg_sampling_rate:.2f} Hz")

plt.clf()
plt.plot(time, adc, label="ADC")
plt.xlabel('time')
plt.ylabel('adc')
plt.legend()
plt.savefig("adc.png")

freq, adc_fft = fft(adc, 4096, 500)

plt.clf()
plt.plot(freq, 20 * np.log10(adc_fft+1e-10))
plt.xlim([0,5])
plt.savefig('fft.png')
