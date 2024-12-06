# import time
# import sys
# import ADS1256
# import spidev
# import RPi.GPIO as GPIO
# import matplotlib
# matplotlib.use("Agg")  # Use non-interactive backend suitable for headless environments
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import find_peaks, correlate
# from matplotlib.patches import Ellipse

# SETUP_SPI = False 
# SET_CHANNEL = False
# FREQUENCY = 500
# GPIO_FREQ_FACTOR = 1
# sense_pin = 10

# class FrequencyDetector:
#     def __init__(self):
#         self.current_time = time.time()
#         self.last_time = self.current_time
#         self.start_time = self.current_time
#         self.count = 0
        
#     def update(self):
#         self.count += 1
#         self.current_time = time.time()
        
#     def realtime_update(self):
#         self.last_time = self.current_time
#         self.current_time = time.time()
#         self.count += 1
#         dt = self.current_time - self.last_time
#         if dt > 0:
#             freq = 1 / dt
#             sys.stdout.write("\r")
#             sys.stdout.write(f"{freq:.4f} Hz       ")
#             sys.stdout.flush()

#     def get_freq(self):
#         return self.count / (self.current_time - self.start_time)
    
# class FrequencyRegulator:
#     """Class to regulate frequency of GPIO output."""
#     def __init__(self, frequency):
#         self.frequency = frequency
#         self.start_time = time.time()
#         self.dt = 1 / self.frequency
#         self.next_pulse_time = self.start_time + self.dt
    
#     def is_next_pulse_ok(self):
#         if time.time() > self.next_pulse_time:
#             self.next_pulse_time += self.dt
#             return True
#         return False

# def main():
#     # GPIO setup
#     GPIO.setmode(GPIO.BCM)
#     GPIO.setwarnings(False)
    
#     # ADC setup
#     ADC = ADS1256.ADS1256()
#     ADC.ADS1256_init()
    
#     # SPI setup
#     if SETUP_SPI:
#         SPI = spidev.SpiDev(0, 0)
#         SPI.mode = 0b01
#         SPI.max_speed_hz = 3000000
#     ADC.ADS1256_SetChannal(sense_pin)
#     ADC.ADS1256_WriteCmd(0xFC)  # sync
#     ADC.ADS1256_WriteCmd(0x00)  # wakeup
#     adc_values = []

#     freq_detector = FrequencyDetector()
#     freq_regulator = FrequencyRegulator(FREQUENCY)

#     while(len(adc_values) < 500): 
        
#         if not freq_regulator.is_next_pulse_ok():
#             continue
#         freq_detector.update()    
           
#         adc_value = ADC.ADS1256_Read_ADC_Data() * 5.0 / 0x7fffff
#         adc_values.append(adc_value)
#         print(adc_value)

#     print(f"Frequency: {freq_detector.get_freq():.4f} Hz")

# if __name__ == "__main__":
#     main()


import time
import sys
import ADS1256
import spidev
import RPi.GPIO as GPIO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend
matplotlib.use("Agg")

SETUP_SPI = True
FREQUENCY = 400
GPIO_FREQ_FACTOR = 1
sense_pin = 7

class FrequencyDetector:
    def __init__(self):
        self.current_time = time.time()
        self.last_time = self.current_time
        self.start_time = self.current_time
        self.count = 0
        
    def update(self):
        self.count += 1
        self.current_time = time.time()
        
    def get_freq(self):
        return self.count / (self.current_time - self.start_time)

class FrequencyRegulator:
    """Class to regulate frequency of GPIO output."""
    def __init__(self, frequency):
        self.frequency = frequency
        self.start_time = time.time()
        self.dt = 1 / self.frequency
        self.next_pulse_time = self.start_time + self.dt
    
    def is_next_pulse_ok(self):
        if time.time() > self.next_pulse_time:
            self.next_pulse_time += self.dt
            return True
        return False

def main():
    # GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # ADC setup
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    
    # ADC.ADS1256_SetChannal(sense_pin)
    # ADC.ADS1256_WriteCmd(0xFC)  # sync
    # ADC.ADS1256_WriteCmd(0x00)  # wakeup
    ADC.ADS1256_ConfigADC(ADS1256.ADS1256_GAIN_E['ADS1256_GAIN_1'], ADS1256.ADS1256_DRATE_E['ADS1256_2000SPS'])

    SPI = spidev.SpiDev(0, 0)
    SPI.mode = 0b01
    SPI.max_speed_hz = 300000

    # timesteps = []
    adc_values = []
    freq_detector = FrequencyDetector()
    freq_regulator = FrequencyRegulator(FREQUENCY)

    # Infinite loop
    try:
        while True:
            if not freq_regulator.is_next_pulse_ok():
                continue
        
            freq_detector.update()
            # adc_value = adc_gain* (ADC.ADS1256_Read_ADC_Data() * 5.0 / 0x7fffff)
            adc_value = ADC.ADS1256_GetChannalValue(sense_pin) * 5.0 / 0x7fffff
            adc_values.append(adc_value)
            # timesteps.append(time.time())
            # print(adc_value)

            # # Keep only the last 500 samples for plotting
            # if len(adc_values) > 500:
            #     adc_values.pop(0)

            # Save plot periodically (every 100 readings)
            # if len(adc_values) % 100 == 0:
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(adc_values, label="ADC Values")
            #     plt.title("ADC Voltage Values")
            #     plt.xlabel("Sample")
            #     plt.ylabel("Voltage (V)")
            #     # plt.ylim(-5, 5)  # Set Y-axis limits to ADC range
            #     plt.legend()
            #     plt.grid()
            #     plt.savefig("adc_plot.png")  # Save the plot as an image file
            #     print("Plot updated and saved as adc_plot.png")
                
    except KeyboardInterrupt:
        print(f"Frequency: {freq_detector.get_freq():.4f} Hz")
        np.savez('ppg.npz', {'adc':np.array(adc_values)})

if __name__ == "__main__":
    main()
