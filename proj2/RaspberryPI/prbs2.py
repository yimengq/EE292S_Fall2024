import time
import sys
import ADS1256
import spidev
import RPi.GPIO as GPIO
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np


SETUP_SPI = False 
SET_CHANNEL = False
FREQUENCY = 20000
GPIO_FREQ_FACTOR = 1 # The frequency of the GPIO signal is GPIO_FREQ_FACTOR times less than the frequency of the ADC signal

prbs_len = ['511', '255', '127', '63', '31', '15', '7']
taps = ['100010000', '10111000', '1100000', '110000', '10100', '1100', '110']
drive_pins = [7, 12, 16, 20, 21]
sense_pins = [1,2,3,4,5,6,7] 

def crosscorrelation_func(sequence1, sequence2):
    N = len(sequence1)
    M = len(sequence2)
    sum_lst = []
    for v in range(N):
        sum_product = 0
        for j in range(M):
            sum_product += sequence1[j] * sequence2[(j + v) % M]
        # find the max value of the sum_product
        sum_lst.append(sum_product)
    return sum_lst

def PRBS(taps, start=1):
	maximal_length = 2 ** len(taps) - 1
	taps = int(taps, 2)
	prbs = ""
	count = 0
	lfsr = start

	while True:
		lsb = lfsr & 1
		prbs += str(lsb)
		lfsr = lfsr >> 1

		if lsb == 1:
			lfsr = lfsr ^ taps
		count +=1

		if lfsr == start:
			break

	PRBS = []
	for bit in prbs:
		if bit == '1':
			PRBS.append(1)
		else:
			PRBS.append(0)

	return np.array(PRBS, dtype=int)

class FrequencyDetector:
    def __init__(self):
        self.current_time = time.time()
        self.last_time = self.current_time
        self.start_time = self.current_time
        self.count = 0
        
    def update(self):
        self.count += 1
        self.current_time = time.time()
        
    def realtime_update(self):
        self.last_time = self.current_time
        self.current_time = time.time()
        self.count += 1
       
        dt = self.current_time - self.last_time
        if dt > 0:
            freq = 1/dt
            sys.stdout.write("\r")
            sys.stdout.write(f"{freq:.4f} Hz       ")
            sys.stdout.flush()
    def get_freq(self):
        return self.count / (self.current_time - self.start_time)
    
class FrequencyRegulator:
    def __init__(self, frequency):
        self.frequency = frequency
        self.start_time = time.time()
        self.dt = 1/self.frequency
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

    for i in drive_pins:
        GPIO.setup(i, GPIO.OUT)
    # ADC setup
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    # ADC.ADS1256_ConfigADC(2, 0xF0)
    # ADC.ADS1256_WriteReg(0, 0x01) 
    
    # SPI setup
    if SETUP_SPI:
        SPI = spidev.SpiDev(0, 0)
        SPI.mode = 0b01
        SPI.max_speed_hz = 3000000
    
    freq_detector = FrequencyDetector()
    freq_regulator = FrequencyRegulator(FREQUENCY)
    
    gpio_freq_factor = GPIO_FREQ_FACTOR
    
    if SET_CHANNEL:
        ADC.ADS1256_SetChannal(7)
        ADC.ADS1256_WriteCmd(0xFC) #sync
        ADC.ADS1256_WriteCmd(0x00) #wakeup  
        
    adc_data = [[] for _ in range(7)] 
    time_data = []
    
    gpio_state = 0
    gpio_count = 0
    
    start_time = time.time()
    
    sense = []
    prbs = PRBS(taps[0])
    shifted_prbs = []
    for i in range(len(drive_pins)):
        shifted_prbs.append(np.roll(prbs, i))
    
    try:
        while(1):
            if not freq_regulator.is_next_pulse_ok():
                continues
            freq_detector.update()

            for pin_index, pin in enumerate(sense_pins):                
                adc_values = []  
                ADC.ADS1256_SetChannal(pin)
                ADC.ADS1256_WriteCmd(0xFC) #sync
                ADC.ADS1256_WriteCmd(0x00) #wakeup   
                for i in range(len(prbs)):
                    t = time.time()
                    GPIO.output(drive_pins[0], int(shifted_prbs[0][i]))
                    GPIO.output(drive_pins[1], int(shifted_prbs[1][i]))
                    GPIO.output(drive_pins[2], int(shifted_prbs[2][i]))
                    GPIO.output(drive_pins[3], int(shifted_prbs[3][i]))
                    GPIO.output(drive_pins[4], int(shifted_prbs[4][i]))
                    adc_value = ADC.ADS1256_Read_ADC_Data()*5.0/0x7fffff
                    adc_data[pin_index].append(adc_value)               
                    time_data.append(t)
            
            print("adc_data",np.array(adc_data).shape)
            print("time",np.array(time_data).shape)
            # for pin in sense_pins:
            #     adc_value = ADC.ADS1256_GetChannalValue(pin) * 5.0 / 0x7fffff
            #     adc_values.append(adc_value)
            # adc_data.append(adc_values)
            # time_data.append(t)

    except KeyboardInterrupt:
        # print("Program terminated by user.")
        
        # # Check if adc_data has been populated
        # if len(adc_data) == 0 or len(time_data) == 0:
        #     print("No data collected. Exiting without saving or plotting.")
        #     return
        
        # adc_data = np.array(adc_data)

        # np.savez("adc_data_log.npz", time=np.array(time_data), adc_data=adc_data)

        # plt.figure(figsize=(10, 5))
        # for channel in range(adc_data.shape[1]):  # Loop over each channel
        #     plt.plot(time_data, adc_data[:, channel], label=f'Channel {channel + 1}')
            
        # plt.xlabel("Time (s)")
        # plt.ylabel("ADC Value")
        # plt.title("ADC Value vs Time for Each Channel")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig("adc_value_vs_time.png")
        print("Plot saved to adc_value_vs_time.png")


if __name__ == "__main__":
    main()