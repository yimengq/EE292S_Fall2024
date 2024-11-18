import time
import sys
import ADS1256
import spidev
import RPi.GPIO as GPIO
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend suitable for headless environments
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, correlate
from matplotlib.patches import Ellipse

# Constants
SETUP_SPI = False 
SET_CHANNEL = False
FREQUENCY = 20000
GPIO_FREQ_FACTOR = 1
prbs_len = ['511', '255', '127', '63', '31', '15', '7']
taps = ['100010000', '10111000', '1100000', '110000', '10100', '1100', '110']
drive_pins = [7, 12, 16, 20, 21]
sense_pins = [1, 2, 3, 4, 5, 6, 7]
phases = [50, 100, 150, 200, 250]

def crosscorrelation_func(sequence1, sequence2):

    # correlation = correlate(sequence1, sequence2)
    # return correlation
    N = len(sequence1)
    correlation = np.zeros(N)
    sequence1 = np.array(sequence1)
    sequence2 = np.array(sequence2)
    for i in range(N):
        seq2_delay = np.roll(sequence2, i)
        prod = sequence1 * seq2_delay
        correlation[i] = np.sum(prod)
    return correlation

def PRBS(taps, seed=1):
    lfsr = seed
    taps_int = int(taps, 2)
    prbs_sequence = []

    while True:
        lsb = lfsr & 1
        prbs_sequence.append(lsb)
        lfsr = (lfsr >> 1) ^ (taps_int if lsb else 0)
        if lfsr == seed:
            break
    return np.array(prbs_sequence, dtype=int)

def compute_centroid(peak_matrix, spacing=1):
    
    total_intensity = np.sum(peak_matrix)
    
    if total_intensity == 0:
        raise ValueError("Peak matrix has no non-zero values.")

    y_coords, x_coords = np.indices(peak_matrix.shape)
    centroid_x = np.sum(x_coords * peak_matrix) / total_intensity
    centroid_y = np.sum(y_coords * peak_matrix) / total_intensity

    std_x = np.sqrt(np.sum((x_coords - centroid_x) ** 2 * peak_matrix) / total_intensity)
    std_y = np.sqrt(np.sum((y_coords - centroid_y) ** 2 * peak_matrix) / total_intensity)

    major_axis = max(std_x, std_y) * spacing * 2
    minor_axis = min(std_x, std_y) * spacing * 2

    return (centroid_x * spacing, centroid_y * spacing), (major_axis, minor_axis)

class FrequencyDetector:
    """Class to detect frequency of signal updates."""
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
            freq = 1 / dt
            sys.stdout.write("\r")
            sys.stdout.write(f"{freq:.4f} Hz       ")
            sys.stdout.flush()

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

def plot_heatmap(data, output_file="heatmap.png"):
    """Generate a heatmap from the provided 2D array."""
    plt.figure(figsize=(8, 6))

    # Create the heatmap
    plt.imshow(data, cmap="hot", interpolation="nearest", aspect="auto")

    # Add color bar for reference
    plt.colorbar(label="Peak Value")

    # Add labels and title
    plt.xlabel("Top Peaks (1 to 5)")
    plt.ylabel("Pin Index")
    plt.title("Heatmap of Peak Values")

    # Add custom tick labels for the y-axis
    plt.yticks(range(data.shape[0]), [f"Pin {i+1}" for i in range(data.shape[0])])

    # Save the heatmap to a file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for i in drive_pins:
        GPIO.setup(i, GPIO.OUT)
    
    # ADC setup
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    
    # SPI setup
    if SETUP_SPI:
        SPI = spidev.SpiDev(0, 0)
        SPI.mode = 0b01
        SPI.max_speed_hz = 3000000
    
    freq_detector = FrequencyDetector()
    freq_regulator = FrequencyRegulator(FREQUENCY)
    
    # PRBS and shifted PRBS setup
    prbs = PRBS(taps[0])
    shifted_prbs = [np.roll(prbs, phase) for phase in phases]
    
    cor = [[] for _ in range(7)]  # Store cross-correlation for each pin
    peak_info = [[] for _ in range(7)]  # Store peak values and indices for each pin
    peak_values_array = np.zeros((7, 5))
    centroid_coords = [] # store centroid coords

    def update_data():
        """Function to read ADC data, calculate correlation, and update cor list."""
        for pin_index, pin in enumerate(sense_pins):
            adc_values = []
            ADC.ADS1256_SetChannal(pin)
            ADC.ADS1256_WriteCmd(0xFC)  # sync
            ADC.ADS1256_WriteCmd(0x00)  # wakeup
            for i in range(len(prbs)):
                # Update GPIO based on shifted PRBS values
                GPIO.output(drive_pins[0], int(shifted_prbs[0][i]))
                GPIO.output(drive_pins[1], int(shifted_prbs[1][i]))
                GPIO.output(drive_pins[2], int(shifted_prbs[2][i]))
                GPIO.output(drive_pins[3], int(shifted_prbs[3][i]))
                GPIO.output(drive_pins[4], int(shifted_prbs[4][i]))
                adc_value = ADC.ADS1256_Read_ADC_Data() * 5.0 / 0x7fffff
                adc_values.append(adc_value)
            
            correlation = crosscorrelation_func(adc_values, prbs)
            correlation_adj = correlation -  np.mean(correlation)          
            correlation_adj = correlation -  np.mean(correlation)
            cor[pin_index].append(correlation_adj)
            
            # Find peaks greater than 2
            peaks, _ = find_peaks(correlation_adj, height=10)
            peak_values = correlation_adj[peaks]
            # print("peak is",peaks)
            
            if len(peaks) == 0:
                peak_values_array[pin_index] = np.zeros((1, 5))
            else:
                for peak in peaks:
                    if abs(peak - 50) < 5:
                        peak_values_array[pin_index][4] = correlation_adj[peak]
                    elif abs(peak - 100) < 5:
                        peak_values_array[pin_index][3] = correlation_adj[peak]
                    elif abs(peak - 150) < 5:
                        peak_values_array[pin_index][2] = correlation_adj[peak]
                    elif abs(peak - 200) < 5:
                        peak_values_array[pin_index][1] = correlation_adj[peak]
                    else:
                        peak_values_array[pin_index][0] = correlation_adj[peak]

            cor[pin_index].append(correlation_adj)
    
    def save_plot(idx=0):
        """Save the plot of the latest correlation data and a heatmap of peak values for each pin."""
        # plt.subplot(2, 1, 1)
        for pin_index in range(len(cor)):
            if len(cor[pin_index]) > 0:
                plt.plot(cor[pin_index][-1], label=f'Pin {pin_index + 1}')
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.title("Real-time Cross-Correlation")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(f"fig/realtime_corr_{idx}.png")

        # # Plot the heatmap of peak values
        # plt.subplot(2, 1, 2)
        plt.clf()
        plt.figure(figsize=(5, 7))
        plt.imshow(peak_values_array, cmap="Greys", aspect="auto", vmin=0, vmax=20)
        plt.colorbar(label="Peak Value")
        plt.xlabel("Top Peaks (1 to 5)")
        plt.ylabel("Pin Index")
        plt.title("Heatmap of Real-time Peak Values for Each Pin")
        plt.yticks(range(7), [f"Pin {i+1}" for i in range(7)])

        try:
            coords, major_minor_axes = compute_centroid(peak_values_array, spacing=1)
            centroid_x, centroid_y = coords  # Get the centroid coordinates in floating-point
            major_axis, minor_axis = major_minor_axes  # Get the lengths of the axes
            print(coords)
            centroid_coords.append(np.array([centroid_x, centroid_y]))
            # print(major_axis,minor_axis)
            # Add the centroid as a red dot
            plt.scatter(centroid_x, centroid_y, color='red', label='Centroid', s=100, edgecolor='black')

            # Draw the ellipse
            ellipse = Ellipse(
                (centroid_x, centroid_y),  # Position of the ellipse
                width= 3*major_axis,  # Major axis length
                height= 3*minor_axis,  # Minor axis length
                angle=0,  # Orientation of the ellipse (optional, default is 0)
                edgecolor='blue',  # Color of the ellipse edge
                facecolor='none',  # Make the ellipse transparent
                linewidth=2  # Thickness of the ellipse border
            )
            
            plt.gca().add_patch(ellipse)  # Add the ellipse to the plot

            plt.legend(loc="upper right")
        except:
            print("error")
        
        plt.xlim(0, 4)  # Limit x-axis to 0–4
        plt.ylim(6, 0)  # Limit y-axis to 0–6 (inverted for heatmap)
        plt.savefig(f"fig/realtime_heatmap_{idx}.png")
        plt.close()  # Close the figure to free memory
        plt.tight_layout()

    dts = []
    try:
        # while True:
        for idx in range(100):
            if freq_regulator.is_next_pulse_ok():
                last_time = time.time()
                update_data()  # Collect and compute correlation in real time
                now = time.time()
                dt = now - last_time
                dts.append(dt)
                freq_detector.update()
                save_plot(idx=idx)  # Save plot after each update
                
    except KeyboardInterrupt:
        print("avg fps:", np.mean(1./np.array(dts)))
        centroid_coords = np.stack(centroid_coords, axis=0)
        centroid_motion = np.diff(centroid_coords, axis=0)
        rms_x = np.sqrt(np.mean(centroid_motion[:,0]**2))
        rms_y = np.sqrt(np.mean(centroid_motion[:,1]**2))
        print(rms_x, rms_y)
        GPIO.cleanup()
        print("Program interrupted by user.")

if __name__ == "__main__":
    main()
