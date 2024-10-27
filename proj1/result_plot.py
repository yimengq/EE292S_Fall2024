import numpy as np
import matplotlib.pyplot as plt

# List of filenames to load
filenames = ['data_1.npz', 'data_3.npz', 'data_6.npz']
lengths = ['1 ft length', '3 ft length', '6 ft length']  # Column subtitles

# Initialize the figure
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Position, Velocity, and Acceleration (Kalman Filter) vs. Time")

# Loop through each file and load data
for col, (filename, length) in enumerate(zip(filenames, lengths)):
    # Load data from file
    data = np.load(filename)
    tn = data['tn']
    pos_kal = data['pos_kal']
    vel_kal = data['vel_kal']
    accel_kal = data['accel_kal']

    # Plot pos_kal
    axs[0, col].plot(tn, pos_kal)
    axs[0, col].set_title(length)
    axs[0, col].set_xlabel('Time')
    axs[0, col].set_ylabel('Position (pos_kal) [m]')

    # Plot vel_kal
    axs[1, col].plot(tn, vel_kal)
    axs[1, col].set_xlabel('Time')
    axs[1, col].set_ylabel('Velocity (vel_kal) [m/s]')

    # Plot accel_kal
    axs[2, col].plot(tn, accel_kal)
    axs[2, col].set_xlabel('Time')
    axs[2, col].set_ylabel('Acceleration (accel_kal) [m/s^2]')

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show plot
plt.show()
