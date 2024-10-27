import numpy as np
import matplotlib.pyplot as plt

# List of filenames and corresponding ground truth distances
filenames = ['data_1.npz', 'data_2.npz', 'data_3.npz', 'data_4.npz', 'data_5.npz', 'data_6.npz']
ground_truth_distances = [0.3048, 0.6096, 0.9144, 1.2192, 1.524, 1.8288]  # in meters

# Initialize lists to store computed max values and errors
max_pos_kal_values = []
distance_errors = []

# Loop through each file to compute max pos_kal and distance error
for filename, true_distance in zip(filenames, ground_truth_distances):
    # Load data from file
    data = np.load(filename)
    pos_kal = data['pos_kal']
    
    # Find the maximum value in pos_kal
    max_value = np.max(pos_kal)
    max_pos_kal_values.append(max_value)
    
    # Compute the distance error
    error = abs(max_value - true_distance)
    distance_errors.append(error)

# Plotting the distance error as a function of ground truth distance
plt.figure(figsize=(8, 6))
plt.plot(ground_truth_distances, distance_errors, marker='o', linestyle='-')
plt.xlabel("Distance Traveled (m)")
plt.ylabel("Distance Error (m)")
plt.title("Distance Error vs. Distance Traveled")
plt.grid(True)
plt.show()

# Print out maximum pos_kal and distance errors for each file
for filename, max_value, true_distance, error in zip(filenames, max_pos_kal_values, ground_truth_distances, distance_errors):
    print(f"{filename}: Max pos_kal = {max_value}, Ground Truth = {true_distance}, Distance Error = {error}")
