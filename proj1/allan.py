from ICM20948 import *
import numpy as np
import allantools
import matplotlib.pyplot as plt

dt = 0.1

if __name__ == '__main__':
    # initialize the ICM20948
    icm20948 = ICM20948()
    # create empty lists to store the sensor data
    clusterAX = []
    clusterAY = []
    clusterAZ = []
    clusterGX = []
    clusterGY = []
    clusterGZ = []
    clusterMX = []
    clusterMY = []
    clusterMZ = []

    try:
        while True:
            # read the sensor data
            icm20948.icm20948_Gyro_Accel_Read()
            icm20948.icm20948MagRead()
            time.sleep(dt)  # delay by delta t
            # append the sensor data to the respective lists
            clusterAX.append(Accel[0])
            clusterAY.append(Accel[1])
            clusterAZ.append(Accel[2])
            clusterGX.append(Gyro[0])
            clusterGY.append(Gyro[1])
            clusterGZ.append(Gyro[2])
            clusterMX.append(Mag[0])
            clusterMY.append(Mag[1])
            clusterMZ.append(Mag[2])

    except KeyboardInterrupt:
        # convert the lists to numpy arrays and save them to a .npz file
        AX = np.array(clusterAX)/16384*9.81
        AY = np.array(clusterAY)/16384*9.81
        AZ = np.array(clusterAZ)/16384*9.81
        GX = np.array(clusterGX)/32.8
        GY = np.array(clusterGY)/32.8
        GZ = np.array(clusterGZ)/32.8
        MX = np.array(clusterMX)
        MY = np.array(clusterMY)
        MZ = np.array(clusterMZ)
        np.savez('IMU_output.npz', AX=AX, AY=AY, AZ=AZ, GX=GX, GY=GY, GZ=GZ, MX=MX, MY=MY, MZ=MZ)
        # calculate the Allan Deviation for the Acceleration, Gyroscope and Magnetometer data
        (ax_tau_out, ax_adev, adeverr, n) = allantools.adev(AX, rate=10, data_type='freq', taus='all')
        (ay_tau_out, ay_adev, adeverr, n) = allantools.adev(AY, rate=10, data_type='freq', taus='all')
        (az_tau_out, az_adev, adeverr, n) = allantools.adev(AZ, rate=10, data_type='freq', taus='all')

        (gx_tau_out, gx_adev, adeverr, n) = allantools.adev(GX, rate=10, data_type='freq', taus='all')
        (gy_tau_out, gy_adev, adeverr, n) = allantools.adev(GY, rate=10, data_type='freq', taus='all')
        (gz_tau_out, gz_adev, adeverr, n) = allantools.adev(GZ, rate=10, data_type='freq', taus='all')

        (mx_tau_out, mx_adev, adeverr, n) = allantools.adev(MX, rate=1/dt, data_type='freq', taus='all')
        (my_tau_out, my_adev, adeverr, n) = allantools.adev(MY, rate=1/dt, data_type='freq', taus='all')
        (mz_tau_out, mz_adev, adeverr, n) = allantools.adev(MZ, rate=1/dt, data_type='freq', taus='all')
        
        # Plot for Acceleration ADEV
        plt.figure()
        plt.loglog(ax_tau_out, ax_adev, label='ax')
        plt.loglog(ay_tau_out, ay_adev, label='ay')
        plt.loglog(az_tau_out, az_adev, label='az')
        plt.legend()
        plt.grid()
        plt.xlabel('taus')
        plt.ylabel('ADEV [ms^-2]')
        plt.savefig('acceleration_adev.png')

        # Plot for Gyroscope ADEV
        plt.figure()
        plt.loglog(gx_tau_out, gx_adev, label='gx')
        plt.loglog(gy_tau_out, gy_adev, label='gy')
        plt.loglog(gz_tau_out, gz_adev, label='gz')
        plt.legend()
        plt.grid()
        plt.xlabel('taus')
        plt.ylabel('ADEV [dps]')
        plt.savefig('gyroscope_adev.png')

        # Plot for Magnetometer ADEV
        plt.figure()
        plt.loglog(mx_tau_out, mx_adev, label='MX')
        plt.loglog(my_tau_out, my_adev, label='MY')
        plt.loglog(mz_tau_out, mz_adev, label='MZ')
        plt.legend()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Tau [s]')
        plt.ylabel('ADEV [µT]')  # Assuming magnetometer units are in microtesla
        plt.title('Allan Deviation for Magnetometer')
        plt.savefig('magnetometer_adev.png')
        plt.close()

        # Combined ADEV plot for both Acceleration and Gyroscope
        plt.figure(figsize=(10, 6))
        # Acceleration
        plt.loglog(ax_tau_out, ax_adev, label='AX', linestyle='-')
        plt.loglog(ay_tau_out, ay_adev, label='AY', linestyle='-')
        plt.loglog(az_tau_out, az_adev, label='AZ', linestyle='-')
        # Gyroscope
        plt.loglog(gx_tau_out, gx_adev, label='GX', linestyle='--')
        plt.loglog(gy_tau_out, gy_adev, label='GY', linestyle='--')
        plt.loglog(gz_tau_out, gz_adev, label='GZ', linestyle='--')
        # Magnetometer
        plt.loglog(mx_tau_out, mx_adev, label='MX', linestyle=':')
        plt.loglog(my_tau_out, my_adev, label='MY', linestyle=':')
        plt.loglog(mz_tau_out, mz_adev, label='MZ', linestyle=':')
        plt.legend()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Tau [s]')
        plt.ylabel('ADEV')
        plt.title('Combined Allan Deviation for IMU Sensors')
        plt.savefig('combined_adev.png')
        plt.close()

        plt.show()
        
