from ICM20948 import *
import time
import sys
import math as m
import matplotlib as plt
import numpy as np
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter

import matplotlib.pyplot as plt

dt = 0.1 # delta time for loop updates, 10 samples/sec
GyroA = np.array([0, 0, 0]) 


# This function is called periodically from FuncAnimation
def animate(i, xs, ys, mode="accel"):
    '''
    Inputs:
        i: int # Frame number
        xs: list # x-axis values
        ys: list # y-axis values
    '''
    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Tilt Sensor {}'.format(mode))
    plt.ylabel('Degrees')
    plt.xlabel('Time (s)')


### calculate tilt angle
# calculate tilt angle using accelerometer
def calc_accel_tilt():
    '''
    Outputs:
        (Output): float # Angle calculated by accelerometer values
    '''
    x = Accel[0] # a_x
    y = Accel[1] # a_y
    z = Accel[2] # a_z

    return math.atan2(math.sqrt(x**2 + y**2), z) * 180/math.pi # computes angle relative to global XY plane (orthog to z-axis)
# calculate tilt angle using gyroscope
def calc_gyro_tilt():
    gyro_sens = 32.8
    a = -math.radians(Gyro[2]*dt/gyro_sens)
    b = -math.radians(Gyro[1]*dt/gyro_sens)
    c = -math.radians(Gyro[0]*dt/gyro_sens)

    sa = math.sin(a)
    ca = math.cos(a)
    sb = math.sin(b)
    cb = math.cos(b)
    sc = math.sin(c)
    cc = math.cos(c)
    g = np.array(GyroA).copy()
    # rotate gyro values to global frame
    GyroA[0] = ca*cb*g[0] + (ca*sb*sc-sa*cc)*g[1] + (ca*sb*cc+sa*sc)*g[2]
    GyroA[1] = sa*cb*g[0] + (sa*sb*sc+ca*cc)*g[1] + (sa*sb*cc-ca*sc)*g[2]
    GyroA[2] = -sb*g[0] + cb*sc*g[1] + cb*cc*g[2]

    return math.atan2(math.sqrt(GyroA[0]**2 + GyroA[1]**2), GyroA[2]) * 180/math.pi


### fuse gyro and accel measurements
def calc_fuse_tilt(accelTilt, gyroTilt, alpha=0.2):
    '''
    Inputs:
        accelTilt: float # Current angle, according to the accelerometer
        gyroTilt: float # Current angle, according to the accelerometer
        alpha: float # Weight of the accelTilt. Weight of gyroTilt is 1-alpha
    Outputs:
        (Output): float # Fused angle
    '''
    # fuse the two measurements
    return accelTilt*alpha + (1-alpha)*gyroTilt

def kalman_filter(accelx,dt):   
    # initial state (accel, velocity, and position)
    x = np.array([0., 0., 0.])
    # state transition matrix
    F = np.array([[1., dt, 0.5*dt**2],
                  [0., 1., dt],
                  [0., 0., 1.]])
    # measurement function
    H = np.array([[0., 0., 1.]])
    # measurement noise
    R = np.array([[0.01]])
    # process noise
    Q = np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]])
    # covariance matrix
    P = np.array([[10000.,  0., 0.],
                 [ 0., 10000., 0.],
                 [ 0., 0., 10000.]])
    # initialize Kalman Filter
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = x
    kf.F = F
    kf.H = H
    kf.R = R
    kf.Q = Q
    kf.P = P
    
    kf.predict()                        # State prediction
    kf.update(accelx)                                   # Update

    return kf.x
    


if __name__ == '__main__':
    # initialize ICM20948
    icm20948=ICM20948()

    # initialize lists to store tilt values
    accel_tilt = []
    gyro_tilt = []
    fuse_tilt = []
    
    pos = []

    accelx_lst = []
    accelx_avg = -1
    
    ts = []
    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("accel")
    ax[1].set_title("gyro")
    ax[2].set_title("fuse")

    icm20948.icm20948_Gyro_Accel_Read()
    GyroA = np.array(Accel).copy()
    currTime = time.time()

    try:
        while True:
            # read data from ICM20948
            icm20948.icm20948_Gyro_Accel_Read()
            time.sleep(dt) # delay by delta t
            currTime += dt

            ts.append(currTime)
            # calculate tilt angles (accel, gyro, fused)
            accel_tilt.append(calc_accel_tilt()) # calculate level using accelerometer
            gyro_tilt.append(calc_gyro_tilt()) # calculate level using gyro
            fuse_tilt.append(calc_fuse_tilt(accel_tilt[-1], gyro_tilt[-1],0.5))

            if len(accelx_lst) < 50:
                accelx_lst.append(Accel[0])
            else:
                accelx_avg = sum(accelx_lst)/50
                new_accelx = Accel[0] - accelx_avg
                # accelx_lst.pop(0)
                # accelx_lst.append(Accel[0])
                
                state = kalman_filter(new_accelx,dt)
                pos.append(state[0])
                print("position",state[0])

    except KeyboardInterrupt:
        # plot tilt angles when program is interrupted
        pos_plot = ax[0].plot(ts, pos)
        # accel_plot = ax[0].plot(ts, accel_tilt)
        # gyro_plot = ax[1].plot(ts, gyro_tilt)
        # fuse_plot = ax[2].plot(ts, fuse_tilt)
        plt.savefig('part2_pos.png')
