import numpy as np

from handpose.tracking import *
from handpose.sensor_fusion import *
from handpose.utils import *


# Motion tracker

# Sensor fusion
dt = 1e-2 # Sample period in seconds
num_iter = 1000
beta = 0.041 # The suggested beta is 0.041 in Madgwick's paper
fast_version = True

# Earth magnetic strength and dip angle
earth_mag=36.0

earth_dip_deg=30.0
earth_dip = earth_dip_deg*np.pi/180   


# Sensor orientation
angle_deg = 30.0
angle = angle_deg*np.pi/180 

# Rotaion quaternion, sensor axes respective to user axes
q_se = Quaternion.from_angle_axis(angle, 0, 1, 0)
angle_axis_in = q_se.to_angle_axis()

num_dim = 3

# Angular rate
gyro_e = np.zeros(num_dim)
gyro_e[0] = 0.0
gyro_e[1] = 0.0 #angle/dt
gyro_e[2] = 0.0

# Analytic dynamic acceleration in Earth axes
accel_dyn_e_in = np.zeros(num_dim)
accel_dyn_e_in[0] = 0.00 # Dynamic acceleration in earth axes
accel_dyn_e_in[1] = 0.00
accel_dyn_e_in[2] = 0.00

# IMU simulator
imu_simulator = IMUSimulator(gyro_e, accel_dyn_e_in, q_se, earth_mag=earth_mag, earth_dip=earth_dip)
gyro_in, accel_in, mag_in = imu_simulator.get_imu_data()

# Prepare the video
timesteps = 5
rows = 48
cols = 48
channels = 1
shape = (timesteps, rows, cols, channels)
video = np.zeros(shape, dtype=np.int32)
video = None

# Eestimate the quaternion
sf = SensorFusion(dt, beta, num_iter=num_iter, fast_version=fast_version)
sf.update_ahrs(gyro_in, accel_in, mag_in)
quat = sf.quat

# Xsens motion tracker
tracker = XsensMotionTracker(accel_th=1e-4, vel_th=1e-5)

q_es_simu = quat
q_se_simu = q_es_simu.inv()

tracker.update(gyro_in, accel_in, mag_in, accel_dyn=accel_dyn_e_in, 
               motion_status=1, quat=q_se_simu)

angle_axis_simu = q_se_simu.to_angle_axis()
angle_deg_simu = angle_axis_simu[0]/np.pi*180   

roll, pitch, yaw = tracker.get_orientation_angles()
roll_deg = roll / np.pi*180
pitch_deg = pitch / np.pi*180
yaw_deg = yaw / np.pi*180

x_screen_axes = earth_to_screen_axes(tracker.x)

print("input angle = {} degree ({} radian)".format(angle_deg, angle))
print("simu  angle = {} degree".format(angle_deg_simu))

print("input angle-axis (s2e): {} degree, {} axis".format(angle_axis_in[0]/np.pi*180, angle_axis_in[1:]))
print("simu  angle-axis (s2e): {} degree, {} axis".format(angle_axis_simu[0]/np.pi*180, angle_axis_simu[1:]))
print("----")
print("accel_dyn_e_in = {}, ".format(accel_dyn_e_in))
print("earth_dip = {} degree or {} radian".format(earth_dip_deg, earth_dip))
print("----")
print("gyro_in = {}".format(gyro_in))
print("accel_in = {}".format(accel_in))
print("mag_in = {}".format(mag_in))
print("----")

print("dynamic accel = {}".format(tracker.accel_dyn))
print("dt = {} (sec), beta = {}, ".format(dt, beta))
print("v = {}".format(tracker.vel))
print("dv = {}".format(tracker.dv))
print("x = {}".format(tracker.x))

print("----")
print("roll, pitch, yaw (degree)= {}, {}, {}".format(roll_deg, pitch_deg, yaw_deg))

print("x_screen_axes = {}".format(x_screen_axes))
