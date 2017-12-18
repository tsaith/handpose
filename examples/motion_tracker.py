import numpy as np
from handpose.sensor_fusion import *

# Sensor fusion
dt = 1e-2 # Sample period in seconds
num_iter = 1000
beta = 0.041 # The suggested beta is 0.041 in Madgwick's paper
fast_version = False

# Earth magnetic strength and dip angle
earth_mag=36.0

earth_dip_deg=30.0
earth_dip = earth_dip_deg*np.pi/180

# Sensor orientation
angle_deg = 3.0
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

# Motion tracker
tracker = MotionTracker()
tracker.update(gyro_in, accel_in, mag_in, quat=quat, video=video)

q_es_simu = tracker.quat
q_se_simu = q_es_simu.inv()

angle_axis_simu = q_es_simu.to_angle_axis()
angle_deg_simu = angle_axis_simu[0]/np.pi*180

# Orientation angles
roll = tracker.get_roll() * 180.0/np.pi
pitch = tracker.get_pitch() * 180.0/np.pi
roll = tracker.get_yaw() * 180.0/np.pi

print("----")
print("accel_dyn_e_in = {}, ".format(accel_dyn_e_in))
print("----")
print("gyro_in = {}".format(gyro_in))
print("accel_in = {}".format(accel_in))
print("mag_in = {}".format(mag_in))
print("----")
print("dynamic accel = {}".format(tracker.accel_dyn))
print("----")
print("roll, pitch, yaw = {}, {}, {} (degree)".format(roll, pitch, yaw))


# Analytic dynamic acceleration in Earth axes
accel_dyn_e_in[0] = 0.1 # Dynamic acceleration in earth axes
accel_dyn_e_in[1] = 0.1
accel_dyn_e_in[2] = 0.1

# IMU simulator
imu_simulator = IMUSimulator(gyro_e, accel_dyn_e_in, q_se, earth_mag=earth_mag, earth_dip=earth_dip)
gyro_in, accel_in, mag_in = imu_simulator.get_imu_data()

# Estimate the quaternion
sf.update_ahrs(gyro_in, accel_in, mag_in)
quat = sf.quat

tracker.update(gyro_in, accel_in, mag_in, quat=quat, video=video)

print("----")
print("accel_dyn_e_in = {}, ".format(accel_dyn_e_in))
print("----")
print("gyro_in = {}".format(gyro_in))
print("accel_in = {}".format(accel_in))
print("mag_in = {}".format(mag_in))
print("----")
print("dynamic accel = {}".format(tracker.accel_dyn))

