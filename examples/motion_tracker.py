import numpy as np
from handpose.sensor_fusion import *

# Fusion parameters
dt = 1e-1 # Sample period in seconds
num_iter = 30
beta = 4.1 # The suggested beta is 0.041 in Madgwick's paper

# Earth magnetic strength and dip angle
earth_mag=36.0

earth_dip_deg=30.0
earth_dip = earth_dip_deg*np.pi/180.0

# Sensor orientation
angle_deg = 10e0
angle = angle_deg*np.pi/180.0

# Rotaion quaternion
q_e2s = Quaternion.from_angle_axis(angle, 0, 1, 0)

num_dim = 3

# Angular rate
gyro_e = np.zeros(num_dim)
gyro_e[0] = 0.0
gyro_e[1] = 0.0 #angle/dt
gyro_e[2] = 0.0

# Dynamic acceleration in Earth axes
accel_dyn_e = np.zeros(num_dim)
accel_dyn_e[0] = 0.00 # Dynamic acceleration in earth axes
accel_dyn_e[1] = 0.00
accel_dyn_e[2] = 0.00

# IMU simulator
imu_simulator = IMUSimulator(gyro_e, accel_dyn_e, q_e2s, earth_mag=earth_mag, earth_dip=earth_dip)
gyro, accel, mag = imu_simulator.get_imu_data()

# Estimate the initial angles
tracker = MotionTracker(dt=dt, beta=beta, num_iter=200)
tracker.update(gyro, accel, mag)
init_q = tracker.q

# Estimate angles with fewer iterations
tracker = MotionTracker(accel_th=1e-4, vel_th=1e-5, dt=dt, beta=beta, num_iter=num_iter)
tracker.q = init_q
tracker.update(gyro, accel, mag)


angle_axis = tracker.q.to_angle_axis()
simu_angle_deg = angle_axis[0] * 180/np.pi


x_s2e, y_s2e, z_s2e = tracker.unit_vectors_s2e()

print("analytic angle = {} degree or {} radian".format(angle_deg, angle))
print("simu angle = {} (degree)".format(simu_angle_deg))
print("angle-axis = {}".format(angle_axis))
print("----")
print("earth_dip = {} degree or {} radian".format(earth_dip_deg, earth_dip))
print("gyro = {}".format(gyro))
print("accel = {}".format(accel))
print("mag = {}".format(mag))
print("----")
print("accel_dyn_e = {}, ".format(accel_dyn_e))

print("----")

print("Spherical angles = {} (degree)".format(tracker.estimate_spherical_angles() * 180.0/np.pi))
print("x_s2e = {}".format(x_s2e))
print("y_s2e = {}".format(y_s2e))
print("z_s2e = {}".format(z_s2e))

print("w = {}".format(tracker.w))
print("dtheta = {}".format(tracker.dtheta))

print("dynamic accel = {}".format(tracker.accel_dyn))
print("v = {}".format(tracker.vel))
print("dv = {}".format(tracker.dv))
print("dx = {}".format(tracker.dx))

print("----")
print("dt = {} (sec), beta = {}, ".format(dt, beta))
print("damping = {}, ".format(tracker.damping))
