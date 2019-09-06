import numpy as np
import matplotlib.pyplot as plt

from handpose.utils import interp1d, interp3d


num_in = 20
num_out = 100

xa = 0.0
xz = 1.0
dx = (xz-xa)/num_in
x_in = np.linspace(xa, xz, num=num_in)    
y_in = np.sin(2.0*np.pi*x_in)   

# Interpolation
z = np.zeros((num_in, 3))
z[:, 0] = y_in
z[:, 1] = y_in
z[:, 2] = y_in

x_out = np.linspace(xa, xz, num=num_out) 
z_out = interp3d(z, num_out)

plt.plot(x_out, z_out[:,0], '-', x_out, z_out[:,1], '--', x_out, z_out[:,2], 'x')
plt.show()
