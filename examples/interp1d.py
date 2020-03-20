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

y_out = interp1d(y_in, num_out)   
x_out = np.linspace(xa, xz, num=num_out) 

plt.plot(x_in, y_in, 'o')
plt.plot(x_out, y_out, '-x')
plt.show()
