# Handpose: A Library designed for hand tracking.


## Main packages used.

* Numpy
* Scipy
* Scikit-learn
* PyTorch
* OpenCV
* Matplotlib
* PyQt5
* Cython


## Compile the module of sensor fusion.

We need to compile the this module with cython before execution.

```sh
cd handpose/handpose/sensor_fusion/madgwick
python setup.py build_ext --inplace
```


