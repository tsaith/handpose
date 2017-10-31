# Handpose library

Python 3.6 is expected.

## Packages required

* [Numpy](http://www.numpy.org/)
* [numpy-quaternion](https://github.com/moble/quaternion)
* [SciPy](http://www.scipy.org/)
* [Ipython](http://ipython.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [TensorFlow](https://www.tensorflow.org/)
* [keras](http://keras.io/)
* [PyWavelets](https://pywavelets.readthedocs.io)
* [matplotlib](http://matplotlib.org/)
* [Boken](http://bokeh.pydata.org)
* [OpenCV](https://opencv.org/)


## Installation

* [numpy-quaternion](https://github.com/moble/quaternion)
  
```sh
conda install -c moble quaternion
```
or
```sh
pip install numpy numpy-quaternion
```

* [OpenCV](https://opencv.org/)
```sh
conda install opencv=3.2.0
```

## Compilation

We need to compile the cython modules before executing the codes.

```sh
cd handpose/handpose/sensor_fusion/madgwick
python setup.py build_ext --inplace
```


