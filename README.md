# Handpose library

Python 3.6 is expected.

## Packages required

* [Numpy](http://www.numpy.org/)
* [numpy-quaternion](https://github.com/moble/quaternion)
* [SciPy](http://www.scipy.org/)
* [Ipython](http://ipython.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](http://keras.io/)
* [Sonnet](https://deepmind.github.io/sonnet/)
* [PyWavelets](https://pywavelets.readthedocs.io)
* [matplotlib](http://matplotlib.org/)
* [Boken](http://bokeh.pydata.org)
* [OpenCV](https://opencv.org/)


## Installation

* [TensorFlow](https://www.tensorflow.org/)

```sh
pip install tensorflow==1.4.0
```
or with GPU supported,
```sh
pip install tensorflow-gpu==1.4.0
```

* [Keras](http://keras.io/)

With GPU support,
```sh
conda install keras-gpu=2.0.2
```
With CPU only,

```sh
conda install keras=2.0.2
```

* [Sonnet](https://deepmind.github.io/sonnet/)

```sh
pip install dm-sonnet==1.14
```


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
conda install opencv=3.3.0
```

## Compilation

We need to compile the cython modules before executing the codes.

```sh
cd handpose/handpose/sensor_fusion/madgwick
python setup.py build_ext --inplace
```
The lib path is as "~/anaconda2/envs/py3/lib" and
the include path is as "~/anaconda2/envs/py3/include/opencv2".


