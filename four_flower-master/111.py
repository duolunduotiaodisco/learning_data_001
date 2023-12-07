import tensorflow as tf
import keras
import sys
import numpy as np
import matplotlib
import pandas as pd
import sklearn
print("GPU 可用：" , tf.test.is_gpu_available())
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("numpy 版本:",np.__version__)
print("matplotlib 版本",matplotlib.__version__)
print("pandas 版本",pd.__version__)
print("sklearn 版本",sklearn.__version__)
print("python 版本:",sys.version)
print("Tensorflow，keras，cuda cudnn都是对应好匹配的版本，禁止进行修改或者删除操作，如果更换或者删除不属于售后范围，红色字体部分是提示信息的哈， 不用管就行了，是提示显卡的调用信息，")