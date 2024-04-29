import numpy as np
import cv2
from keras import backend as k
from keras.models import sequential, load_model
from keras.layers.core import Dense

import pickle

model= load_model('Datos/archivo.pkl')
