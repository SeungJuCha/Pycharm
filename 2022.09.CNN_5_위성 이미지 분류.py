import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

DATA_DIR = 'dataset/'

(train_ds, valid_ds), info = tfds.load('eurosat/rgb',split=['train[:80%]','train[80%]'])

"""나머지 colab eurosat classify"""