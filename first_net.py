import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import to_categorical
np.random.seed(1671)
import os
#import pydotplus
#from ipywidgets import Image
#from io import StringIO
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.system('export http_proxy=http://165.225.66.34:10015')
os.system('export https_proxy=https://165.225.66.34:10015')

NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

(X_train, y_train), (X_test, y_test) = mnist.load_data()

RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], ' training samples')
print(X_test.shape[0], ' testing samples')

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(NB_CLASSES, input_shape = (RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print('Test score {}'.format(score[0]))
print('Test accuracy {}'.format(score[1]))


