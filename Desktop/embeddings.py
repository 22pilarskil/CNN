import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Sequential()
model.add(Embedding(5, 2, input_length=5))

input_array = np.random.randint(5, size=(1, 5))
print(input_array)

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array)