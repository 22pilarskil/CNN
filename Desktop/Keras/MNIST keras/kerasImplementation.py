from keras.models import Sequential
from keras.layers import Dense
import codecs
import numpy as np


def loadData():
  print("Loading Data...")
  def toInt(b):
    return int(codecs.encode(b, "hex"), 16)
  def normalize(rawArray, range_):
    array = np.copy(rawArray).astype(np.float32)
    if range_ == (0, 1):
      return range_
    array-=range_[0]
    dist = abs(range_[0])+abs(range_[1])
    array /= dist
    return array
  def vectorize(num):
    array = np.zeros(10)
    array[num] = 1
    return array
  def loadFile(fileName, mode="rb"):
    with open(fileName, mode) as raw:
      data = raw.read()
      magicNumber = toInt(data[:4])
      length = toInt(data[4:8])
      if magicNumber==2049:
        parsed = np.frombuffer(data, dtype=np.uint8, offset = 8)
      elif magicNumber==2051:
        numOfRows = toInt(data[8:12])
        numOfColumns = toInt(data[12:16])
        parsed = normalize(np.frombuffer(data, dtype=np.uint8, offset = 16).reshape(length, numOfRows*numOfColumns), (0, 255))
      else: return -1
      return parsed
  data = {"train":[], "test":[]}
  trainLabels = loadFile("/Users/michaelpilarski/Desktop/CNNs/Data/train-labels-idx1-ubyte")
  data["trainImages"] = loadFile("/Users/michaelpilarski/Desktop/CNNs/Data/train-images-idx3-ubyte")
  data["trainLabels"] = np.asarray([vectorize(i) for i in trainLabels])
  testLabels = loadFile("/Users/michaelpilarski/Desktop/CNNs/Data/t10k-labels-idx1-ubyte")
  data["testImages"] = loadFile("/Users/michaelpilarski/Desktop/CNNs/Data/t10k-images-idx3-ubyte")
  data["testLabels"] = np.asarray([vectorize(i) for i in testLabels])
  return data["trainImages"], data["trainLabels"], data["testImages"], data["testLabels"]

x_train, y_train, x_test, y_test = loadData()


model = Sequential()
model.add(Dense(units=30, activation='relu', input_shape=(784,)))
model.add(Dense(units=10, activation='softmax', input_shape=(30,)))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
