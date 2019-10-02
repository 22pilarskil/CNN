#math
import numpy as np 
import random
import math
#visual
import matplotlib.pyplot as plt
from tkinter import *
import PIL
from PIL import ImageTk, Image, ImageDraw
#functionality
import codecs
import json
from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
#time
import time

def saveNetwork(fileNames, weights, biases, layer="Dense"):
    trainedBiases = {}
    trainedWeights = {}
    if layer=="Dense":
      for i in range(len(weights)):
        trainedBiases[i] = []
        trainedWeights[i] = []
        for j, k in zip(biases[i], weights[i]):
          trainedBiases[i].append(j.tolist())
          trainedWeights[i].append(k.tolist())
    else:
      trainedWeights[0] = []
      trainedBiases[0] = []
      trainedWeights[0].append(weights.tolist())
      trainedBiases[0].append(biases.tolist())
    with open (fileNames[0], 'w+') as JSONFile:
      json.dump(trainedWeights, JSONFile)
    with open(fileNames[1], 'w+') as JSONFile:
      json.dump(trainedBiases, JSONFile)

def retreiveNetwork(fileNames):
  biases = {}
  weights = {}
  b = []
  w = []
  def take(fileName, mode, dictionary, listName):
    with open(fileName, mode) as JSONFile:
      data = json.load(JSONFile)
      for i in data:
        dictionary[i] = []
        for j in range(len(data[i])):
          dictionary[i].append(data[i][j])
      for i in dictionary:
        dictionary[i] = np.asarray(dictionary[i])
    placeHolder = 0
    while (placeHolder<(len(dictionary))):
      for i in dictionary:
        if (int(i)==placeHolder):
          listName.append(dictionary[i])
          placeHolder+=1
  take(fileNames[0], 'r', weights, w)
  take(fileNames[1], 'r', biases, b)
  return w, b

def displayImage(pixels, label = None):
  figure = plt.gcf()
  figure.canvas.set_window_title("Number display")
  if label != None: plt.title("Label: \"{label}\"".format(label = label))
  else: plt.title("No label")  
  plt.imshow(pixels, cmap = "gray")
  plt.show()
  plt.close()

off = 144#529
fss = 576#2116
tf = 24#46
te = 28#50
t = 12#23
  
class featureMap():
  def __init__ (self, kernelDimensions, trainedNet=None):
    self.kernelDimensions = kernelDimensions
    if trainedNet==None:
      self.weights = np.random.randn(5, 5)/np.sqrt(25)
      self.biases = np.random.randn((1))
    else:
      weights, biases = trainedNet
      self.weights = weights[0][0]
      self.biases = biases[0][0]
    self.weightError = np.zeros(self.weights.shape)
    self.biasError = np.zeros((1))
    self.storedWeights = self.weights
    self.storedBiases = self.biases
  def reset(self):
    self.Image = None
    self.Chunks = None
    self.Pools = None
    self.num = None
    self.Z = None
  def convolve(self, image, num, strideLength=1):
    self.num = (num+1)*off
    self.Image = image.reshape(te, te)
    self.Chunks = view_as_windows(self.Image, self.kernelDimensions, strideLength).reshape(fss, 25)*self.weights.reshape(25)
    self.Z = np.array([np.sum(self.Chunks, axis=1)]).reshape(tf, tf)+self.biases
    return self.pool(self.Z)
  def pool(self, chunks):
    pools = view_as_blocks(chunks, (2, 2)).reshape(off, 4)**2
    pieces = [np.sum(pools, axis=1)]
    self.Pools = pools.reshape(off, 2, 2)
    return np.sqrt(pieces).reshape(t, t)
  def getPoolDerivatives(self, delta):
    pools = self.Pools.reshape(off, 4)
    rootSummedSquares = np.sqrt(np.sum(pools**2, axis = 1))
    poolDerivatives = (pools.T/rootSummedSquares).T.reshape(off, 2, 2)
    return (delta*poolDerivatives.T).T
  def reconstructShape(self, poolDerivatives):
    remade = poolDerivatives.reshape(2*off, 2)
    z = [remade[i:i+tf] for i in range(0, 2*off, tf)]
    remade = np.concatenate([zs for zs in z], axis = 1)
    return remade.T
  def backpropCONV(self, delta, inputWeights):
    weights = inputWeights.T[self.num-off:self.num].T
    delta = np.dot(delta, weights).reshape(off)
    poolDerivatives = self.getPoolDerivatives(delta)
    alignedPoolDerivatives = self.reconstructShape(poolDerivatives.transpose(0, 2, 1)).reshape(fss)
    weightError = np.sum((self.Chunks.T*(alignedPoolDerivatives)).T, axis=0).reshape(5, 5)/fss
    biasError = np.sum(alignedPoolDerivatives)/fss
    self.weightError+=weightError
    self.biasError+=biasError
    start = times()
    self.reset()
  def updateMiniBatchCONV(self, miniBatch, eta, lmbda, trainingData):
    self.weights = (1-eta*lmbda/len(trainingData))*self.weights-(float(eta)/len(miniBatch))*self.weightError
    self.biases = self.biases-(float(eta)/len(miniBatch))*self.biasError
    self.weightError = np.zeros(self.weights.shape)
    self.biasError = np.zeros((1))
  def update(self):
    self.storedWeights = self.weights
    self.storedBiases = self.biases
  def get(self, what):
    if what=='w': return self.storedWeights
    else: return self.storedBiases
  def display(self):
      displayImage(self.weights)
    
def sigmoid(x):
  return np.maximum(0, x)
  #return 1/(1+np.exp(-x))
def derivSigmoid(x):
  return np.where(x<0, 0, 1)
  #return sigmoid(x)*(1-sigmoid(x))
def times():
  return time.process_time()
    
class Network():
  def __init__ (self, numFeatures, inputSizes, kernelDimensions, saveNet=False, takeNet=False):
    self.numFeatures = numFeatures
    self.sizes = np.asarray([numFeatures*off, inputSizes[0], inputSizes[1]])
    self.saveNet = saveNet
    if self.saveNet: self.fileNames = [[("weights%dLetter.txt" %(i+1)), ("biases%dLetter.txt" %(i+1))] for i in range(numFeatures)]
    if takeNet:
      self.weights, self.biases = retreiveNetwork(["weightsLetter.txt", "biasesLetter.txt"])
      self.features = [featureMap(kernelDimensions, retreiveNetwork([("weights%dLetter.txt" %(i+1)), ("biases%dLetter.txt" %(i+1))])) for i in range(numFeatures)]
    else:
      self.weights = [np.random.randn(y, x)/np.sqrt(x*x) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
      self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
      self.features = [featureMap(kernelDimensions) for i in range(numFeatures)]
    self.storedWeights = self.weights
    self.storedBiases = self.biases
    self.highestCorrect = 0
    self.streak = 0
    #for feature in self.features: feature.display()
  def backpropMLP(self, image):
    activation = np.asarray([self.features[f].convolve(image[0], f) for f in range(len(self.features))]).reshape(self.numFeatures*off, 1)
    activations = [activation]
    zs = []
    biasError = [np.zeros(b.shape) for b in self.biases]
    weightError = [np.zeros(w.shape) for w in self.weights]
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = [np.zeros(b.shape) for b in self.biases]
    delta[-1] = activation.T-image[1]
    weightError[-1] = np.dot(activations[-2], delta[-1]).T
    biasError[-1] = delta[-1].T
    for i in range(2, len(self.sizes)):
      z = zs[-i]
      delta[-i] = np.dot(delta[-i+1], self.weights[-i+1]) * derivSigmoid(z).T
      biasError[-i] = delta[-i].T
      weightError[-i] = (activations[-i-1]*delta[-i]).T
    for feature in self.features: feature.backpropCONV(delta[0], self.weights[0])
    return weightError, biasError
  def feedforward(self, x):
    activation = np.asarray([self.features[f].convolve(x, f) for f in range(len(self.features))]).reshape(self.numFeatures*off, 1)
    for w, b in zip(self.weights, self.biases): activation = sigmoid(np.dot(w, activation)+b)
    return activation
  def mse(self, x, y):
    prediction = self.feedforward(x).reshape(4, )
    correct = 0
    percent = (np.linalg.norm(y.reshape(1, 4)-prediction) ** 2)
    if np.argmax(prediction) == np.argmax(y):
        correct = 1
    return percent, correct
  def updateMiniBatch(self, miniBatch, lmbda, eta):
    weightError = [np.zeros(w.shape) for w in self.weights]
    biasError = [np.zeros(b.shape) for b in self.biases]
    for image in miniBatch:
      deltaWeightError, deltaBiasError = self.backpropMLP(image)
      weightError = [we+dwe for we, dwe in zip(weightError, deltaWeightError)]
      biasError = [be+dbe for be, dbe in zip(biasError, deltaBiasError)]
    self.weights = [(1-eta*lmbda/len(trainingData))*w-(((float(eta)/len(miniBatch))*we)) for w, we in zip(self.weights, weightError)]
    self.biases = [b-(((float(eta)/len(miniBatch))*be)) for b, be in zip(self.biases, biasError)]
    for feature in self.features: feature.updateMiniBatchCONV(miniBatch, eta, lmbda, trainingData)
  def earlyStop(self, totalCorrect):
    if totalCorrect>=self.highestCorrect:
      self.storedWeights = self.weights
      self.storedBiases = self.biases
      self.highestCorrect = totalCorrect
      self.streak = 0
      for feature in self.features: feature.update()
    else: self.streak+=1
    if self.streak>=10: return False
    else: return False
  def display(self):
    for feature in self.features: feature.display()
  def report(self, testData, j):
      totalCorrect = 0
      totalPercent = 0
      for x, y in testData:
        percent, correct = network.mse(x, y)
        totalCorrect+=correct
        totalPercent+=percent
      totalPercent/=(len(testData))
      print("Epoch %d - Percent Error: %.8f. Total Correct: %d/%d" %(j+1, totalPercent, totalCorrect, len(testData)))
      return totalCorrect
  def SGD(self, lmbda, eta, trainingData, testData, epochs, miniBatchSize):
    self.lmbda=lmbda
    self.eta=eta
    totalCorrect = self.report(testData, -1)
    self.earlyStop(totalCorrect)
    for j in range(epochs):
      #self.lmbda=lmbda*(1-j/2/epochs)
      #self.eta=eta*(1-j/2/epochs)
      start = times()
      random.shuffle(trainingData)
      random.shuffle(testData)
      #for x, y in trainingData: displayImage(x)
      miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
      for miniBatch in miniBatches: self.updateMiniBatch(miniBatch, self.lmbda, self.eta)
      totalCorrect = self.report(testData, j)
      self.report(trainingData, j)
      if self.earlyStop(totalCorrect): break
    self.weights = self.storedWeights
    self.biases = self.storedBiases
    self.report(testData, 100)
    if self.saveNet:
      saveNetwork(["weightsLetter.txt", "biasesLetter.txt"], self.storedWeights, self.storedBiases, layer="Dense")
      for feature, fileName in zip(self.features, self.fileNames): saveNetwork([fileName[0], fileName[1]], feature.get('w'), feature.get('b'), layer="Conv")
  def classify(self, image):
        x = self.feedforward(image.reshape(784, 1)).reshape(1, 4)
        print("Network prediction: %d" %(np.argmax(x)))
        displayImage(image.reshape(28, 28))
def take(file):    
    with open(file, 'r') as JSONFile:
          data = json.load(JSONFile)
          return [[np.array(image), np.array(label)] for image, label in data.values()]
trainingData = take("PracticeLetterImages.txt")
testData = take("TestLetterImages.txt")
kernelDimensions = (5, 5)
sizes = [30, 4]

network = Network(5, sizes, kernelDimensions, saveNet=True, takeNet=True)
network.SGD(0.1, .008, trainingData, testData, 100, 10)
#network.display()
random.shuffle(testData)
for test in testData: network.classify(test[0])
#for td in testData: network.classify(td[0])
