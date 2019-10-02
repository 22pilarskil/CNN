import numpy as np
import codecs
import sys
import matplotlib.pyplot as plt
import json
import skimage as sk
import random
from PIL import ImageOps
import random


def random_noise(image_array, mode=None):
    return sk.util.random_noise(image_array, mode)

def random_rotation(image_array):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def pad(image):
    return ImageOps.expand(image,border=8,fill='white')

def random_crop(image_array):
    seed = []
    seed.append(random.randint(0, 8))
    seed.append(random.randint(0, 8))
    return image_array[seed[0]:seed[0]+28, seed[1]:seed[1]+28]


from PIL import Image
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

def displayImage(pixels, label = None):
  figure = plt.gcf()
  figure.canvas.set_window_title("Number display")
  if label != None: plt.title("Label: \"{label}\"".format(label = label))
  else: plt.title("No label")  
  plt.imshow(pixels, cmap = "gray")
  plt.show()
 
