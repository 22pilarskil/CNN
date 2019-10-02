from ImageOpen import *
import os
counter = 0
import numpy

def vectorize(num):
    array = np.zeros(4)
    array[num] = 1
    return array
def normalize(rawArray, range_):
    array = np.copy(rawArray).astype(np.float32)
    if range_ == (0, 1):
      return range_
    array-=range_[0]
    dist = abs(range_[0])+abs(range_[1])
    array /= dist
    return array
def addImage(img, value, counter, Images):
    for i in range(15):
        Images[counter] = [random_crop(random_noise(img, "pepper")).tolist(), vectorize(value).tolist()]
        counter+=1
def dump(dump, take):
    Images = {}
    counter = 0
    for filename in os.listdir("/Users/michaelpilarski/Desktop/PennApps_copy/"+take):
        if not filename==".DS_Store":
            #print(filename)
            img = Image.open(take+"/"+filename)
            img = img.convert("L")
            img = img.resize((28, 28))
            img = pad(img)
            img = normalize(np.asarray(img, dtype="int32"), (0, 255))
            if filename[0]=="a": addImage(img, 0, counter, Images)
            elif filename[0]=="b": addImage(img, 1, counter, Images)
            elif filename[0]=="c": addImage(img, 2, counter, Images)
            elif filename[0]=="d": addImage(img, 3, counter, Images)
            counter+=15
            print(len(Images))
    with open(dump, 'w+') as JSONFile:
        json.dump(Images, JSONFile)
        print(len(Images))
dump("PracticeLetterImages.txt", "PracticeLetterImages")
dump("TestLetterImages.txt", "TestLetterImages")

