import glob

from six.moves import cPickle as pickle
import cv2 as cv
from sklearn import preprocessing
import numpy as np
# "
#
# a = {"sdfg", "sdfsdf"}
#
# with open("images.pickle", 'wb') as f:
#     pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open("images.pickle", 'rb') as ff:
#     bb = pickle.load(ff)

def createFiles(path, classes):
    datalist = {}
    datalist["images"] = []
    datalist["labels"] = []
    for filename in glob.iglob(path, recursive=True):
        image = cv.imread(filename)
        image = np.array(image, dtype=np.float64)
        currentClass = ""
        for cla in classes:
            if cla in filename:
                currentClass = cla
                break
        if currentClass == "":
            print("NO claass is foune rggreub")
        else:
            datalist["images"].append(image.flatten())
            datalist["labels"].append(classes.index(currentClass))
    return datalist

classesList = ["agricultural",'airplane','baseballdiamond','beach','buildings','chaparral'
    ,'denseresidential','forest','freeway','golfcourse','harbor','intersection','mediumresidential',
              'mobilehomepark','overpass','parkinglot','river','runway','sparseresidential',
              'storagetanks','tenniscourt']

lb = preprocessing.LabelBinarizer()
datapath = "/home/runge/Downloads/data/uc_test_256/**.jpg"
datalist = createFiles(datapath, classesList)

with open("/data/satellite/satellite_train.pickle", 'wb') as f:
    pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("/data/satellite/satellite_train.pickle", 'rb') as ff:
    bb = pickle.load(ff)

print(len(bb["images"]))
