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
from cifar import download_and_convert_satellite


def createFiles(path, classes):
    datalist = {}
    datalist["images"] = []
    datalist["labels"] = []
    for filename in glob.iglob(path, recursive=True):
        image = cv.imread(filename)
        image = np.array(image, dtype=np.uint8)
        # image = tf.gfile.FastGFile(filename, 'rb').read()
        currentClass = ""
        for cla in classes:
            if cla in filename:
                currentClass = cla
                break
        if currentClass == "":
            print("NO claass is foune rggreub")
        else:
            datalist["images"].append(image)
            datalist["labels"].append(classes.index(currentClass))
    return datalist

classesList = ["agricultural",'airplane','baseballdiamond','beach','buildings','chaparral'
    ,'denseresidential','forest','freeway','golfcourse','harbor','intersection','mediumresidential',
              'mobilehomepark','overpass','parkinglot','river','runway','sparseresidential',
              'storagetanks','tenniscourt']

# lb = preprocessing.LabelBinarizer()
# datapath = "/home/geesara/Downloads/data/uc_train_256_data/**.jpg"
# datalist = createFiles(datapath, classesList)
#
# with open("/data/satellite/satellite_train.pickle", 'wb') as f:
#     pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)


download_and_convert_satellite.run("/data/satellite/")


