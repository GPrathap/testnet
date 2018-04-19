import glob

from six.moves import cPickle as pickle
from sklearn import preprocessing
import numpy as np
import scipy.misc
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

        image = scipy.misc.imread(filename).astype(np.float)
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


train = False
if train:
    lb = preprocessing.LabelBinarizer()
    datapath = "/data/images/uc_train_256_data/**.jpg"
    datalist = createFiles(datapath, classesList)
    with open("/data/satellitegpu/satellite_train.pickle", 'wb') as f:
         pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
    download_and_convert_satellite.run("/data/satellitegpu/")
else:
    lb = preprocessing.LabelBinarizer()
    datapath = "/data/images/uc_test_256/**.jpg"
    datalist = createFiles(datapath, classesList)
    with open("/data/satellitegpu/satellite_test.pickle", 'wb') as f:
        pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
    download_and_convert_satellite.run("/data/satellitegpu/", train=False)

