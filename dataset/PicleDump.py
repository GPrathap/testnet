import glob
from six.moves import cPickle as pickle
import numpy as np
import scipy.misc

import convert_to_tf_record


def createFiles(path, classes):
    datalist = {}
    datalist["images"] = []
    datalist["labels"] = []
    for filename in glob.iglob(path, recursive=True):
        image = scipy.misc.imread(filename).astype(np.float)
        currentClass = ""
        for cla in classes:
            if cla in filename:
                currentClass = cla
                break
        if currentClass == "":
            print("No class label is found")
        else:
            datalist["images"].append(image)
            datalist["labels"].append(classes.index(currentClass))
    return datalist

classesList = ["agricultural",'airplane','baseballdiamond','beach','buildings','chaparral'
    ,'denseresidential','forest','freeway','golfcourse','harbor','intersection','mediumresidential',
              'mobilehomepark','overpass','parkinglot','river','runway','sparseresidential',
              'storagetanks','tenniscourt']
train = False
datapath = "/data/images/uc_train_256_data/**.jpg"
#datapath = "/data/images/uc_test_256/**.jpg"
dataset_storage_location = "/data/satellitegpu/satellite_train.pickle"
#dataset_storage_location = "/data/satellitegpu/satellite_test.pickle"

def convert_into_tfrecord(dataset_path, classes_list, dataset_storage_location, is_train):
    if is_train:
        datalist = createFiles(datapath, classesList)
        with open(dataset_storage_location, 'wb') as f:
             pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
        convert_to_tf_record.run("/data/satellitegpu/")
    else:
        datalist = createFiles(datapath, classesList)
        with open(dataset_storage_location, 'wb') as f:
            pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
        convert_to_tf_record.run("/data/satellitegpu/", train=False)

