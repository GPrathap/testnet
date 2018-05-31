import glob
import os
import numpy as np
import scipy.misc
from six.moves import cPickle as pickle

from convert_to_tf_record import DataConvertor

style_label_files_train = ["/data/brazilian_coffee_scenes/fold1.txt",
                     "/data/brazilian_coffee_scenes/fold2.txt",
                     "/data/brazilian_coffee_scenes/fold3.txt"]

style_label_files_test = [ "/data/brazilian_coffee_scenes/fold4.txt",
                     "/data/brazilian_coffee_scenes/fold5.txt"]

paths_train = ["/data/brazilian_coffee_scenes/fold1/{}.jpg",
         "/data/brazilian_coffee_scenes/fold2/{}.jpg",
         "/data/brazilian_coffee_scenes/fold3/{}.jpg"]

paths_test = ["/data/brazilian_coffee_scenes/fold4/{}.jpg"
         ,"/data/brazilian_coffee_scenes/fold5/{}.jpg"]

classes_list = ["noncoffee", "coffee"]

def create_file(paths, style_label_files):
    datalist = {}
    datalist["images"] = []
    datalist["labels"] = []
    for path, style_label_file in zip(paths, style_label_files):
        image_list = list(np.loadtxt(style_label_file, str, delimiter='\n'))
        currentClass = ""
        for filename in image_list:
            for cla in classes_list:
                if cla in filename:
                    currentClass = cla
                    filename = filename.replace(cla + ".", "")
                    break
            filename = path.format(filename)
            image = scipy.misc.imread(filename).astype(np.float)
            image = scipy.misc.imresize(image, [64, 64])

            datalist["images"].append(image)
            datalist["labels"].append(classes_list.index(currentClass))

    return datalist


def store_pikle_file(type, datalist):
    dataset_storage_location = os.path.join("/data/neotx", "brazilian_coffee_scenes" + "_"+type+".pickle")
    with open(dataset_storage_location, 'wb') as f:
        pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

training_dataset = create_file(paths_train, style_label_files_train)
testing_dataset = create_file(paths_test, style_label_files_test)

store_pikle_file("train", training_dataset)
store_pikle_file("test", testing_dataset)

data_convotor = DataConvertor(64, "brazilian_coffee_scenes", "/data/neotx")
data_convotor.create_from_pickle(True)
data_convotor.create_from_pickle(False)


