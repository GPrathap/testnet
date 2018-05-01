from convert_to_tf_record import DataConvertor

classesList = ["agricultural", 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral'
        , 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential',
                   'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                   'storagetanks', 'tenniscourt']

dataset_path_train = "/data/images/uc_train_256_data/**.jpg"
dataset_path_test = "/data/images/uc_test_256/**.jpg"
dataset_storage_location = "/data/neotx"
dataset_name = "ucdataset"
image_size = 64
batch_size = 64

data_convotor = DataConvertor(classesList, image_size, dataset_name, dataset_storage_location)
data_convotor.convert_into_tfrecord(dataset_path_train, True)
data_convotor.convert_into_tfrecord(dataset_path_test, False)
#images, one_hot_labels, sss, ss = data_convotor.provide_data(batch_size, 'train')
#print("----------------------")