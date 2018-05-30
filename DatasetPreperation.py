from convert_to_tf_record import DataConvertor

dataset_path_train = "/data/data/uc_train_256_data/**.jpg"
#dataset_path_train = "/data/nwpu_resisc45_train/**/**.jpg"
#dataset_path_train = "/data/pattennet_train/**/**.jpg"
dataset_path_test = "/data/data/uc_test_256/**.jpg"
#dataset_path_test = "/data/nwpu_resisc45_test/**/*.jpg"
#dataset_path_test = "/data/pattennet_test/**/*.jpg"
dataset_storage_location = "/data/neotx"
#dataset_name = "nwpu_resisc45"
dataset_name = "ucdataset"
image_size = 64
batch_size = 64

data_convotor = DataConvertor(image_size, dataset_name, dataset_storage_location)
data_convotor.convert_into_tfrecord(dataset_path_train, True)
data_convotor.convert_into_tfrecord(dataset_path_test, False)
#images, one_hot_labels, sss, ss = data_convotor.provide_data(batch_size, 'train')
#print("----------------------")