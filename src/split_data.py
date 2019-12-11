import os
import numpy as np
from scipy import misc

def copy_image(source, des):
	for path_img_in in source:
		name_img = os.path.split(path_img_in)[1]
		print(name_img)
		path_img_out = os.path.join(des, name_img)
		image = misc.imread(path_img_in)
		misc.imsave(path_img_out, image)

def split_data(path_data_set, path_train, path_test, path_valid):
	if not os.path.exists(path_data_set):
		sys.exit("Invalid path of data set!!")
	# Get folder contain data
	path_groups = [os.path.join(path_data_set, group) for group in os.listdir(path_data_set) if os.path.splitext(group)[1]!='.txt']
	for path_group in path_groups:
		ids = os.path.split(path_group)[1]
		train_ids = os.path.join(path_train, ids)
		test_ids = os.path.join(path_test, ids)
		valid_ids = os.path.join(path_valid, ids)
		
		path_imgs = [os.path.join(path_group, image) for image in os.listdir(path_group)]
		np.random.shuffle(path_imgs)

		if not os.path.exists(train_ids):
			os.makedirs(train_ids)
			img_trains = path_imgs[0:30]
			copy_image(img_trains, train_ids)
		if not os.path.exists(test_ids):
			os.makedirs(test_ids)
			img_tests = path_imgs[30:40]
			copy_image(img_tests, test_ids)
		if not os.path.exists(valid_ids):
			os.makedirs(valid_ids)
			img_valids = path_imgs[40:50]
			copy_image(img_valids, valid_ids)

if __name__=="__main__":
	path_data_set = "../dataset/aligned_dataset"
	path_train = "../dataset/split_dataset/trainset"
	path_test = "../dataset/split_dataset/testset"
	path_valid = "../dataset/split_dataset/validset"
	split_data(path_data_set, path_train, path_test, path_valid)



