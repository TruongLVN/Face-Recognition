import os
import numpy as np
from scipy import misc
np.random.seed(7)

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
	path_names = [os.path.join(path_data_set, name) for name in os.listdir(path_data_set) if os.path.splitext(name)[1]!='.txt']
	print("number of people", len(path_names))
	for path_name in path_names:
		name = os.path.split(path_name)[1]

		# Create dataset name
		train_name = os.path.join(path_train, name)
		test_name = os.path.join(path_test, name)
		valid_name = os.path.join(path_valid, name)
		
		# Get path image
		path_imgs = [os.path.join(path_name, image) for image in os.listdir(path_name)]
		np.random.shuffle(path_imgs)

		# Case number of image on each people < 7
		# Split randomly 
		if (6<len(path_imgs)<8):
			print("len of path image", len(path_imgs))

			x = np.random.rand(len(path_imgs))

			if not os.path.exists(train_name):
				os.makedirs(train_name)
				img_trains = [path_imgs[i] for i in np.where(x<0.5)[0]]
				copy_image(img_trains, train_name)

			if not os.path.exists(test_name):
				os.makedirs(test_name)
				img_tests = [path_imgs[i] for i in np.where((0.5<= x) & (x <0.7))[0]]
				copy_image(img_tests, test_name)

			if not os.path.exists(valid_name):
				os.makedirs(valid_name)
				img_valids = [path_imgs[i] for i in np.where(x>=0.7)[0]]
				copy_image(img_valids, valid_name)
if __name__=="__main__":
	path_data_set = "../dataset/align_lfw"
	path_train = "../dataset/split_dataset_lfw/trainset"
	path_test = "../dataset/split_dataset_lfw/testset"
	path_valid = "../dataset/split_dataset_lfw/validset"
	split_data(path_data_set, path_train, path_test, path_valid)



