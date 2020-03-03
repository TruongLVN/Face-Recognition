import os
import numpy as np
import shutil
np.random.seed(7)

def copy_image(source, path_des):
	if not os.path.exists(path_des):
		os.makedirs(path_des)
		for path_img_in in source:
			name_img = os.path.split(path_img_in)[1]
			print(name_img)
			path_img_out = os.path.join(path_des, name_img)
			shutil.copyfile(path_img_in, path_img_out)

def split_data(path_data_set, path_train, path_test, path_valid):
	if not os.path.exists(path_data_set):
		sys.exit("Invalid path of data set!!")
	# Get folder contain data
	path_names = [os.path.join(path_data_set, name) for name in os.listdir(path_data_set) if os.path.splitext(name)[1]!='.txt']
	print("number of people", len(path_names))
	np.random.shuffle(path_names)

	#count_x to count classes.
	count_1 = 0
	count_2 = 0
	count = 0
	for path_name in path_names:
		name = os.path.split(path_name)[1]
		# Create dataset name
		train_name = os.path.join(path_train, name)
		test_name = os.path.join(path_test, name)
		valid_name = os.path.join(path_valid, name)
		# Get path image
		path_imgs = [os.path.join(path_name, image) for image in os.listdir(path_name)]
		np.random.shuffle(path_imgs)
		count += len(path_imgs)
		# Case number of image on each people < 7
		# Split randomly

		# Split classes have number_of_data(n) with n <3
		if len(path_imgs) < 3:
			if(count_1 < 4500):
				copy_image(path_imgs, train_name)
			elif( count_1 < 4620):
				copy_image(path_imgs, valid_name)
			else:
				copy_image(path_imgs, test_name)
			count_1 += 1

		# Split classes have number_of_data(n) with 3 <= n < 10
		elif len(path_imgs) < 10:
			copy_image(path_imgs, train_name)

		# Split classes have number_of_data(n) with 10 <= n < 20
		elif len(path_imgs) < 20:
			copy_image(path_imgs[0:-4], train_name)

			if count_2 >= 17:
				copy_image(path_imgs[-4:], test_name)
			else:
				copy_image(path_imgs[-4:], valid_name)
			count_2 += 1

		# Split classes have number_of_data(n) with n >= 20	
		else:
			a1 = int(np.ceil(len(path_imgs)*5/9))
			a2 = int(np.ceil(len(path_imgs)*8/9))
			copy_image(path_imgs[0:a1], train_name)
			copy_image(path_imgs[a1:a2], test_name)
			copy_image(path_imgs[a2:], valid_name)

	print("Total", count)
def count_image(paths):
	names = [os.path.join(paths, name) for name in os.listdir(paths)]
	count = 0
	# for name in names
	for name in names:
		imgs = [img for img in os.listdir(name)]
		count += len(imgs)
	return count
if __name__=="__main__":
	path_data_set = "../dataset/align_lfw"
	path_train = "../datasettest/split_dataset_lfw/trainset"
	path_test = "../datasettest/split_dataset_lfw/testset"
	path_valid = "../datasettest/split_dataset_lfw/validset"
	split_data(path_data_set, path_train, path_test, path_valid)
	a = count_image(path_valid)
	b = count_image(path_test)
	c = count_image(path_train)

	print ("number image of train: ", c)
	print ("number image of test: ", b)
	print ("number image of valid: ", a)
	print ("Total: ", a+b+c)


