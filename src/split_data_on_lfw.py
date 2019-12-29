import os
import numpy as np
import shutil
np.random.seed(7)

def copy_image(source, des):
	for path_img_in in source:
		name_img = os.path.split(path_img_in)[1]
		print(name_img)
		path_img_out = os.path.join(des, name_img)
		shutil.copyfile(path_img_in, path_img_out)

def split_data(path_data_set, path_train, path_test, path_valid):
	if not os.path.exists(path_data_set):
		sys.exit("Invalid path of data set!!")
	# Get folder contain data
	path_names = [os.path.join(path_data_set, name) for name in os.listdir(path_data_set) if os.path.splitext(name)[1]!='.txt']
	print("number of people", len(path_names))
	np.random.shuffle(path_names)
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
		if(len(path_imgs) < 3):
			if not os.path.exists(train_name) and count_1 < 4500:
				os.makedirs(train_name)
				copy_image(path_imgs, train_name)

			if not os.path.exists(valid_name) and count_1 < 4620 and count_1 >= 4500:
				os.makedirs(valid_name)
				copy_image(path_imgs, valid_name)
				
			if not os.path.exists(test_name) and count_1 >= 4620:
				os.makedirs(test_name)
				copy_image(path_imgs, test_name)
			count_1 += 1

		# if(len(path_imgs) == 2):
		# 	if not os.path.exists(train_name):
		# 		os.makedirs(train_name)
		# 		if(count_2 < 650):
		# 			copy_image(path_imgs, train_name)
		# 		else:
		# 			copy_image([path_imgs[0]], train_name)
					
		# 	if not os.path.exists(test_name) and count_2 >= 679:
		# 		os.makedirs(test_name)
		# 		copy_image([path_imgs[1]], test_name)

		# 	if not os.path.exists(valid_name) and count_2 >= 650 and count_2 < 679:
		# 		os.makedirs(valid_name)
		# 		copy_image([path_imgs[1]], valid_name)
		# 	count_2 += 1
		if (len(path_imgs) >= 3 and len(path_imgs) < 10):
			if not os.path.exists(train_name):
				os.makedirs(train_name)
				copy_image(path_imgs, train_name)

		if (len(path_imgs) >= 10 and len(path_imgs) < 20):
			if not os.path.exists(train_name):
				os.makedirs(train_name)
				copy_image(path_imgs[0:-4], train_name)
			if not os.path.exists(test_name) and count_2 >= 17:
				os.makedirs(test_name)
				copy_image(path_imgs[-4:], test_name)

			if not os.path.exists(valid_name) and count_2 < 17:
				os.makedirs(valid_name)
				copy_image(path_imgs[-4:], valid_name)
			count_2 += 1

		if (len(path_imgs) >= 20):
			a1 = int(np.ceil(len(path_imgs)*5/9))
			a2 = int(np.ceil(len(path_imgs)*8/9))

			if not os.path.exists(train_name):
				os.makedirs(train_name)
				copy_image(path_imgs[0:a1], train_name)
			if not os.path.exists(test_name):
				os.makedirs(test_name)
				copy_image(path_imgs[a1:a2], test_name)
			if not os.path.exists(valid_name):
				os.makedirs(valid_name)
				copy_image(path_imgs[a2:], valid_name)

		# if (6<len(path_imgs)<8):
		# 	print("len of path image", len(path_imgs))

		# 	x = np.random.rand(len(path_imgs))

		# 	if not os.path.exists(train_name):
		# 		os.makedirs(train_name)
		# 		img_trains = [path_imgs[i] for i in np.where(x<0.5)[0]]
		# 		copy_image(img_trains, train_name)

		# 	if not os.path.exists(test_name):
		# 		os.makedirs(test_name)
		# 		img_tests = [path_imgs[i] for i in np.where((0.5<= x) & (x <0.7))[0]]
		# 		copy_image(img_tests, test_name)

		# 	if not os.path.exists(valid_name):
		# 		os.makedirs(valid_name)
		# 		img_valids = [path_imgs[i] for i in np.where(x>=0.7)[0]]
		# 		copy_image(img_valids, valid_name)
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
	path_train = "../dataset/split_dataset_lfw/trainset"
	path_test = "../dataset/split_dataset_lfw/testset"
	path_valid = "../dataset/split_dataset_lfw/validset"
	split_data(path_data_set, path_train, path_test, path_valid)
	a = count_image(path_valid)
	b = count_image(path_test)
	c = count_image(path_train)

	print ("number image of train: ", c)
	print ("number image of test: ", b)
	print ("number image of valid: ", a)
	print ("Total: ", a+b+c)


