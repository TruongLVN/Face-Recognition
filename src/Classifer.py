import tensorflow.compat.v1 as tf
import numpy as np
import facenet
import os
import sys
import math
import pickle
import scipy.io as sio
from sklearn.svm import SVC


def feature_extraction(dataset, model_path, batch_size, image_size):
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with tf.Session() as sess:
			
			# Check that there are at least one training image per class
			for cls in dataset:
				assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')

			paths, labels = facenet.get_image_paths_and_labels(dataset)
			print('Number of classes: %d' % len(dataset))
			print('Number of images: %d' % len(paths))

			# load model
			print('Loading feature extraction model...')
			facenet.load_model(model_path)

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")   # last fully connected layer 
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]
			print("size last connected layer : %d", embedding_size)

			# Run forward pass to calculate embeddings
			print('Calculating features for images')
			nrof_images = len(paths)
			nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, embedding_size)) 

			for i in range(nrof_batches_per_epoch):
			    start_index = i*batch_size
			    end_index = min((i+1)*batch_size, nrof_images)
			    paths_batch = paths[start_index:end_index]		# path of images in batch 
			    images = facenet.load_data(paths_batch, False, False, image_size)
			    # feed dictionary
			    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
			    # extract feature of images
			    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
			return  emb_array, labels

def train_svm(dataset, emb_trains, train_labels, emb_valids, valid_labels,path_svm):

	print('Training classifier')

	model = SVC(kernel='linear', probability=True)
	model.fit(emb_array, labels)

	# Evaluate model
	predict = model.predict(emb_valids)
	acc = accuracy_score(valid_labels, predict)
	print("Accuracy: %f" % acc)

	# Create a list of class names
	class_names = [ cls.name.replace('_', ' ') for cls in dataset]
	# Saving classifier model
	classifier_filename_exp = os.path.expanduser(path_svm)
	with open(classifier_filename_exp, 'wb') as outfile:
	    pickle.dump((model, class_names), outfile)
	print('Saved classifier model to file "%s"' % classifier_filename_exp)

def save_feature(path, embs, labels):
	filename_exp = os.path.expanduser(path)
	with open(filename_exp, 'wb') as outfile:
	    pickle.dump((embs, labels), outfile)
	print('Saved data "%s"' % filename_exp)

if __name__=="__main__":
	model_path = "../model/20180402-114759/20180402-114759.pb"			# change following your dir
	train_dir = "../raw_dataset/dataset/trainset"						# change following your dir
	valid_dir = "../raw_dataset/dataset/validdir"
	test_dir = "../raw_dataset/dataset/testset"
	# get dataset
	trainset = facenet.get_dataset(train_dir)
	validset = facenet.get_dataset(valid_dir)
	testset = facenet.get_dataset(test_dir)
	batch_size = 10
	image_size = 160

	# Extract feature
	emb_trains, train_labels = feature_extraction(trainset, model_path, batch_size, image_size)
	emb_valids, valid_labels = feature_extraction(validset, model_path, batch_size, image_size)
	emb_tests, test_labels = feature_extraction(testset, model_path, batch_size, image_size)
	# Save feature, labels and override old data 
	save_feature("../raw_dataset/dataset/testset", emb_trains, train_labels)
	save_feature("../raw_dataset/dataset/testset", emb_valids, valid_labels)
	save_feature("../raw_dataset/dataset/testset", emb_tests, test_labels)


	print("success")
	print(train_labels)
	print(".................................")
	print(emb_train.shape)