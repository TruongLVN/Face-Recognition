import tensorflow.compat.v1 as tf
import numpy as np
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC


def feature_extraction(data_dir, model_path, batch_size, image_size):
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with tf.Session() as sess:
			
			# Check that there are at least one training image per class
			dataset = facenet.get_dataset(data_dir)
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

if __name__=="__main__":
	model_path = "../model/20180402-114759/20180402-114759.pb"
	train_dir = "../pre_dataset/dataset/testset"
	emb_array, labels = feature_extraction(train_dir, model_path, 10, 160)

	print("success")
	print(labels)
	print(".................................")
	print(emb_array.shape)





		
