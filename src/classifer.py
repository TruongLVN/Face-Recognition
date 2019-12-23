import tensorflow.compat.v1 as tf
import numpy as np
import facenet
import os
import sys
import math
import pickle
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def feature_extraction(dataset, model_path, batch_size, image_size, path_labels_csv):
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with tf.Session() as sess:
			
			# Check that there are at least one training image per class
			for cls in dataset:
				assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')

			paths, labels = facenet.get_image_paths_and_labels(dataset, path_labels_csv)
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

			    images = facenet.load_data(paths_batch, False, False, image_size)  #images shape [10, 160, 160, 3]
			    # feed dictionary
			    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
			    # extract feature of images
			    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
			return  emb_array, labels

def svm_training(train_dataset, emb_trains, train_labels, emb_valids, valid_labels, svm_model_path):

	print('Training classifier')

	model = SVC(kernel='linear', probability=True)
	model.fit(emb_trains, train_labels)

	# Evaluate model
	predict = model.predict(emb_valids)
	acc = accuracy_score(valid_labels, predict)
	print("Accuracy: %f" % acc)
	print(valid_labels)
	print(predict)
	# Create a list of class names
	class_names = [ cls.name.replace('_', ' ') for cls in train_dataset]
	# Saving classifier model
	classifier_filename_exp = os.path.expanduser(svm_model_path)
	with open(classifier_filename_exp, 'wb') as outfile:
	    pickle.dump((model, class_names), outfile)
	print('Saved classifier model to file "%s"' % classifier_filename_exp)

def get_threshold_proba(embs, labels, batch_size, svm_model_path):
	print("Calculating threshold probability")
	classifier_filename_exp = os.path.expanduser(svm_model_path)
	with open(classifier_filename_exp, 'rb') as infile:
		(model, class_name) = pickle.load(infile)

	print('Loaded classifier model from file "%s"' % classifier_filename_exp)
	# shuffle data
	index = np.array(np.arange(np.shape(labels)[0])) 
	np.random.shuffle(index)
	embs_t = embs[index, :]
	labels_t = labels[index]
	# match
	predictions = model.predict_proba(embs_t) 
	best_class_indices = np.argmax(predictions, axis=1)
	best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
	# Caculate threshold_proba
	thres_proba = 0
	nrof_images = len(labels_t)
	nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
	for i in range(nrof_batches):
		start_index = i*batch_size
		end_index = min((i+1)*batch_size, nrof_images)
		batch_label_true = labels_t[start_index:end_index]
		batch_label_predict = best_class_indices[start_index:end_index]
		batch_proba = best_class_probabilities[start_index:end_index]
		# Find the max threshold probability of each batch that must give the highest accuracy prediction.
		thres_proba += np.min(batch_proba[batch_label_true == batch_label_predict])
		print ("thres_batch:", np.min(batch_proba[batch_label_true == batch_label_predict]))
	thres_proba /= nrof_batches
	return thres_proba

def thresh_validate(embs, labels, thres, svm_model_path):
	classifier_filename_exp = os.path.expanduser(svm_model_path)
	unknown = max(test_labels) + 1
	with open(classifier_filename_exp, 'rb') as infile:
		(model, class_name) = pickle.load(infile)
	predictions = model.predict_proba(emb_array)
	best_class_indices = np.argmax(predictions, axis=1)
	#predict_class_indices = [pred for pred in best_class_indices if pred >= thres else unknown]

def save_feature(path, embs, labels):
	# filename_exp = os.path.expanduser(path)
	with open(path, 'w') as outfile:
	    pickle.dump((embs, labels), outfile)
	print('Saved data "%s"' % path)

def get_thres(emb_array, labels, path_SVM):
	num_embs = len(labels) 
	classifier_filename_exp = os.path.expanduser(path_SVM) 
	thres = np.arange(0, 1, 0.001) 
	with open(classifier_filename_exp, 'rb') as infile:
		(model, class_names) = pickle.load(infile) 
		predictions = model.predict_proba(emb_array) 
		best_class_indices = np.argmax(predictions, axis=1) 
		acc_max = 0 
		best_thres = 0 
		for i in thres:
			new_indices = np.zeros(num_embs)
			acc = 0 
			for j in range(num_embs):
				label_true = labels[j] 
				label_predict = best_class_indices[j] 
				proba = predictions[j, label_predict] 
				if (proba >= i) and (label_predict == label_true):
					acc = acc + 1
			if acc >= acc_max:
				acc_max = acc 
				best_thres = i 
	return best_thres

if __name__=="__main__":
	# vggface2_model_path = "../model/20180402-114759/20180402-114759.pb"			# change following your dir
	# svm_model_path = "../model/svm/svm.ckpt"
	# train_dir = "../dataset/split_dataset/trainset"								# change following your dir
	# valid_dir = "../dataset/split_dataset/validset"
	# test_dir = "../dataset/split_dataset/testset"
	# path_labels_csv = "../dataset/train.csv"
	# trainset = facenet.get_dataset(train_dir)
	# validset = facenet.get_dataset(valid_dir)
	# testset = facenet.get_dataset(test_dir)
	# batch_size = 10
	# image_size = 160

	# # Extract feature
	# emb_trains, train_labels = feature_extraction(trainset, vggface2_model_path, batch_size, image_size)
	# emb_valids, valid_labels = feature_extraction(validset, vggface2_model_path, batch_size, image_size, path_labels_csv)
	# print ("ákdakldjlasd")
	# emb_tests, test_labels = feature_extraction(testset, vggface2_model_path, batch_size, image_size)
	# print ("after extract")
	# print("train_labels", train_labels)
	# print("valid_labels", valid_labels)
	# print("test_labels", test_labels)
	# # Save feature, labels and override old data 
	# save_feature("../dataset/feature_labels/train_emb.dat", emb_trains, train_labels)
	# save_feature("../dataset/feature_labels/valid_emb.dat", emb_valids, valid_labels)
	# save_feature("../dataset/feature_labels/test_emb.dat", emb_tests, test_labels)

	# svm_training(trainset, emb_trains, train_labels, emb_valids, valid_labels, svm_model_path)

	# x = np.concatenate((emb_trains, emb_valids), axis=0)
	# y = np.concatenate((train_labels, valid_labels), axis=0)
	# batch_size = 20
	# # cua tui
	# thres = get_threshold_proba(x, y, batch_size, svm_model_path)
	# print("thres_proba:", thres)
	# print("success")
	# print(train_labels)
	# print(".................................")
	# print(emb_trains.shape)

	##########################################LFW##########################################################3333
	path_labels_csv = "../dataset/train.csv"
	facenet_model_path = "../model_lfw/20180402-114759/20180402-114759.pb"
	batch_size = 50
	image_size = 160
	refresh = False

	valid_dir = "../dataset/split_dataset_lfw/validset"
	emb_trains_save = "../dataset/split_dataset_lfw/save_emb/train_emb.mat"
	# Extract feature
	if not os.path.exists(emb_trains_save) or refresh == True:
		print("Extract again again ne")
		validset = facenet.get_dataset(valid_dir)
		emb_valids, valid_labels = feature_extraction(validset, facenet_model_path, batch_size, 
														image_size, path_labels_csv)

		sio.savemat(emb_trains_save, mdict={"embs": emb_valids, "labels": valid_labels})

	valid_data = sio.loadmat(emb_trains_save)
	valid_embs = valid_data["embs"]
	valid_labels = valid_data["labels"]

	print (valid_labels[0])
	print (np.sum(valid_embs[12]))