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
from scipy.spatial import distance

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
	# Create a list of class names
	class_names = [ cls.name.replace('_', ' ') for cls in train_dataset]
	# Saving classifier model
	classifier_filename_exp = os.path.expanduser(svm_model_path)
	with open(classifier_filename_exp, 'wb') as outfile:
	    pickle.dump((model, class_names), outfile)
	print('Saved classifier model to file "%s"' % classifier_filename_exp)

def thres_dist(train_embs, train_label, test_embs, test_label):
    thres = np.arange(0.6, 0.9, 0.02)
    best_acc = 0
    best_th = 0
    for th in thres:
        predict = []
        for i in range(test_embs.shape[0]):
            min_index = np.argmin([distance.euclidean(test_embs[i], train_embs[k]) for k in range(train_embs.shape[0])])
            # print(distance.euclidean(test_embs[i], train_embs[min_index]))
            if (distance.euclidean(test_embs[i], train_embs[min_index])) >= th:
                predict.append(5401)
            else:
                predict.append(train_label[min_index])
            # print(min_index , ":", test_label[min_index] ,":", test_label[i])
        acc = accuracy_score(predict, test_label)
        if (acc > best_acc):
            best_acc = acc
            best_th = th
    print (best_th, ": ", best_acc)
    return best_th, best_acc

def thresh_validate(embs, labels, thres, svm_model_path):
	classifier_filename_exp = os.path.expanduser(svm_model_path)
	unknown = max(test_labels) + 1
	with open(classifier_filename_exp, 'rb') as infile:
		(model, class_name) = pickle.load(infile)
	predictions = model.predict_proba(emb_array)
	best_class_indices = np.argmax(predictions, axis=1)
	#predict_class_indices = [pred for pred in best_class_indices if pred >= thres else unknown]

# predict one image
def predict(train_embs, train_labels, emb_test, model_path, thres_dis):
    min_dis = np.min([distance.euclidean(emb_test, train_embs[k]) for k in range(train_embs.shape[0])])
    if min_dis >= thres_dis:
        return train_embs.shape[0]
    else:
        classifier_filename_exp = os.path.expanduser(model_path)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_name) = pickle.load(infile)
        predictions = model.predict(emb_test)
        return predictions

def get_unknown(train_embs, train_labels, test_embs, thres_dis):
    index = []
    for i in range(test_embs.shape[0]):
        min_dis = np.min([distance.euclidean(test_embs[i], train_embs[k]) for k in range(train_embs.shape[0])])
        if min_dis >= thres_dis:
            index.append(i)
    return index

def validate_on_test(train_embs, train_labels, emb_tests, label_test, model_path, thres_dis):
    ## Get unknown
    index_unknown = get_unknown(train_embs, train_labels, emb_tests, thres_dis)
    predict = np.zeros(emb_tests.shape[0])
    predict[index_unknown] = 5401
    index_known = [k for k in range(emb_tests.shape[0]) if k not in index_unknown]

    emb_test_known = emb_tests[index_known]
    classifier_filename_exp = os.path.expanduser(model_path)
    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_name) = pickle.load(infile)
    predictions = model.predict(emb_test_known)
    predict[index_known] = predictions
    acc = accuracy_score(label_test, predict)
    print("Accuracy: %f" % acc)
    return index_unknown, acc

def save_feature(path, embs, labels):
	# filename_exp = os.path.expanduser(path)
	with open(path, 'w') as outfile:
	    pickle.dump((embs, labels), outfile)
	print('Saved data "%s"' % path)

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
	path_labels_csv = "../dataset/split_dataset_lfw/train.csv"
	facenet_model_path = "../model_lfw/20180402-114759/20180402-114759.pb"
	batch_size = 90
	image_size = 160
	refresh = False

	valid_dir = "../dataset/split_dataset_lfw/validset"
	emb_valid_save = "../dataset/split_dataset_lfw/save_emb/valid_emb.mat"
	# Extract feature
	if not os.path.exists(emb_valid_save) or refresh == True:
		print("Extract Validate data")
		validset = facenet.get_dataset(valid_dir)
		emb_valids, valid_labels = feature_extraction(validset, facenet_model_path, batch_size, 
														image_size, path_labels_csv)
		with open(emb_valid_save, 'w'): pass
		sio.savemat(emb_valid_save, mdict={"embs": emb_valids, "labels": valid_labels})

	test_dir = "../dataset/split_dataset_lfw/testset"
	emb_test_save = "../dataset/split_dataset_lfw/save_emb/test_emb.mat"
	# Extract feature
	if not os.path.exists(emb_test_save) or refresh == True:
		print("Extract Testing Data")
		testset = facenet.get_dataset(test_dir)
		emb_tests, test_labels = feature_extraction(testset, facenet_model_path, batch_size, 
														image_size, path_labels_csv)
		with open(emb_test_save, 'w'): pass
		sio.savemat(emb_test_save, mdict={"embs": emb_tests, "labels": test_labels})

	train_dir = "../dataset/split_dataset_lfw/trainset"
	emb_trains_save = "../dataset/split_dataset_lfw/save_emb/train_emb.mat"
	# Extract feature
	if not os.path.exists(emb_trains_save) or refresh == True:
		print("Extract Training Data")
		trainset = facenet.get_dataset(train_dir)
		emb_trains, train_labels = feature_extraction(trainset, facenet_model_path, batch_size, 
														image_size, path_labels_csv)
		with open(emb_trains_save, 'w'): pass
		sio.savemat(emb_trains_save, mdict={"embs": emb_trains, "labels": train_labels})

	valid_data = sio.loadmat(emb_valid_save)
	valid_embs = valid_data["embs"]
	valid_labels = valid_data["labels"][0]
	print (valid_embs.shape, "  :", len(valid_labels))

	test_data = sio.loadmat(emb_test_save)
	test_embs = test_data["embs"]
	test_labels = test_data["labels"][0]
	print (test_embs.shape, "  :", len(test_labels))

	train_data = sio.loadmat(emb_trains_save)
	train_embs = train_data["embs"]
	train_labels = train_data["labels"][0]
	print (train_embs.shape, "  :", len(train_labels))

	svm_model_path = "../dataset/split_dataset_lfw/model_svm/svm.ckpt"

	if not os.path.exists(svm_model_path) or refresh == True:
		trainset = facenet.get_dataset(train_dir)
		svm_training(trainset, train_embs, train_labels, valid_embs, valid_labels, svm_model_path)

	# get thres dist
	batch_size = 1600
	index = np.array(np.arange(len(train_labels)))
	np.random.shuffle(index)
	x = train_embs[index, :]
	y = train_labels[index]
	nrof_batches = int(math.ceil(1.0*len(train_labels) / batch_size))
	thresh = []
	acc = []
	for i in range(nrof_batches):
	    start_index = i*batch_size
	    end_index = min((i+1)*batch_size, len(train_labels))
	    best_th, best_acc = thres_dist(x[start_index:end_index, :], y[start_index:end_index],
	                                   valid_embs, valid_labels)
	    thresh.append(best_th)
	    acc.append(acc)
	print ("thresh to find unknown image: ", np.mean(thresh))

	thres_dis = np.mean(thresh)

	index_unknown, acc = validate_on_test(train_embs, train_labels, test_embs, test_labels, svm_model_path, thres_dis = 0.80)
