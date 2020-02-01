# Face-Recognition Project
## Introduction
* This is project learn how to using Deep Face Recognition technique.
* I have implemented this technique on [LFW dataset](http://vis-www.cs.umass.edu/lfw/) and have built a simple propram to recognize people on realtime.

## Main Process
* Step 1. Using [MTCNN](https://github.com/TruongLVN/Face-Recognition/tree/master/model/mtcnn) to aligned datasets. You can refer this techique at [paper](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).
* Step 2. Using [20180402-114759](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) model to extract feature of datasets. This is a pre-trained models of [Inception-ResNet v1](https://arxiv.org/pdf/1602.07261v1.pdf) architecture and has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.
* Step 3. Using [Linear Support Vector Machine(SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) classifer to recognize face.

## Running on LFW Dataset
	The code is using tensorflow 1.15.0 with python 2.7 and 3.7.
1. Aliging and cropping faces

		python alignment_face.py

2. Split datasets into 3 sets: trainset(11127 images), testset(1590 images with 270 unknown images), validset(516 images with 141 unknown images).

		python alignment_face.py

3. Assign lable (0 ~ N-1) to trainset and export svc file.

		python export_labels_to_svc.py

4. Extract feature, train data and evaluate on testset.

		python classifer.py

* Note: I used 2 method to get a threshold(euclidean distance between 2 feature) for identifying unknown people.

1. To get unknown face on test datasets, i using threshold method. First, drawing probability distribution of 2 variables(the euclidean distance between 2 images if they belong to the same face, and the euclidean distance between 2 images if they are on different faces). Then, select the value of the intersection between the two distributions as the threshold.

<p align="center">
  <img src="https://github.com/TruongLVN/Face-Recognition/blob/master/model/thresh.png" width="400" alt="accessibility text">
</p>

		Choose threshold = 1.1

		Accuracy = 0,84 (evaluate on testset).

2. Split trainset into 7 batch(each batch 1600 image). Then evaluate each batch on validsets to get best threshold(0.6-1.2) with highest accuracy. Finaly, take the average of best-thresholds. Choose threshold = 0.802

		Accuracy = 0,9487 (evaluate on testset).

## Running to recognize people on realtime.

