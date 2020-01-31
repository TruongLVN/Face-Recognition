# Face-Recognition Project
## Introduction
* This is project to learn using Deep Face Recognition technique.
* I have implemented this technique on [LFW dataset](http://vis-www.cs.umass.edu/lfw/) and have built a simple propram to recognize people on realtime.

## Main Process
* Step 1. Using [MTCNN](https://github.com/TruongLVN/Face-Recognition/tree/master/model/mtcnn) to aligned datasets. You can refer this techique at [paper](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).
* Step 2. Using [20180402-114759](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) model to extract feature of datasets. This is a pre-trained models of [Inception-ResNet v1](https://arxiv.org/pdf/1602.07261v1.pdf) architecture and has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.
* Step 3. Using [Linear Support Vector Machine(SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) classifer to recognize face.

* Note: To get unknown face on test datasets, i using threshold method. First, drawing probability distribution of 2 variables(the euclidean distance between 2 images if they belong to the same face, and the euclidean distance between 2 images if they are on different faces). Then, select the value of the intersection between the two distributions as the threshold.

<p align="center">
  <img src="https://github.com/TruongLVN/Face-Recognition/blob/master/model/thresh.png" width="300" alt="accessibility text">
</p>