from scipy import misc
import os
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep

class PreProcessor:
    def __init__(self, image_size=182 , margin=44, random_order=True, gpu_memory_fraction=0.5, 
                detect_multiple_faces=False, model_path=None):
        # minimum size of faces 
        self.minsize = 20 
        # three step's threshold of mtcnn
        self.threshold = [0.6, 0.7, 0.7]
        # scale factor 
        self.factor = 0.709 
        self.margin = margin
        self.image_size = image_size
        self.random_order = random_order
        self.detect_multiple_faces = detect_multiple_faces
        self.gpu_memory_fraction = gpu_memory_fraction

        # Create graph
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, 
                                                                            log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, model_path)
    
    def align(self, input_dir, output_dir):
        sleep(random.random())
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
        # Get dataset
        dataset = facenet.get_dataset(input_dir)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            if self.random_order:
                random.shuffle(dataset)
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if self.random_order:
                        random.shuffle(cls.image_paths)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename+'.png')
                    print(image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim<2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            img = img[:,:,0:3]
                            bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, 
                            												self.onet, self.threshold, self.factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('detected_face: %d' % nrof_faces)
                            if nrof_faces>0:
                                det = bounding_boxes[:,0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces>1:             
                                    if self.detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([(det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0]])
                                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                        det_arr.append(det[index,:])
                                else:
                                    det_arr.append(np.squeeze(det))
                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0]-self.margin/2, 0)
                                    bb[1] = np.maximum(det[1]-self.margin/2, 0)
                                    bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                                    bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                    scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                                    nrof_successfully_aligned += 1
                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    if self.detect_multiple_faces:
                                        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                    else:
                                        output_filename_n = "{}{}".format(filename_base, file_extension)
                                    misc.imsave(output_filename_n, scaled)
                                    text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                            else:
                            	print ('Unable to align %s' % image_path)
                            	text_file.write('%s\n' % (output_filename))
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

if __name__ == "__main__":
	aligner = PreProcessor(model_path="../model/mtcnn")
	aligner.align("../dataset/raw_data_lfw", "../dataset/align_lfw")