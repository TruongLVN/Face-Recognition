import os
import numpy as np
import csv

path_data = "../dataset/split_dataset_lfw/trainset"
path_file_csv = "../dataset/split_dataset_lfw/train.csv"
if __name__ == '__main__':
	if not os.path.exists(path_data):
		sys.exit("Invalid path of data set!!")
	names = [name for name in os.listdir(path_data) if os.path.splitext(name)[1]!='.txt']
	labels = np.arange(0, len(names))
	dictionary = dict(zip(names, labels))

	with open(path_file_csv, 'w', newline="") as csv_file:  
	    writer = csv.writer(csv_file)
	    for key, value in dictionary.items():
	        writer.writerow([key, value])
