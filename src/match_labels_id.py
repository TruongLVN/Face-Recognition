# import csv
# import os
# import facenet
# path = "../data_lfw/train.csv"
# fodel = "../dataset/aligned_dataset"
# if __name__=="__main__":
# 	with open(path) as csv_file:
# 		csv_reader = csv.reader(csv_file, delimiter=',')
# 		line_count = 0
# 		for row in csv_reader:
# 			if line_count == 0:
# 				print('column name are', row)
# 			line_count += 1
# 	kk = os.listdir(fodel)
# 	id_ = os.listdir(fodel)
# 	train_dir = "../dataset/raw_dataset"
# 	trainset = facenet.get_dataset(train_dir)
# 	