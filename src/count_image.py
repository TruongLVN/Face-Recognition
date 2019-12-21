import os
import numpy as np

path_data = "../dataset/align_lfw"
if __name__ == '__main__':
	if not os.path.exists(path_data):
		sys.exit("Invalid path of data set!!")

	path_groups = [os.path.join(path_data, group) for group in os.listdir(path_data) if os.path.splitext(group)[1]!='.txt']

	soluong = []
	sosoluong = np.array([])
	for path_group in path_groups:
		img = os.listdir(path_group)

		if(len(img) not in soluong):	
			soluong.append(len(img))   # add different numbers
			sosoluong = sosoluong.tolist()
			sosoluong.append(0)
			sosoluong = np.array(sosoluong)
		# count same numbers of number
		sosoluong += np.array(soluong) == len(img)
	tb = np.sum(sosoluong*np.array(soluong))/np.sum(sosoluong)

	# sort
	soluong = np.array(soluong)
	indices = np.argsort(soluong)

	soluong = soluong[indices]
	sosoluong = sosoluong[indices]

	print(".........numbers of image............")
	print(soluong)
	print(".........number of folder correspond to numbers of image............")
	print(sosoluong)
	print("so buc anh trung binh", tb)
	print(np.sum(sosoluong))