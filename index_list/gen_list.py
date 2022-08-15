import numpy as np
import os
from scipy import io



index = np.arange(1,10001)
index2 = np.arange(1,10001) + 10000
for i in range(1,4):
	os.makedirs(str(i), exist_ok = True)

	np.random.shuffle(index)

	train_list1 = index[:4000]
	train_list = np.append(train_list1, index2)
	val_list = index[4000:5000]
	test_list = index[5000:10000]
	# train_and_val_list = index[:10000]
	# test_list = index[10000:20001]
	# retrain_train_list = test_list[:5000]
	# retrain_test_list = test_list[5000:]
	all_list = np.append(index, index2)

	save_file_train = './{}/train_list_14k.npy'.format(i)
	save_file_val = './{}/valid_list_1k.npy'.format(i)
	save_file_test = './{}/test_list_5k.npy'.format(i)

	save_file_all= './{}/all_list_2w.npy'.format(i)
	save_file_all_mat = './{}/all_list_2w.mat'.format(i)

	np.save(save_file_train, train_list)
	np.save(save_file_val, val_list)
	np.save(save_file_test, test_list)

	np.save(save_file_all, all_list)
	
	io.savemat(save_file_all_mat, {'all_list': all_list,'index': index, 'train':train_list, 'train_5k': index[:5000], 'test':test_list})



