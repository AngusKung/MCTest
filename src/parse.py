import numpy as np

def mk_all(data):
	data_num = len(data)
	word_dim = len(data[0][0][0][0])
	print "Total data: %d\nWord dim:  %d\n" % (data_num, word_dim)

	label_size = len(data[0][1])
	#data[0][0] contains P,Q,A1,A2,A3,A4
	#data[0][1] contains label
	#data[0][0][0] contains P is a list of words
	#data[0][0][0][0] is word's np vector dim=300

	data_shuf = []
	label_shuf = np.zeros((data_num, label_size))
	for i in range(data_num):
		fuck=[]
		for j in range(len(data[i][0])):
			fuck=fuck+data[i][0][j]
		data_shuf.append(fuck)
		label_shuf[i, ] = data[i][1]
	return data_shuf, label_shuf

def mk_batch(data, batch_size, shuffle=True):
	data_num = len(data)
	data_size = len(data[0][0])
	print "Total data: %d\nData size:  %d\n" % (data_num, data_size)
	n_batch = data_num/batch_size
	n_left_data = data_num % batch_size
	if(n_left_data != 0):
		n_batch += 1
	batch_data = np.zeros((n_batch, batch_size, data_size))

	label_size = len(data[0][1])
	batch_label = np.zeros((n_batch, batch_size, label_size))

	data_shuf = np.zeros((data_num, data_size))
	label_shuf = np.zeros((data_num, label_size))
	for i in range(data_num):
		data_shuf[i, ] = data[i][0] 
		label_shuf[i, ] = data[i][1]
	if(shuffle):
		idx_shuffle = np.arange(data_num)
		np.random.shuffle(idx_shuffle)
		i=0
		for i_shuf in idx_shuffle:
			data_shuf[i, :] = data[i_shuf][0]
			label_shuf[i] = data[i_shuf][1]
			i+=1

	# bad implementation, need modification...
	if(n_left_data != 0):
		for i in range(batch_size - n_left_data):
			data_shuf = np.concatenate((data_shuf, np.asarray([data_shuf[i]])))
			label_shuf = np.concatenate((label_shuf, np.asarray([label_shuf[i]])))

	for i in range(n_batch):
		batch_data_matrix = np.zeros((batch_size, data_size))
		batch_label_matrix = np.zeros((batch_size, label_size))
		for j in range(batch_size):
			idx = j + i * batch_size
			batch_data_matrix[j, :] = data_shuf[idx]
			batch_label_matrix[j, :] = label_shuf[idx]
		batch_data[i, :] = batch_data_matrix
		batch_label[i, :] = batch_label_matrix
		del batch_data_matrix
		del batch_label_matrix
	return batch_data, batch_label


if __name__ == "__main_":
	pass
