import sys
import numpy
import timeit
import cPickle
import argparse
import random
import pdb
from DeepNNet import DNN
from parse import mk_batch
import csv

LEARNING_RATE = 0.001
HIDDEN_LAYER=5
HIDDEN_LAYER_DIM=2048

def my_print(epoch, ith_batch, n_batch, cost):
	sys.stdout.write("\repoch %i, batch: %i/%i, cost: %f" % \
				(epoch, ith_batch, n_batch, cost))
	sys.stdout.flush()

def train():
	parser = argparse.ArgumentParser()
	parser.add_argument("-dataset", type=str, default="mod1")
	parser.add_argument("-batch_size", type=int, default=1)
	parser.add_argument("-training_epochs", type=int, default=100)
	args = parser.parse_args()

	print "Loading MCTest training data...\n",
	# len(data) == total num of data
	# traing_data[i] == [input_vec, label]
	input_dim=1800
	training_data = cPickle.load(open("Pickle/mc500.train.mod1.pickle"))
	valid_data = cPickle.load(open("Pickle/mc500.dev.mod1.pickle"))
	pdb.set_trace()
	if(args.dataset=="mod2"):
		input_dim=4500
		training_data = cPickle.load(open("Pickle/mc500.train.mod2.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))
	elif(args.dataset=="mod2.x24"):
		input_dim=4500
		training_data = cPickle.load(open("Pickle/mc500.train.mod2.x24.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))
	elif(args.dataset=="mod1.x24"):
		input_dim=1800
		training_data = cPickle.load(open("Pickle/mc500.train.mod1.x24.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod1.pickle"))

	batch_training_data, batch_train_label = mk_batch(training_data,
										batch_size=args.batch_size,
										shuffle=True
										)
	batch_valid_data, batch_val_label = mk_batch(valid_data,
										batch_size=args.batch_size,
										shuffle=True
										)
	statistical_result_name = 'result_record/theano_batch='+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(HIDDEN_LAYER)+'_'+str(HIDDEN_LAYER_DIM)+'_'+str(LEARNING_RATE)+'_4.6.csv'
	print "saving result to:", statistical_result_name
	print "data & label shape:", batch_training_data.shape, batch_train_label.shape
	
	print "training started..."
	start_time = timeit.default_timer()
	
	dnn = DNN(
		input_dim=input_dim,
		n_hid_layer=HIDDEN_LAYER,
		hid_layer_dim=HIDDEN_LAYER_DIM,
		output_dim=4,
		batch_size=args.batch_size,
	)
	
	csvfile = open(statistical_result_name, 'wb')
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerow(['epoch','acc_val','acc_tra'])
	n_batch = len(batch_training_data)
	for epoch in range(1, args.training_epochs+1):
		shuffleOrder = range(n_batch)
		random.shuffle(shuffleOrder)
		for j in shuffleOrder:
			cost = dnn.train(batch_training_data[j], batch_train_label[j])
			if(epoch == 1): dnn.first = False
			#print cost
			my_print(epoch, j+1, n_batch, cost)
		# validation
		acc_val = 0.0
		if(epoch%1 == 0):
			for j in range(len(batch_valid_data)):
				acc_val += (dnn.valid(batch_valid_data[j], batch_val_label[j])) \
							/len(valid_data)
		acc_tra = 0.0
		if(epoch%1 == 0):
			for j in range(len(batch_training_data)):
				acc_tra += (dnn.valid(batch_training_data[j], batch_train_label[j])) \
							/len(training_data)
		print ", acurracy of validation set:",acc_val,"  , acurracy of train set:",acc_tra
		csvwriter.writerow([epoch,acc_val,acc_tra])
		#if(epoch%20 == 0):
		#	print "==========================testing=========================="
		#	acc_test = 0.0
		#	for j in range(len(batch_test_data)):
		#		acc_test += (dnn.valid(batch_test_data[j], batch_test_label[j]) \
		#					/len(test_img))
		#	print("acurracy of testing set: %f" % acc_test)
		#	print "saving trained weights...\n"
		#	dnn.save_weights("trained_w_dnn/dnn_reduced_rbm_trained_epoch_"+str(epoch))
		"""early stoping"""
		#if(epoch%50 == 0):
		#	go_on = raw_input("Keep training...?(yes/no)")
		#	if(go_on == "no"):
		#		break

	print "training finished..."
	csvfile.close()
	end_time = timeit.default_timer()
	training_time = end_time - start_time
	print ("training took %f minutes" % (training_time/60.))


if __name__ == '__main__':
	train()


