import cPickle
import argparse
import numpy as np
import sys

"""my .py file"""
from parse import mk_batch, mk_all

"""Keras"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, activity_l2
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD,RMSprop
from keras.utils import generic_utils
from keras.preprocessing.sequence import pad_sequences

iii=1
jjj=0
while jjj<iii:
	parser = argparse.ArgumentParser()
	parser.add_argument("-n_hid_units", type=int, default=1024)
	parser.add_argument("-n_hid_layers", type=int, default=2)
	parser.add_argument("-dropout", type=float, default=0.2)
	parser.add_argument("-activation", type=str, default="relu")
	parser.add_argument("-epochs", type=int, default=200)
	parser.add_argument("-model_save_interval", type=int, default=10)
	parser.add_argument("-batch_size", type=int, default=1)
	parser.add_argument("-dataset", type=str, default="mod1")
	args = parser.parse_args()

	print "Loading MCTest training data...\n",
	# len(data) == total num of data
	# traing_data[i] == [input_vec, label]
	#input_dim=1800
	training_data = cPickle.load(open("Pickle/mc500.train.lstm.noStopWord.pickle"))
	#training1_data = cPickle.load(open("Pickle/mc500.train.mod1.pickle"))
	valid_data = cPickle.load(open("Pickle/mc500.dev.lstm.noStopWord.pickle"))
	if(args.dataset=="mod2"):
		input_dim=4500
		training_data = cPickle.load(open("Pickle/mc500.train.mod2.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))
	
	batch_training_data, batch_label = mk_all(training_data)
	batch_valid_data, batch_valid_label = mk_all(valid_data)
	maxlen =0 
	for ff in batch_training_data:
		if len(ff) > maxlen:
			maxlen = len(ff)
	for opp,gg in enumerate(batch_training_data):
		for i in range(maxlen-len(gg)):
			batch_training_data[opp].append(np.zeros(300))
	for opp,gg in enumerate(batch_valid_data):
		for i in range(maxlen-len(gg)):
			batch_valid_data[opp].append(np.zeros(300))

	#batch_valid_data, batch_valid_label = mk_batch(valid_data,
	#									batch_size=len(valid_data),
	#									shuffle=False
	#									)
#
#	for each in training_data:
#		fuck = each[0]
	#sys.exit(1)
	n_classes=4 # number of classes to be classified == 4 choices
	batch_training_data = np.array(batch_training_data)
	batch_valid_data = np.array(batch_valid_data)
	model = Sequential()
	model.add(LSTM(output_dim = 128, return_sequences = False, input_shape = (662,300), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid'))
	model.add(Dense(args.n_hid_units,
		input_dim=128,
		init="glorot_uniform",
		activation=args.activation,
		W_regularizer=l2(0.01))
	)

	if args.dropout > 0:
		model.add(Dropout(args.dropout))
	
	for i in range(args.n_hid_layers-1):
		model.add(Dense(args.n_hid_units,
			init="glorot_uniform",
			activation=args.activation,
			W_regularizer=l2(0.01))
		)
		if args.dropout > 0:
			model.add(Dropout(args.dropout))
	
	model.add(Dense(n_classes, init="glorot_uniform"))
	model.add(Activation("softmax"))

	json_string = model.to_json()
	model_file_name="models_dnn/"\
					+str(args.dataset)\
					+"_hid_neurons_"+str(args.n_hid_units)\
					+"_hid_layers_"+str(args.n_hid_layers)\
					+"_"+str(args.activation)\
					+"_bsize_"+str(args.batch_size)
	if(args.dropout > 0): model_file_name += ("_drop"+str(args.dropout))
	open(model_file_name  + ".json", "w").write(json_string)

	print "Compiling model..."
	model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
	print "Compilation done..."
	
	print "Training started..."
	model.fit(
		X=batch_training_data,
		y=batch_label,
		batch_size=args.batch_size,
		nb_epoch=args.epochs,
		#validation_split=0.2,
		validation_data=(batch_valid_data, batch_valid_label),
		shuffle=True,
		show_accuracy=True
	)
	#for epoch in range(args.epochs):
	#	progbar = generic_utils.Progbar(len(data))
	#	
	#	for training_data, label in zip(batch_training_data, batch_label):
	#
	#		loss_tr = model.train_on_batch(training_data, label)
	#		#add to progress bar
	#		progbar.add(args.batch_size, values=[("train loss", loss_tr[0])])
	#
	#	if (epoch+1)%args.model_save_interval == 0:
	#		model.save_weights(model_file_name + "_epoch_{:02d}.hdf5".format(epoch+1),\
	#						overwrite = True)
	#		print epoch+1, "epoch"
	jjj =jjj+1


