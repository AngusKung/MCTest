import cPickle
import argparse
import numpy as np

"""my .py file"""
from parse import mk_batch

"""Keras"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import generic_utils

def train():

	parser = argparse.ArgumentParser()
	parser.add_argument("-n_hid_units", type=int, default=300)
	parser.add_argument("-n_hid_layers", type=int, default=3)
	parser.add_argument("-dropout", type=float, default=0.2)
	parser.add_argument("-activation", type=str, default="relu")
	parser.add_argument("-epochs", type=int, default=200)
	parser.add_argument("-model_save_interval", type=int, default=10)
	parser.add_argument("-batch_size", type=int, default=5)
	parser.add_argument("-dataset", type=str, default="mod2")
	args = parser.parse_args()

	print "Loading MCTest training data...\n",
	# len(data) == total num of data
	# traing_data[i] == [input_vec, label]
	#input_dim=1800
	#data = cPickle.load(open("Pickle/mc500.train.mod1.pickle"))
	#valid_data = cPickle.load(open("Pickle/mc500.dev.mod1.pickle"))
	#if(args.dataset=="mod2"):
	input_dim=4500
	data = cPickle.load(open("Pickle/mc500.train.mod2.pickle"))
	valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))

	batch_training_data, batch_train_label = mk_batch(data,
										batch_size=len(data),
										shuffle=True
										)
	batch_valid_data, batch_val_label = mk_batch(valid_data,
										batch_size=len(valid_data),
										shuffle=True
										)
	print "data & label shape:", batch_training_data[0].shape, batch_train_label[0].shape
	n_classes=4 # number of classes to be classified == 4 choices
	model = Sequential()
	model.add(Dense(args.n_hid_units,
		input_dim=input_dim,
		init="glorot_uniform",
		activation=args.activation)
	)

	if args.dropout > 0:
		model.add(Dropout(args.dropout))
	
	for i in range(args.n_hid_layers-1):
		model.add(Dense(args.n_hid_units/(2**i),
			init="glorot_uniform",
			activation=args.activation)
		)
		if args.dropout > 0:
			model.add(Dropout(args.dropout))
	
	model.add(Dense(n_classes, init="glorot_uniform"))
	model.add(Activation("softmax"))

#	json_string = model.to_json()
#	model_file_name="../models_dnn/"\
#					+str(args.dataset)\
#					+"_hid_neurons_"+str(args.n_hid_units)\
#					+"_hid_layers_"+str(args.n_hid_layers)\
#					+"_"+str(args.activation)\
#					+"_bsize_"+str(args.batch_size)
#	if(args.dropout > 0): model_file_name += ("_drop"+str(args.dropout))
#	open(model_file_name  + ".json", "w").write(json_string)

	print "Compiling model..."
        #rmsprop = RMSprop()
	model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
	print "Compilation done..."
	
	print "Training started..."
	model.fit(batch_training_data[0], batch_train_label[0], nb_epoch = 200, validation_data=(batch_valid_data[0],batch_val_label[0]), show_accuracy = True)
	#model.fit(batch_valid_data[0], batch_val_label[0], nb_epoch = 200, validation_data=(batch_training_data[0],batch_train_label[0]), show_accuracy = True)
        #for epoch in range(args.epochs):
	#	progbar = generic_utils.Progbar(len(data))
		
	#	for training_data, label in zip(batch_training_data, batch_label):

#			loss_tr = model.train_on_batch(training_data, label)
			#add to progress bar
#			progbar.add(args.batch_size, values=[("train loss", loss_tr[0])])

#		if (epoch+1)%args.model_save_interval == 0:
#			model.save_weights(model_file_name + "_epoch_{:02d}.hdf5".format(epoch+1),\
#							overwrite = True)
#			print epoch+1, "epoch"

if __name__ == "__main__":
	train()


