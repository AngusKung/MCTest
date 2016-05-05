import cPickle
import argparse
import numpy as np
import pdb
import sys

"""my .py file"""
from parse import *

"""Keras"""
from keras.models import Sequential, Graph
from keras.layers.core import *
from keras.regularizers import l2, activity_l2
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD,RMSprop
from keras.utils import generic_utils
from keras import backend as K

if __name__ == '__main__' :	
	parser = argparse.ArgumentParser()
	parser.add_argument("-n_hid_units", type=int, default=64)
	parser.add_argument("-n_hid_layers", type=int, default=2)
	parser.add_argument("-dropout", type=float, default=0.2)
	parser.add_argument("-activation", type=str, default="relu")
	parser.add_argument("-epochs", type=int, default=1000)
	parser.add_argument("-model_save_interval", type=int, default=10)
	parser.add_argument("-batch_size", type=int, default=20)
	parser.add_argument("-dataset", type=str, default="mod2.x24.merge")
	args = parser.parse_args()

	print "Running with args:"
	print args

	print "Loading MCTest training data...\n",
	training_data = cPickle.load(open("Pickle/mc500.train.mod2.x24.merge.pickle"))
	valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))
	if(args.dataset=="mod2"):
		input_dim=4500
		training_data = cPickle.load(open("Pickle/mc500.train.mod2.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))
	elif(args.dataset=="mod1"):
		input_dim=1800
		training_data = cPickle.load(open("Pickle/mc500.train.mod1.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod1.pickle"))
	elif(args.dataset=="mod1.x24"):
		input_dim=1800
		training_data = cPickle.load(open("Pickle/mc500.train.mod1.x24.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod1.pickle"))
	elif(args.dataset=="mod1.x24.noStopWord"):
		input_dim=1800
		training_data = cPickle.load(open("Pickle/mc500.train.mod1.x24.noStopWord.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod1.pickle"))
	elif(args.dataset=="mod2.x24"):
		input_dim=4500
		training_data = cPickle.load(open("Pickle/mc500.train.mod2.x24.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.mod2.pickle"))
	elif(args.dataset=="auto"):
		input_dim = 4000
		training_data = cPickle.load(open("Pickle/mc500.train.auto.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.auto.pickle"))
	elif(args.dataset=="auto.txt"):
		input_dim = 5500
		training_data = cPickle.load(open("Pickle/mc500.train.auto.txt.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.auto.txt.pickle"))
	else:
		print "Using default dataset: mod2.x24.merge"
	
	# =========== No need when using mod2.x24.merge =============
	#batch_training_data, batch_label = mk_merge_batch(training_data,args.batch_size)
	#batch_valid_data, batch_valid_label = mk_merge_batch(valid_data,args.batch_size)
	#sys.exit(1)
	#batch_training_data = np.array(batch_training_data,dtype='float32')
	#batch_valid_data = np.array(batch_valid_data,dtype='float32')
	#sys.exit(1)
	
	
	n_classes= 4 # number of classes to be classified == 4 choices
	dim_glove = 300
	
	model = Graph()
	model.add_input(name='input_que', input_shape=(dim_glove,),dtype='float32')
	for i in range(10):
		model.add_input(name='input_para_'+str(i+1), input_shape=(dim_glove,),dtype='float32')
		model.add_node(Activation('linear'), name='para_'+str(i+1)+'_act', inputs=['input_que','input_para_'+str(i+1)], merge_mode='sum')
		model.add_node(Dense(256,input_dim=600,activation='relu',init='glorot_uniform'), name='para_'+str(i+1), input='para_'+str(i+1)+'_act')
	#	if args.dropout > 0:
	#		model.add_node(Dropout(args.dropout), name='para_drop_'+str(i+1),input='para_'+str(i+1))
	# ----------- para ready -----------
	model.add_node( Activation('linear'),name='para_seen_act', inputs=['para_1','para_2','para_3','para_4','para_5','para_6','para_7','para_8','para_9','para_10'], merge_mode='concat' )
	model.add_node( Dense(512,input_dim=2560,activation='relu',init='glorot_uniform'),name='para_seen', input='para_seen_act')
	#if args.dropout > 0:
	#	model.add_node(Dropout(args.dropout),name='para_seen_drop',input='para_seen')
	# ----------- para+que ready ------------
	for i in range(4):	
		model.add_input(name='input_opt_'+str(i+1), input_shape=(dim_glove,),dtype='float32')
	model.add_node(Activation('linear'), name='opt_seen_act', inputs=['para_seen','input_opt_1','input_opt_2','input_opt_3','input_opt_4'],merge_mode='concat')
	model.add_node(Dense(512,input_dim=512+300*4,activation='relu',init='glorot_uniform'), name='opt_seen', input='opt_seen_act')
	# ----------- option seen and merge with para+que ------------
	#if args.dropout > 0:
	#	model.add_node(Dropout(args.dropout),name='opt_seen_drop',input='opt_seen')
	model.add_node(Dense(64,input_dim=512,activation='relu',init='glorot_uniform'),name = 'all_seen',input='opt_seen')
	#if args.dropout > 0:
	#	model.add_node(Dropout(args.dropout),name='all_seen_drop',input='all_seen')
	model.add_node(Dense(4,input_dim=64,activation='softmax',init='glorot_uniform'),name='final',input='all_seen')
	model.add_output(name='output',input='final')

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
	model.compile(loss={'output':'categorical_crossentropy'}, optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6))
	print "Compilation done..."
	
	print "Training started..."
	for k in range(args.epochs):
		print "Epoch",k+1,"..."
		train_loss = 0
		for i in range(len(training_data)):
			train_loss += model.train_on_batch(
			       {'input_para_1':training_data[i][0][0].reshape(1,300),
				'input_para_2':training_data[i][0][1].reshape(1,300),
				'input_para_3':training_data[i][0][2].reshape(1,300),
				'input_para_4':training_data[i][0][3].reshape(1,300),
				'input_para_5':training_data[i][0][4].reshape(1,300),
				'input_para_6':training_data[i][0][5].reshape(1,300),
				'input_para_7':training_data[i][0][6].reshape(1,300),
				'input_para_8':training_data[i][0][7].reshape(1,300),
				'input_para_9':training_data[i][0][8].reshape(1,300),
				'input_para_10':training_data[i][0][9].reshape(1,300),
				'input_que':training_data[i][0][10].reshape(1,300),
				'input_opt_1':training_data[i][0][11].reshape(1,300),
				'input_opt_2':training_data[i][0][12].reshape(1,300),
				'input_opt_3':training_data[i][0][13].reshape(1,300),
				'input_opt_4':training_data[i][0][14].reshape(1,300),
				'output':np.asarray(training_data[i][1],dtype='float32').reshape(1,4)
				})[0]
			print "Now on",i+1,"   loss = ",train_loss/(i+1)
			sys.stdout.write("\033[F")
			'''for j in range(9):
				model.nodes['para_'+str(j)].set_weights(model.nodes['para_10'].get_weights())'''
		print "training loss = ",train_loss/len(training_data)
		train_acc = 0
		for i in range(len(training_data)):
			ans = model.predict_on_batch(
			       {'input_para_1':training_data[i][0][0].reshape(1,300),
				'input_para_2':training_data[i][0][1].reshape(1,300),
				'input_para_3':training_data[i][0][2].reshape(1,300),
				'input_para_4':training_data[i][0][3].reshape(1,300),
				'input_para_5':training_data[i][0][4].reshape(1,300),
				'input_para_6':training_data[i][0][5].reshape(1,300),
				'input_para_7':training_data[i][0][6].reshape(1,300),
				'input_para_8':training_data[i][0][7].reshape(1,300),
				'input_para_9':training_data[i][0][8].reshape(1,300),
				'input_para_10':training_data[i][0][9].reshape(1,300),
				'input_que':training_data[i][0][10].reshape(1,300),
				'input_opt_1':training_data[i][0][11].reshape(1,300),
				'input_opt_2':training_data[i][0][12].reshape(1,300),
				'input_opt_3':training_data[i][0][13].reshape(1,300),
				'input_opt_4':training_data[i][0][14].reshape(1,300)
				})
			pdb.set_trace()
			if np.argmax(ans) == np.argmax(training_data[i][0]):
				train_acc += 1
		print "train_acc = ",train_acc/len(training_data)*100,"%"
		val_acc = 0
		for i in range(len(validation_data)):
			ans = model.predict_on_batch(
			       {'input_para_1':validation_data[i][0][0].reshape(1,300),
				'input_para_2':validation_data[i][0][1].reshape(1,300),
				'input_para_3':validation_data[i][0][2].reshape(1,300),
				'input_para_4':validation_data[i][0][3].reshape(1,300),
				'input_para_5':validation_data[i][0][4].reshape(1,300),
				'input_para_6':validation_data[i][0][5].reshape(1,300),
				'input_para_7':validation_data[i][0][6].reshape(1,300),
				'input_para_8':validation_data[i][0][7].reshape(1,300),
				'input_para_9':validation_data[i][0][8].reshape(1,300),
				'input_para_10':validation_data[i][0][9].reshape(1,300),
				'input_que':validation_data[i][0][10].reshape(1,300),
				'input_opt_1':validation_data[i][0][11].reshape(1,300),
				'input_opt_2':validation_data[i][0][12].reshape(1,300),
				'input_opt_3':validation_data[i][0][13].reshape(1,300),
				'input_opt_4':validation_data[i][0][14].reshape(1,300)
				})
			pdb.set_trace()
			if np.argmax(ans) == np.argmax(validation_data[i][0]):
				val_acc += 1
		print "val_acc = ",val_acc/len(validation_data)*100,"%"
	'''model.fit(
		X=batch_training_data,
		y=batch_label,
		batch_size=args.batch_size,
		nb_epoch=args.epochs,
		#validation_split=0.2,
		validation_data=(batch_valid_data, batch_valid_label),
		#shuffle=True,
		show_accuracy=True
	)'''


