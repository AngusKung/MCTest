import cPickle
import argparse
import numpy as np
import pdb
import sys

"""my .py file"""
from parse import *

"""Keras"""
from keras.models import Sequential, Graph
from keras.layers import *
from keras.regularizers import l2, activity_l2
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
	training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.noStopWord.pickle"))
	valid_data = cPickle.load(open("Pickle/mc160.dev.lstm.noStopWord.pickle"))
	if(args.dataset=="dev:mc500"):
		training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.noStopWord.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.lstm.noStopWord.pickle"))
	else:
		print "Using default dataset: mc500+160.train.lstm.noStopWord.pickle"
		print "with dev dataset: mc160.dev.lstm.noStopWord.pickle"
	passages, questions, A1, A2, A3, A4, true_ans = mk_gru300(training_data)
	passages_val, questions_val, A1_val, A2_val, A3_val, A4_val, true_ans_val = mk_gru300(valid_data)
	#sys.exit(1)

	#padding zeros!!!!
	maxlen = 0
	# 3 stories more than 400 words in train(4xx,5xx,6xx), 2 stories more than 400 words in test 
	for i,ff in enumerate(questions):
		if len(ff) > maxlen:
			#print "train Line "+str(i)+" :"+str(len(ff))
			maxlen = len(ff)
	for opp,gg in enumerate(questions):
		for i in range(maxlen-len(gg)):
			questions[opp].insert(0,np.zeros(300))
	for opp,gg in enumerate(questions_val):
		for i in range(maxlen-len(gg)):
			questions_val[opp].insert(0,np.zeros(300))
	questions = np.array(questions, dtype='float32')
	questions_val = np.array(questions_val, dtype='float32')
	
	maxlen_pass = 0
	for i,ff in enumerate(passages):
		if len(ff) > maxlen_pass:
			#print "train Line "+str(i)+" :"+str(len(ff))
			maxlen_pass = len(ff)
	for opp,gg in enumerate(passages):
		for i in range(maxlen_pass-len(gg)):
			passages[opp].insert(0,np.zeros(300))
	for opp,gg in enumerate(passages_val):
		for i in range(maxlen_pass-len(gg)):
			passages_val[opp].insert(0,np.zeros(300))
	passages = np.array(passages, dtype='float32')
	passages_val = np.array(passages_val, dtype='float32')
	#end of padding zeros!!!
	
	n_classes= 4 # number of classes to be classified == 4 choices
	dim_glove = 300
	dim_gru = 128
	
	theGRU = GRU(output_dim = dim_gru, return_sequences = False, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'sigmoid')
	gru_to_embed = Dense(dim_gru,init='glorot_uniform',activation='relu')
	#the_backGRU = GRU(output_dim = dim_gru, go_backwards=True, return_sequences = True, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'sigmoid')
	embd = Embedding(input_dim = dim_glove,output_dim = dim_glove,input_length = maxlen, init='glorot_uniform')
	#def return_last_time(x):	
	#def return_last_time_output_shape(input_shape):

	model = Graph()
	model.add_input(name='passage', input_shape=(maxlen,),dtype='float32')
	model.add_node(embd, name='embd',input='passage')
	model.add_node(theGRU ,name='gru', input='embd')
	model.add_node(Dropout(0.2),name='drop1',input='gru')
	#model.add_node(the_backGRU ,name='backgru', input='input')
	model.add_node(gru_to_embed, name='gru_to_embed', input='drop1') # shape (b_s, dim_latent)
	model.add_node(RepeatVector(maxlen),name='repeat1',input='gru_to_embed')# shape (batch_size, maxlen, dim_latent)

	#model_pass = Graph()
	#model_pass.add_input(name='pass_input', input_shape=(maxlen,), dtype='float32')

	model.add_node(Dense(300, init="glorot_uniform",activation='relu'),name='last_hidden',input='gru_to_embed')

	model.add_node(Dense(n_classes, init="glorot_uniform",activation='softmax'),name='final',input='last_hidden')
	model.add_output(name = 'output', input='final')

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
	model.fit(
		{'passage':passages,'':}
		batch_size=args.batch_size,
		nb_epoch=args.epochs,
		#validation_split=0.2,
		validation_data=(batch_valid_data, batch_valid_label),
		#shuffle=True,
		show_accuracy=True
	)
