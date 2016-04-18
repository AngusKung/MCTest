import random
import cPickle
import theano
import theano.tensor as T
import numpy as np
from itertools import izip


class DNN:

	def __init__(
		self,
		input_dim=0,
		hid_layer=[128,128,128],
		output_dim=0,
		batch_size=0,
		lr=0.001,
		momentum=0.9,
		pre_train=None
	):
		self.input_dim = input_dim
		self.hid_layer = hid_layer
		self.output_dim = output_dim
		self.batch_size = np.cast["float32"](batch_size)
		self.lr = np.cast["float32"](lr)
		self.momentum = np.cast["float32"](momentum)

		#Function model
		if pre_train is not None:
			self.w_rbm = cPickle.load(open(pre_train))
			print "using RBM as input feature..."
		else:
			self.w_rbm = np.eye(self.input_dim, dtype="float32")
		self.init_model(pre_train)

	def init_model(self, pre_train): # default 1 hidden layer

		x = T.matrix(dtype="float32")
		x_pre = x
		if(pre_train is not None):
			x_pre = T.dot(x, self.w_rbm)

		self.parameters = []
		for i in range(0,len(self.hid_layer)+1):
			if(i == 0):
				w = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=
							(self.input_dim, self.hid_layer[i])).astype(dtype="float32"))
				b = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=
							(self.hid_layer[i])).astype(dtype="float32"))
			elif(i == len(self.hid_layer)):
				w = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=
							(self.hid_layer[i-1], self.output_dim)).astype(dtype="float32"))
				b = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=
							(self.output_dim)).astype(dtype="float32"))
			else:	
				w = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=
							(self.hid_layer[i-1], self.hid_layer[i])).astype(dtype="float32"))
				b = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=
							(self.hid_layer[i])).astype(dtype="float32"))
			self.parameters.append(w)
			self.parameters.append(b)

		for i in range(len(self.hid_layer)+1):
			if(i == 0):
				self.z = T.dot(x_pre, self.parameters[2*i]) + self.parameters[2*i+1].dimshuffle('x', 0)
				self.a = T.log(1+T.exp(self.z))
			else:
				self.z = T.dot(self.a, self.parameters[2*i]) + self.parameters[2*i+1].dimshuffle('x', 0)
				self.a = T.log(1+T.exp(self.z))
		#a = 1/(1 + T.exp(-z1))
		#a = T.switch(z1 < 0, 0, z1)

		#y = 1/(1 + T.exp(self.z)) 
		#y = T.switch(z < 0, 0, z)
		#y = T.log(1+T.exp(z))

		y_hat = T.matrix(dtype="float32")

		y_softmax = T.exp(self.z) / T.sum(T.exp(self.z), axis=1).dimshuffle(0, 'x')
		cost = T.sum(y_hat * -T.log(y_softmax)) / self.batch_size
	
		#cost = (T.sum((y_T - y_hat)**2)) / self.batch_size

		gradients = T.grad(cost, self.parameters)
		#for denugging...
		debug = -T.log(y_softmax)

		self.sqr_grads = []
		self.first = True
		self.movements = []
		for p in self.parameters:
			self.sqr_grads.append(theano.shared(np.array(np.zeros
							(p.get_value().shape)).astype(dtype="float32")))
			self.movements.append(theano.shared(np.array(np.zeros
							(p.get_value().shape)).astype(dtype="float32")))

		self.debug_func = theano.function(
			inputs=[x, y_hat],
			#on_unused_inputs="ignore"
			allow_input_downcast=True,
			updates=self.myUpdate_ada(self.parameters, gradients, self.sqr_grads),
			outputs=debug
		)
		
		self.train_func = theano.function(
			inputs=[x, y_hat],
			allow_input_downcast=True,
			updates=self.myUpdate_ada(self.parameters, gradients, self.sqr_grads),
			outputs=cost)

		self.test_func = theano.function(
			inputs=[x],
			allow_input_downcast=True,
			outputs=y_softmax)
	
	def train(self, x_batch, y_batch):
		return self.train_func(x_batch, y_batch)

	def test(self, x_batch):
		return self.test_func(x_batch)

	def debug(self, x_batch, y_batch):
		return self.debug_func(x_batch, y_batch)

	def valid(self, x_batch, y_batch):
		y_predict = self.test_func(x_batch)

		y_predict_indices = np.argmax(y_predict, axis=1)
		y_answers_indices = np.argmax(y_batch, axis=1)
		diff = np.subtract(y_predict_indices, y_answers_indices)
		corrects = self.batch_size - np.count_nonzero(diff)
		return float(corrects)

	def myUpdate_gd(self, parameters, gradients):
		parameters_updates = [(p, p - (self.lr * g)) \
							for p, g in izip(parameters, gradients)]
		return parameters_updates

	def myUpdate_m(self, parameters, gradients, movements):
		parameters_updates = [(p, p + self.momentum * v - self.lr * g)
							for p, v, g in izip(parameters, movements, gradients)]

		parameters_updates += [(v, self.momentum * v - self.lr * g)
							for v, g in izip(movements, gradients)]
		return parameters_updates

	def myUpdate_ada(self, parameters, gradients, sqr_grads):
		parameters_updates = [(sqr_g, sqr_g + g**2)
							for sqr_g, g in izip(sqr_grads, gradients)]
		if(self.first == True):
			parameters_updates += [(p, p - (self.lr * g))
							for p, g in izip(parameters, gradients)]
		else:
			parameters_updates += [(p, p - (self.lr * g)/(T.sqrt(sqr_g)))
							for p, g, sqr_g in izip(parameters, gradients, sqr_grads)]
		return parameters_updates

	def save_weights(self, file_name):
		cPickle.dump(self.parameters, open(file_name+".pkl", "wb"))






