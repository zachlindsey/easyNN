'''

This is a module for me to learn about/test some basics
about neural nets.

Each of the classes defined in this document is a single
component of a neural net architecture. 

'''


import numpy

class maxAct(object):
	'''
	A RELU gate: computes x -> max(0,x) pointwise on input
	'''
	def __init__(self, dim):
		dX = numpy.zeros(dim)
		return None

	def forward(self, X):
		foo = numpy.maximum(X,0)
		self.deriv = []
		for x in foo:
			if x == 0:
				self.deriv.append(0)
			else:
				self.deriv.append(1)
		self.deriv = numpy.array(self.deriv)
		return numpy.maximum(X,0)

	def backprop(self, dY):
		return self.deriv*dY

	def update(self):
		pass

class MSEScorer(object):
	'''
	Calling forward gives the MSE between the two inputs
	'''
	def __init__(self):
		return None

	def forward(self, X, target):
		'''
		X, target - numpy 1d arrays of the same size
		'''
		self.deriv = 2*(X-target)/X.size
		return numpy.square(X-target).sum()/X.size

	def backprop(self, dscore):
		return dscore*self.deriv

	def update(self):
		pass



class SVMScorer(object):
	'''
	Foward takes a vector and an index of that vector
	which is the "correct" index and returns the sum
	of the hinge losses over other indices
	'''
	def __init__(self):
		return None

	def forward(self, X, i):
		'''
		i = index of correct class
		'''

		res = numpy.maximum(X - X[i] + 1,0)
		self.deriv = []
		Xicount = 1
		# why isn't this easier
		for x in res:
			if x > 0:
				self.deriv.append(1)
				Xicount -= 1
			else:
				self.deriv.append(0)
		self.deriv[i] = Xicount
		self.deriv = numpy.array(self.deriv)
		return res.sum()-1

	def backprop(self, dscore):
		return dscore*self.deriv

	def update(self):
		pass


class linUnit(object):
	'''
	simply does an affine transformation:
	x -> Wx + b

	W, b are internal

	'''
	def __init__(self, learnrate, n, m):
		'''
		learnrate - step size when doing grad desc
		n - input size
		m - output size
		'''
		self.learnrate = learnrate
		self.W = numpy.random.rand(m,n)
		# this extra factor at the end is probably not good
		self.b = numpy.zeros(m)
		self.dW = numpy.zeros((m,n))
		self.db = numpy.zeros(m)
		self.batchsize = 0

		return None

	def forward(self, X):
		'''
		X = numpy array of inputs

		note that this is slow b/c the activation isnt vectorized
		'''
		self.X = X
		self.out = numpy.dot(self.W, X)+self.b
		return self.out

	def backprop(self, dY):
		'''
		dY = incoming gradient
		'''
		#print('size of X:', self.X.shape)
		#print('size of W:', self.W.shape)
		#print('size of Y:', dY.shape)
		self.dW += numpy.dot(dY[:,None], (self.X)[:,None].T)
		self.db += dY
		self.batchsize += 1
		dX = numpy.dot(self.W.T, dY)
		return dX

	def update(self):
		self.W += -self.dW*self.learnrate/self.batchsize
		self.b += -self.db*self.learnrate/self.batchsize
		self.dW = numpy.zeros(self.dW.shape)
		self.db = numpy.zeros(self.db.shape)
		self.batchsize = 0
		return None

class convUnit(object):
	def __init__(self, *params):
		return None

class maxPoolUnit(object):
	def __init__(self, *params):
		return None

