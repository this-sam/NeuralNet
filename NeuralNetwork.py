#Neural Network by Sam Brown
global copy, math, re, pprint, random
import copy, math, re, pprint, random


class NeuralNetwork(object):
	def __init__(self):
		self.depth = 3
		self.layers =  []
	
		#Weight-related definitions:
		self.L = lambda : len(self.layers)
		#self.N = lambda : len(self.layers[-1])
		self.n = lambda l: len(self.layers[l])
		#self.l() = range(1, layers)
		#i = range(1, len(layers[n]))
		#j = range(0, len(layers[n-1]))
		weights = [] #l, i, j
		
		for l in range(0, self.L()):
			weights.append([])
			for i in range(0, self.n(l)):
				weights[l].append([])
				for j in range(0, self.n(l-1)):
					weights[l][i].append(1)
		self.weights = weights
		
		#Node-related definitions:
		#l = range(0, self.L())
		#i = range(0, self.n(l))
		self.values = []#copy.deepcopy(self.layers) #l, i
		self.signals = []#copy.deepcopy(self.layers) #l, i		
		
		#Data-related definitions:
		self.M = lambda : len(self.data)
		self.m = lambda p : len(self.data[p])
		
		self.labels = []
		self.data = []

		self.u = .5
		self.sigma = lambda x : 1.0/(1+math.exp(-1+self.u))
		self.dSigma = lambda x : 2*x*(1-x)
		
		self.eta = .4
		
		self.Delta = []
		self.resetDelta()
		
		self.errorRecord = []
	
	
	def backPropogation(self):
		#feed forward loop
		for p in range(0,self.M()):
			self.resetDelta()
			
			self.loadInitialValues(p)
			self.pushSignalForward(p)	
			self.compareLabels(p)
			
 			#propogate error backward
			for l in range(self.L()-2, 0, -1):
				for j in range(0, self.n(l)):
					self.Delta[l][j] = 0
					for i in range(self.n(l+1)):
						self.Delta[l][j] += self.Delta[l+1][i]*self.weights[l+1][i][j]
					self.Delta[l][j] *= 2*self.values[l][j]*(1-self.values[l][j])
					
					#adjust the weights
					for k in range(self.n(l-1)):
						self.weights[l][j][k] += self.eta*self.Delta[l][j]*self.values[l-1][k]
	
	
	def predictData(self):
		for p in range(0,self.M()):
			self.loadInitialValues(p)
			self.pushSignalForward(p)	
			self.compareLabels(p)
	
	
	def resetDelta(self):
		self.Delta = []
		for i in range(self.L()):
			self.Delta.append([])
			for j in range(self.n(i)):
				self.Delta[i].append(0)					
	
	
	def predictLabel(self, p):
		self.loadInitialValues(p)
		self.pushSignalForward(p)
		self.compareLabels(p)
		
	
	#load initial values
	def loadInitialValues(self, p):
		for i in range(0, self.m(p)):
			self.values[0][i] = self.data[p][i]
		print "Values:",self.values[0]

	#push forward through the network
	def pushSignalForward(self, p):
		for l in range(1, self.L()):
			for i in range(1, self.n(l)):
				self.signals[l][i] = 0
				for j in range(0, self.n(l-1)):
					self.signals[l][i] += self.weights[l][i][j]*self.values[l-1][j]
				self.values[l][i] = self.sigma(self.signals[l][i])
	
	
	#compare labels to prediction
	def compareLabels(self, p):
		error = 0
		print "Labels:", self.labels[p]
		print "Values:",self.values[self.L()-1]

		for i in range(0, self.n(self.L()-1)):
			error += (self.labels[p][i]-self.values[self.L()-1][i])**2
			self.Delta[self.L()-1][i] = (self.labels[p][i] - self.values[self.L()-1][i]*\
										 self.dSigma(self.values[self.L()-1][i]))
		
		print "Error:", error
		self.errorRecord.append(error)
		print
		
		
	def loadData(self, fname):
		data = []
		labels = []
		description = []
		try:
			f = open(fname, 'r')
		except IOError:
			print "IOError on file:", fname
		else:
			lines = f.readlines()
			description = re.split("[,][, ]*",lines[0].strip())
			for i in range(len(description)):
				description[i] = int(description[i])
			for i in range(1, len(lines)-1, 2):
				inputs = re.split("[,][, ]*",lines[i].strip())
				for j in range(len(inputs)):
					inputs[j] = float(inputs[j])
				data.append(inputs)
				
				label = int(lines[i+1].strip())
				if label == 0:
					labels.append([1,0,0])
				elif label == 1:
					labels.append([0,1,0])
				elif label == 2:
					labels.append([0,0,1])
				
		self.data = data
		self.labels = labels
		self.description = description
		
		self.prepareNetwork()
		
	def prepareNetwork(self):
		for i in range(self.depth):
			self.layers.append([])
			for j in range(self.description[1]):
				self.layers[i].append(1)
	
		weights = [] #l, i, j
		
		for l in range(0, self.L()):
			weights.append([])
			for i in range(0, self.n(l)):
				weights[l].append([])
				for j in range(0, self.n(l-1)):
					weights[l][i].append(1)
		
		self.weights = weights

		self.values = copy.deepcopy(self.layers) #l, i
		self.signals = copy.deepcopy(self.layers) #l, i
		
		self.randomizeArray(self.values)
		self.randomizeArray(self.signals)
		self.randomizeArray(self.weights)
		
		self.resetDelta()
		
		
	def randomizeArray(self, array):
		for i in range(len(array)):
			if type(array[i]) == type([]):
				self.randomizeArray(array[i])
			else:
				array[i] *= random.random()
		return array
	

if __name__ == '__main__':
	n = NeuralNetwork()
	n.loadData('hw3_training.csv')
	n.backPropogation()
	pprint.pprint(n.weights)
	#pprint.pprint(n.errorRecord)