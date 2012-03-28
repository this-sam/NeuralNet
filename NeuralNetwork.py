#Neural Network by Sam Brown
global copy, math
import copy, math


class NeuralNetwork(object):
	def __init__(self):
		self.layers =  [[1,1,1,1,1,1,1],
							 [1,1,1,1,1,1],
							 [1,1,1,],
							 [1]]
	
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
		self.values = copy.deepcopy(self.layers) #l, i
		self.signals = copy.deepcopy(self.layers) #l, i
		
		print self.values
		print self.signals
		
		
		#Data-related definitions:
		self.M = lambda : len(self.data)
		self.m = lambda p : len(self.data[p])
		
		self.labels = [[1],[0],[1],[1]]
		self.data = [[1,1,1,1,1,1,1],
						[0,0,0,0,0,0,0],
						[1,1,1,1,1,1,1],
						[1,1,1,1,1,1,1]]

		self.u = .5
		self.sigma = lambda x : 1.0/(1+math.exp(-1+self.u))
		self.dSigma = lambda x : 2*x*(1-x)
		
	
	def backPropogation(self):
		error = 0
		Delta = []
		for i in range(self.L()):
			Delta.append([])
			for j in range(self.n(i)):
				Delta[i].append(0)
					
		#feed forward loop
		for p in range(0,self.M()):
			#load initial values
			for i in range(0, self.m(p)):
				self.values[0][i] = self.data[p][i]
				
			#push the signal through the network
			for l in range(1, self.L()):
				for i in range(1, self.n(l)):
					self.signals[l][i] = 0
					for j in range(0, self.n(l-1)):
						self.signals[l][i] += self.weights[l][i][j]*self.values[l-1][j]
					self.values[l][i] = self.sigma(self.signals[l][i])

			#compare labels to prediction
			for i in range(0, self.n(self.L()-1)):
				error += (self.labels[p][i])**2
				Delta[self.L()-1][i] = (self.labels[p][i] - self.values[self.L()-1][i]*\
											 self.dSigma(self.values[self.L()-1][i]))
				
 			#propogate error backward
			for l in range(self.L()-2, 0, -1):
				for j in range(0, self.n(self.L()-1)):
					Delta[l][j] = 0
					for i in range(self.n(self.L()-1)):
						pass
						
if __name__ == '__main__':
	n = NeuralNetwork()
	n.backPropogation()