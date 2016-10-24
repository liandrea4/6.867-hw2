import sys
import matplotlib.pyplot    as plt
import numpy
import math
import sys
import pylab as pl
from plotBoundary import *
from sklearn                import linear_model

def find_L2_margin(weight):
	cum_sum = 0
	for i in range(len(weight)):
		cum_sum += weight[i]**2
	return cum_sum**(0.5)

def run_pegasos(X, Y, reg_parameter, max_epochs):
	t = 0
	epoch = 0
	weights = numpy.array([0.0] * len(X[0]))
	weights.reshape(2,1)
	weights_matrix = [weights] * (max_epochs * len(X)+2)

	print len(X)
	
	
	step_size = 0
	while (epoch < max_epochs):
		epoch +=1
		print"pre for loop t", t
		for i in range(len(X)):
			t += 1
			step_size = (1/ (t * reg_parameter))


			inner_product = numpy.dot(weights_matrix[t].reshape(1,2), X[i].reshape(2,1))

			if numpy.dot(Y[i], inner_product) < 1:
	
				a = numpy.dot((1 - step_size*reg_parameter), weights_matrix[t].reshape(2,1))
				constant = step_size * Y[i]
				b = numpy.dot(constant[0], X[i].reshape(2,1))
		
				print "in for loop t", t
				weights_matrix[t+1] = a+b
			else:
				print "in for loop t", t

				weights_matrix[t+1] = numpy.dot((1 - step_size*reg_parameter) , weights_matrix[t])


	print "margin: ", 1.0/(find_L2_margin(weights_matrix[-1]))			
	return weights_matrix[-1]


if __name__ == '__main__':

	train = loadtxt('../data/data3_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	epochs = 100;
	lmbda = 2**(-10);


	print run_pegasos(X, Y, lmbda, epochs)
