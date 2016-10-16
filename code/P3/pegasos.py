import sys
import matplotlib.pyplot    as plt
import numpy
import math
import sys
import pylab as pl
from plotBoundary import *
from sklearn                import linear_model

def run_pegasos(X, Y, reg_parameter, max_epochs):
	t = 0 
	weight = numpy.array([0.0] * len(X[0]))
	weights_matrix = numpy.matrix([weight] * (max_epochs+1))
	step_size = 0
	while (t < max_epochs):
		for i in range(len(X)):
			t += 1
			step_size = (1/ (t * reg_parameter))
			inner_product = numpy.dot(numpy.transpose(weights_matrix[t]), X[i])
			if numpy.dot(Y[i], inner_product) < 1:
				weights_matrix[t+1] = (1 - step_size*reg_parameter) * weights_matrix[t] + numpy.dot(numpy.dot(step_size, Y[i]), X[i])
			else:
				weights_matrix[t+1] = (1 - step_size*reg_parameter) * weights_matrix[t]


	return weights_matrix



