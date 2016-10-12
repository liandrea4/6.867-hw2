import sys
from gradient_descent       import gradient_descent,  make_numeric_gradient_calculator, plot_gradient_descent
import matplotlib.pyplot    as plt
import numpy
import math
import sys
import pylab as pl
###Loss Functions####

def logistic_loss(x, y, w, w0):
	cumulative_sum = 0
	for i in range(len(x)):
		
		inner_term = -y[i] * (numpy.dot(w, x[i]) + w0)
		term = 1 + numpy.exp(inner_term)
		log_term = numpy.log(term)
		cumulative_sum += log_term

	print "sum", cumulative_sum[0]
	return cumulative_sum[0]

def L2_regularization(w):
	cum_sum = 0
	for i in range(len(w)):
		cum_sum += w[i]**2

	print "cumsum", cum_sum
	return cum_sum**(0.5)

def create_L2_logistic_objective(x, y, w0, reg_parameter):
	def L2_logistic_objective(w):
		return logistic_loss(x, y, w, w0) + numpy.dot(reg_parameter, L2_regularization(w)**2)

	return L2_logistic_objective

#### Actual Execution #####
if __name__ == '__main__':
	   # parameters
	name = '1'
	print '======Training======'
	# load data from csv files
	train = numpy.loadtxt('../data/data'+name+'_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	weight_vector_length = len(X[0])
	w0 = 0.0
	reg_parameter = 0



	initial_guess = numpy.array([100.0] * (weight_vector_length))

	# Parameters for logistic regression

	step_size = 0.1
	threshold = 0.01

	# Carry out training.

	objective_f = create_L2_logistic_objective(X, Y, w0, reg_parameter)


	gradient_f = make_numeric_gradient_calculator(objective_f, 0.001)


	previous_values = gradient_descent(objective_f, gradient_f, initial_guess, step_size, threshold)
	min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
	print "min_x: ", min_x, "  min_y",  min_y
	print "number of steps: ", len(previous_values)
	

	plot_gradient_descent(objective_f, previous_values)
