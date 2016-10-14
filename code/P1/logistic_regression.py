import sys
from gradient_descent       import gradient_descent,  make_numeric_gradient_calculator, plot_gradient_descent
import matplotlib.pyplot    as plt
import numpy
import math
import sys
import pylab as pl
from sklearn                import linear_model
###Loss Functions####

def logistic_loss(x, y, w):
	w0 = w[0]
	weight = w[1:len(w)]
	cumulative_sum = 0

	for i in range(len(x)):
		
		inner_term = numpy.dot(-y[i], (numpy.dot(weight, x[i]) + w0))
		term = 1 + numpy.exp(inner_term)
		log_term = numpy.log(term)
		cumulative_sum += log_term

	return cumulative_sum[0]

def L2_regularization(w):
	weight = w[1: len(w)]
	cum_sum = 0
	for i in range(len(weight)):
		cum_sum += weight[i]**2

	return cum_sum**(0.5)

def L1_regularization(w):
	weight = w[1: len(w)]
	cum_sum = 0
	for i in range(len(weight)):
		cum_sum += math.fabs(weight[i])

	return cum_sum

def create_L2_logistic_objective(x, y, reg_parameter):
	def L2_logistic_objective(w):
		return logistic_loss(x, y, w) + numpy.dot(reg_parameter, L2_regularization(w)**2)

	return L2_logistic_objective

def create_L1_logistic_objective(x, y, reg_parameter):
	def L1_logistic_objective(w):
		return logistic_loss(x, y, w) + numpy.dot(reg_parameter, L1_regularization(w))

	return L1_logistic_objective

def create_Logistic_predictor(objective_f):
	def predictor(x):
		return objective_f.predict(x)
	return predictor

#### Actual Execution #####
if __name__ == '__main__':
	   # parameters
	name = '1'
	print '======Training======'
	# load data from csv files
	train = numpy.loadtxt('../data/data'+name+'_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	weight_vector_length = len(X[0])+1
	
	reg_parameter = 1



	initial_guess = numpy.array([100.0] * (weight_vector_length))

	# Parameters for logistic regression

	step_size = 0.05
	threshold = 0.001

	#### Sk Learn Logistic Regression #######

	L1_logistic_regressor = linear_model.LogisticRegression(penalty = 'l1', tol =0.001, C = 1)
	L2_logistic_regressor = linear_model.LogisticRegression(penalty = 'l2', tol =0.001, C = 1)

	L1_logistic_regressor.fit(X, Y)
	print "L1 weights", L1_logistic_regressor.coef_
	print "L1 error rate", L1_logistic_regressor.score(X, Y)

	L2_logistic_regressor.fit(X, Y)
	print "L2 weights", L2_logistic_regressor.coef_
	print "L2 error rate", L2_logistic_regressor.score(X, Y)

	# Carry out training.
	##### Our own gradient descent #####
	# objective_f = create_L2_logistic_objective(X, Y, reg_parameter)


	# gradient_f = make_numeric_gradient_calculator(objective_f, 0.00001)


	# previous_values = gradient_descent(objective_f, gradient_f, initial_guess, step_size, threshold)
	# min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
	# print "min_x: ", min_x, "  min_y",  min_y
	# print "number of steps: ", len(previous_values)
	

	# plot_gradient_descent(objective_f, previous_values)
