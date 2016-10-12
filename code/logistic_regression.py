import sys
from gradient_descent       import gradient_descent,  make_numeric_gradient_calculator, plot_gradient_descent
import matplotlib.pyplot    as plt
import numpy
import math
import sys

###Loss Functions####

def logistic_loss(x, y, w, w0):
	cumulative_sum = 0
	for i in range(len(x)):
		inner_term = numpy.dot(-y[i],(numpy.dot(w, x[i]) + w0)) 
		term = 1 + numpy.exp(inner_term)
		log_term = numpy.log(term)
		cumulative_sum += log_term

	return cumulative_sum

def L2_regularization(w):
	cum_sum = 0
	for i in range(w):
		cum_sum += w[i]**2

	return cum_sum**(0.5)

def create_L2_logistic_objective(x, y, w0, reg_parameter):
	def L2_logistic_objective(w):
		logistic_loss(x, y, w, w0) + numpy.dot(reg_parameter, L2_regularization(w)**2)


#### Actual Execution #####
if __name__ == '__main__':
    parameters = getData()
    initial_guess = numpy.array([0, 0])

    # Parameters for logistic regression

    step_size = 100000000
    threshold = 0.00001


    #Setup for logistic regression
    gaussian_mean = parameters[0]
    gaussian_cov = parameters[1]
    objective_f = make_negative_gaussian(gaussian_mean, gaussian_cov)
    gradient_f = make_negative_gaussian_derivative(objective_f, gaussian_mean, gaussian_cov)

