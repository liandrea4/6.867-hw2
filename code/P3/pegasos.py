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

def make_gaussian_rbf_kernel_fn(gamma):
  def gaussian_rbf_kernel_fn(x_i, x_j):
    x_i_np = numpy.array(x_i)
    x_j_np = numpy.array(x_j)
    magnitude = numpy.linalg.norm(x_i_np - x_j_np) ** 2
    return math.exp(-1 * gamma * magnitude)
  return gaussian_rbf_kernel_fn

def run_kernalized_pegasos(X, Y, reg_parameter, K, max_epochs):
	t = 0
	epoch = 0
	A = numpy.array([0.0] * len(X))
	step_size = 0

	while(epoch < max_epochs):
		epoch +=1
		print "epoch", epoch
		for i in range(len(X)):
			t+=1
			step_size = (1/(t * reg_parameter))
			inner_product = 0 
			for j in range(len(X)):
				inner_product += numpy.dot(A[j], K(X[j], X[i]))
			if numpy.dot(Y[i], inner_product) < 1:
				a = numpy.dot((1 - step_size*reg_parameter), A[i])
				print "a", a
				b = numpy.dot(step_size, Y[i])
				print "b", b[0]
				A[i] = numpy.add(a,b[0])
			else:
				A[i] = numpy.dot((1 - step_size*reg_parameter), A[i])
			print "t", t
	return A


def run_pegasos(X, Y, reg_parameter, max_epochs):
	t = 0
	epoch = 0
	weights_len = len(X[0])
	weights = numpy.array([0.0] * weights_len)
	weights.reshape(weights_len,1)
	weights_matrix = [weights] * (max_epochs * len(X)+2)
	weight_bias = 0

	step_size = 0
	while (epoch < max_epochs):
		epoch +=1

		for i in range(len(X)):
			t += 1
			step_size = (1/ (t * reg_parameter))

			inner_product = numpy.dot(weights_matrix[t], X[i])

			if numpy.dot(Y[i], inner_product) < 1:

				a = numpy.dot((1 - step_size*reg_parameter), weights_matrix[t])
				constant = step_size * Y[i]
				b = numpy.dot(constant, X[i])

				weights_matrix[t+1] = a+b
				weight_bias = weight_bias + constant

			else:
				weights_matrix[t+1] = numpy.dot((1 - step_size*reg_parameter) , weights_matrix[t])

	print "margin: ", 1.0/(find_L2_margin(weights_matrix[-1]))
	print "weight bias: ", weight_bias

 	return weight_bias, weights_matrix[-1]


if __name__ == '__main__':

	train = loadtxt('../data/data3_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	epochs = 10;
	lmbda = 2**(-2);
	gauss_kernel = make_gaussian_rbf_kernel_fn(2**1)


	print run_kernalized_pegasos(X, Y, lmbda, gauss_kernel, epochs)
	#print run_pegasos(X, Y, lmbda, epochs)


