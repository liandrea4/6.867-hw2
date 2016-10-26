import sys
sys.path.append('../P2')
import matplotlib.pyplot as plt
import pylab             as pl
from sklearn             import linear_model
from plot_svm_boundary   import *
from svm                 import make_gaussian_rbf_kernel_fn, linear_kernel_fn
import numpy
import math


def find_L2_margin(weight):
	cum_sum = 0
	for i in range(len(weight)):
		cum_sum += weight[i]**2
	return cum_sum**(0.5)

def predict_pegasos_kernel(x, x_training, y_training, alpha_vals, b, kernel_fn):
  summed = 0
  for x_i, y_i, alpha_i in zip(x_training, y_training, alpha_vals):
    summed += alpha_i * kernel_fn(x_i, x)

  prediction = summed + b
  if prediction > 0:
    return 1
  elif prediction < 0:
    return -1
  else:
    return 0

def get_classification_error_rate_kernel_pegasos(x, y, alpha_vals, b, kernel_fn):
  num_errors = 0
  for x_i, y_i in zip(x,y):
    prediction = predict_pegasos_kernel(x_i, x, y, alpha_vals, b, kernel_fn)
    if prediction != y_i:
      num_errors += 1

  return float(num_errors) / len(x)

def run_kernalized_pegasos(X, Y, reg_parameter, K, max_epochs):
	t = 0
	epoch = 0
	A = numpy.array([0.0] * len(X))
	step_size = 0

	while(epoch < max_epochs):
		epoch +=1

		for i in range(len(X)):
			t+=1
			step_size = (1/(t * reg_parameter))
			inner_product = 0

			for j in range(len(X)):
				inner_product += numpy.dot(A[j], K(X[j], X[i]))

			a = numpy.dot((1 - step_size*reg_parameter), A[i])
			if numpy.dot(Y[i], inner_product) < 1:
				b = numpy.dot(step_size, Y[i])
				A[i] = a + b[0]

			else:
<<<<<<< HEAD
				A[i] = numpy.dot((1 - step_size*reg_parameter), A[i])
			print "t", t

	print "number of non-zero SVMs", numpy.count_nonzero(A)
	print "total length of alpha", len(A)
=======
				A[i] = a

>>>>>>> b210998ad9c294c02ec93ed3aeb181362ab39234
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
  file_num = sys.argv[1]

  epochs = 30
  lmbda = 0.02
  # kernel_fn = linear_kernel_fn
  kernel_fn = make_gaussian_rbf_kernel_fn(0.02)
  b = 0

  train = loadtxt('../data/data'+file_num+'_train.csv')
  x_training = train[:, 0:2].copy()
  y_training = train[:, 2:3].copy()

<<<<<<< HEAD
	epochs = 10;
	lmbda = 0.02;
	gauss_kernel = make_gaussian_rbf_kernel_fn(2**(2))
=======
  print "calculating alphas..."
  alpha_vals = run_kernalized_pegasos(x_training, y_training, lmbda, kernel_fn, epochs)
  print "alpha_vals: ", alpha_vals, len(alpha_vals)
>>>>>>> b210998ad9c294c02ec93ed3aeb181362ab39234

  training_error = get_classification_error_rate_kernel_pegasos(x_training, y_training, alpha_vals, b, kernel_fn)
  print "training_error: ", training_error

  plotDecisionBoundary_kernel(x_training, y_training, predict_pegasos_kernel, [-1, 0, 1], alpha_vals, b, kernel_fn,
    title = 'Pegasos Training, data' + str(file_num))


