import numpy
from numpy import *
# from plotBoundary import *
import pylab as pl
from logistic_regression import logistic_loss, L2_regularization, create_L2_logistic_objective
from gradient_descent import gradient_descent,  make_numeric_gradient_calculator, plot_gradient_descent
# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('../data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

weight_vector_length = len(X[0])
w0 = 0.0
reg_parameter = 0



initial_guess = numpy.array([100.0] * (weight_vector_length))

# Parameters for logistic regression

step_size = 100000000
threshold = 0.00001

# Carry out training.

objective_f = create_L2_logistic_objective(X, Y, w0, reg_parameter)

print objective_f(initial_guess)

gradient_f = make_numeric_gradient_calculator(objective_f, 0.001)

print gradient_f(initial_guess)

previous_values = gradient_descent(objective_f, gradient_f, initial_guess, step_size, threshold)
min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
print "min_x: ", min_x, "  min_y",  min_y
print "number of steps: ", len(previous_values)
print "w_mle: ", w_mle

plot_data(previous_values, 0)

# Define the predictLR(x) function, which uses trained parameters
### TODO ###

# # plot training results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

# print '======Validation======'
# # load data from csv files
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:,0:2]
# Y = validate[:,2:3]

# # plot validation results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
# pl.show()
