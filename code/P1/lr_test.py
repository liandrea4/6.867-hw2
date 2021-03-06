import numpy
from numpy import *
# from plotBoundary import *
import pylab as pl
from logistic_regression import logistic_loss, L2_regularization, create_L2_logistic_objective, create_Logistic_predictor
from gradient_descent import gradient_descent,  make_numeric_gradient_calculator, plot_gradient_descent
from sklearn                import linear_model
from plotBoundary import   plotDecisionBoundary
# import your LR training code

# parameters
name = '3'
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

L1_logistic_regressor = linear_model.LogisticRegression(penalty = 'l1', tol =0.001, C = 10**20)
L2_logistic_regressor = linear_model.LogisticRegression(penalty = 'l2', tol =0.001, C = 1)

L2_logistic_regressor.fit(X, Y)
L1_logistic_regressor.fit(X, Y)

predictor = create_Logistic_predictor(L1_logistic_regressor)

##Define the predictLR(x) function, which uses trained parameters
# def predictLR(x, regressor = L2_logistic_regressor):
# 	return regressor.predict

# plot training results
plotDecisionBoundary(X, Y, predictor, [0.5], title = 'LR Train')
pl.show()

# print '======Validation======'
# # load data from csv files
# validate = loadtxt('../data/data'+name+'_validate.csv')
# X_v = validate[:,0:2]
# Y_v = validate[:,2:3]

# # plot validation results
# plotDecisionBoundary(X_v, Y_v, predictor, [0.5], title = 'LR Validate')
# pl.show()


# print '======Test======'
# # load data from csv files
# test = loadtxt('../data/data'+name+'_test.csv')
# X_t = test[:,0:2]
# Y_t = test[:,2:3]

# # plot validation results
# plotDecisionBoundary(X_t, Y_t, predictor, [0.5], title = 'LR Test (lambda = 0, L2, data1)')
# pl.show()
