import sys
sys.path.append('../')
from plotBoundary import *
from numpy        import *
from svm          import solve_dual_svm_slack, get_classification_error_rate
import pylab      as pl


def train_model(file_num, C, threshold):
  print '======Training======'
  train = loadtxt('../data/data'+file_num+'_train.csv')

  # use deep copy here to make cvxopt happy
  X = train[:, 0:2].copy()
  Y = train[:, 2:3].copy()
  C = 1.

  # Carry out training, primal and/or dual
  alpha_vals = solve_dual_svm_slack(X, Y, C)
  return alpha_vals







# Define the predictSVM(x) function, which uses trained parameters
def predict_svm():
  pass

# plot training results
# plotDecisionBoundary(X, Y, predict_svm, [-1, 0, 1], title = 'SVM Training')


def validate_model(file_num):
  print '======Validation======'

  validate = loadtxt('data/data'+file_num+'_validate.csv')
  X = validate[:, 0:2]
  Y = validate[:, 2:3]

  # plot validation results
  plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
  pl.show()



if __name__ == '__main__':
  file_num = sys.argv[1]
  C = 1.
  threshold = 0.00001

  alpha_vals = train_model(file_num, C, threshold)

  classification_error_rate = get_classification_error_rate(alpha_vals, C, threshold)
  print "classification_error_rate: ", classification_error_rate

