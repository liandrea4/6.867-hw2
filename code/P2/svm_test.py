from plot_svm_boundary  import *
from svm                import solve_dual_svm_slack, solve_dual_svm_kernel, linear_kernel_fn, make_gaussian_rbf_kernel_fn
import pylab            as pl
import numpy            as np
import sys

def predict_svm_slack(weight_vector, x, b):
  prediction = np.dot(weight_vector, x) + b
  if prediction > 0:
    return 1
  elif prediction < 0:
    return -1
  else:
    return 0

def predict_svm_kernel(x, x_training, y_training, alpha_vals, b, kernel_fn):
  summed = 0
  for x_i, y_i, alpha_i in zip(x_training, y_training, alpha_vals):
    summed += y_i * alpha_i * kernel_fn(x_i, x)

  prediction = summed + b
  if prediction > 0:
    return 1
  elif prediction < 0:
    return -1
  else:
    return 0

def get_classification_error_rate_slack(x, y, weight_vector, b):
  num_errors = 0
  for x_i, y_i in zip(x,y):
    prediction = predict_svm_slack(weight_vector, x_i, b)
    if prediction != y_i:
      num_errors += 1

  return float(num_errors) / len(x)

def get_classification_error_rate_kernel(x, y, alpha_vals, b, kernel_fn):
  num_errors = 0
  for x_i, y_i in zip(x,y):
    prediction = predict_svm_kernel(x_i, x, y, alpha_vals, b, kernel_fn)
    if prediction != y_i:
      num_errors += 1

  return float(num_errors) / len(x)

##############################################
################# TRAINING ###################
##############################################

def train_model(X, Y, alpha_vals, threshold):
  weight_vector = np.array([0., 0.])
  for alpha_i, x_i, y_i in zip(alpha_vals, X, Y):
    weight_vector += alpha_i * y_i * x_i

  return weight_vector

def calculate_b_slack(weight_vector, x, y, alpha_vals, C, threshold, b_threshold):
  old_b = None
  for x_i, y_i, alpha_i in zip(x, y, alpha_vals):
    if alpha_i > threshold and alpha_i < C - threshold: ### NOTE THIS IN THE HW, THRESHOLD VS 0
      product = np.dot(weight_vector, x_i)
      b = y_i - product

      if old_b is not None:
        if abs(old_b - b) > b_threshold:
          raise Exception("Different values of b: " + str(old_b) + " and " + str(b))

      old_b = b

  return old_b

def calculate_b_kernel(x, y, alpha_vals, C, threshold, b_threshold, kernel_fn):
  old_b = None
  for x_i, y_i in zip(x, y):
    summed = 0
    for x_j, alpha_j in zip(x, alpha_vals):
      if alpha_j > threshold and alpha_j < C - threshold:
        summed += alpha_j * kernel_fn(x_i, x_j)

    b = y_i - summed
    if old_b is not None:
      if abs(old_b - b) > b_threshold:
        raise Exception("Different values of b: " + str(old_b) + " and " + str(b))

    old_b = b

  return old_b


###### Big picture methods ######

def run_slack_var_svm(file_num, C, threshold, b_threshold):

  ###### Training ######

  train = loadtxt('../data/data'+file_num+'_train.csv')
  x_training = train[:, 0:2].copy()
  y_training = train[:, 2:3].copy()

  alpha_vals = solve_dual_svm_slack(x_training, y_training, C)
  weight_vector = train_model(x_training, y_training, alpha_vals, threshold)
  b = calculate_b_slack(weight_vector, x_training, y_training, alpha_vals, C, threshold, b_threshold)
  plotDecisionBoundary_slack(x_training, y_training, predict_svm_slack, [-1, 0, 1], weight_vector, b, title = 'SVM Training, data' + str(file_num))

  training_error_rate = get_classification_error_rate_slack(x_training, y_training, weight_vector, b)
  print "training_error_rate: ", training_error_rate

  ###### Validation ######

  validate = loadtxt('../data/data'+file_num+'_validate.csv')
  x_validate = validate[:, 0:2]
  y_validate = validate[:, 2:3]
  plotDecisionBoundary_slack(x_validate, y_validate, predict_svm_slack, [-1, 0, 1], weight_vector, b, title = 'SVM Validation, data' + str(file_num))

  validation_error_rate = get_classification_error_rate_slack(x_validate, y_validate, weight_vector, b)
  print "validation_error_rate: ", validation_error_rate


def run_kernel_svm(file_num, C, threshold, b_threshold, gamma):

  ###### Training ######

  train = loadtxt('../data/data'+file_num+'_train.csv')
  x_training = train[:, 0:2].copy()
  y_training = train[:, 2:3].copy()
  # kernel_fn = linear_kernel_fn
  kernel_fn = make_gaussian_rbf_kernel_fn(gamma)

  alpha_vals = solve_dual_svm_kernel(x_training, y_training, C, kernel_fn)
  b = calculate_b_kernel(x_training, y_training, alpha_vals, C, threshold, b_threshold, kernel_fn)
  plotDecisionBoundary_kernel(x_training, y_training, predict_svm_kernel, [-1, 0, 1], x_training, y_training, alpha_vals, b, kernel_fn,
    title = 'SVM Training, data' + str(file_num))

  training_error_rate = get_classification_error_rate_kernel(x_training, y_training, alpha_vals, b, kernel_fn)
  print "training_error_rate: ", training_error_rate

  ###### Validation ######

  validate = loadtxt('../data/data'+file_num+'_validate.csv')
  x_validate = validate[:, 0:2]
  y_validate = validate[:, 2:3]
  plotDecisionBoundary_kernel(x_validate, y_validate, predict_svm_kernel, [-1, 0, 1], x_validate, y_validate, alpha_vals, b, kernel_fn,
   title = 'SVM Validation, data' + str(file_num))

  validation_error_rate = get_classification_error_rate_kernel(x_validate, y_validate, alpha_vals, b, kernel_fn)
  print "validation_error_rate: ", validation_error_rate




###### MAIN ######
if __name__ == '__main__':
  file_num = sys.argv[1]
  C = 1.
  threshold = 0.0001
  b_threshold = 3
  gamma = 1.

  # run_slack_var_svm(file_num, C, threshold, b_threshold)

  run_kernel_svm(file_num, C, threshold, b_threshold, gamma)

