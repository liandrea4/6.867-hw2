from plot_svm_boundary  import *
from svm                import solve_dual_svm_slack
import pylab            as pl
import numpy            as np
import sys

def predict_svm(weight_vector, x, b):
  prediction = np.dot(weight_vector, x) + b
  if prediction > 0:
    return 1
  elif prediction < 0:
    return -1
  else:
    return 0

def get_classification_error_rate(x, y, weight_vector, b):
  num_errors = 0
  for x_i, y_i in zip(x,y):
    prediction = predict_svm(weight_vector, x_i, b)
    if prediction != y_i:
      num_errors += 1

  return float(num_errors) / len(alpha_vals)

##############################################
################# TRAINING ###################
##############################################

def train_model(X, Y, file_num, C, threshold):
  alpha_vals = solve_dual_svm_slack(X, Y, C)

  weight_vector = np.array([0., 0.])
  for alpha_i, x_i, y_i in zip(alpha_vals, X, Y):
    weight_vector += alpha_i * y_i * x_i

  return alpha_vals, weight_vector

def calculate_b(weight_vector, x, y, alpha_vals, C, threshold, b_threshold):
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

###### MAIN ######
if __name__ == '__main__':
  file_num = sys.argv[1]
  C = 1.
  threshold = 0.00001
  b_threshold = 0.1

  ###### Training ######

  train = loadtxt('../data/data'+file_num+'_train.csv')
  x_training = train[:, 0:2].copy()
  y_training = train[:, 2:3].copy()

  alpha_vals, weight_vector = train_model(x_training, y_training, file_num, C, threshold)
  b = calculate_b(weight_vector, x_training, y_training, alpha_vals, C, threshold, b_threshold)
  plotDecisionBoundary(x_training, y_training, predict_svm, [-1, 0, 1], weight_vector, b, title = 'SVM Training, data' + str(file_num))

  training_error_rate = get_classification_error_rate(x_training, y_training, weight_vector, b)
  print "training_error_rate: ", training_error_rate

  ###### Validation ######

  validate = loadtxt('../data/data'+file_num+'_validate.csv')
  x_validate = validate[:, 0:2]
  y_validate = validate[:, 2:3]
  plotDecisionBoundary(x_validate, y_validate, predict_svm, [-1, 0, 1], weight_vector, b, title = 'SVM Validation, data' + str(file_num))

  validation_error_rate = get_classification_error_rate(x_validate, y_validate, weight_vector, b)
  print "validation_error_rate: ", validation_error_rate


