from plot_svm_boundary   import *
from svm                 import solve_dual_svm_slack, solve_dual_svm_kernel, linear_kernel_fn, make_gaussian_rbf_kernel_fn
import pylab             as pl
import numpy             as np
import matplotlib.pyplot as plt
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
  errors = []
  for x_i, y_i in zip(x,y):
    prediction = predict_svm_slack(weight_vector, x_i, b)
    if prediction != y_i:
      errors.append(x_i)

  return float(len(errors)) / len(x), errors

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
  weight_vector = np.array([0.] * len(X[0]))
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

  num_support_vectors = 0
  for x_i, y_i, alpha_i in zip(x, y, alpha_vals):
    if alpha_i > threshold and alpha_i < C - threshold:
      num_support_vectors += 1

      product = sum([ alpha_vals[j] * y[j] * kernel_fn(x_i, x[j]) for j in range(len(x)) ])
      b = y_i - product

      if old_b is not None:
        if abs(old_b - b) > b_threshold:
          raise Exception("Different values of b: " + str(old_b) + " and " + str(b))

      old_b = b

  return old_b, num_support_vectors

###### Big picture methods ######

def run_slack_var_svm(x_training, y_training, x_validate, y_validate, x_testing, y_testing, C, threshold, b_threshold):

  ###### Training ######

  alpha_vals = solve_dual_svm_slack(x_training, y_training, C)
  weight_vector = train_model(x_training, y_training, alpha_vals, threshold)
  # print "weight_vector: ", weight_vector

  b = calculate_b_slack(weight_vector, x_training, y_training, alpha_vals, C, threshold, b_threshold)
  print "b: ", b

  # plotDecisionBoundary_slack(x_training, y_training, predict_svm_slack, [-1, 0, 1], weight_vector, b,
  #   title = 'SVM Training, data' + str(file_num))

  training_error_rate, training_errors = get_classification_error_rate_slack(x_training, y_training, weight_vector, b)
  print "training_error_rate: ", training_error_rate

  ###### Validation ######

  # plotDecisionBoundary_slack(x_validate, y_validate, predict_svm_slack, [-1, 0, 1], weight_vector, b,
  #   title = 'SVM Validation, data' + str(file_num))

  validation_error_rate, validation_errors = get_classification_error_rate_slack(x_validate, y_validate, weight_vector, b)
  print "validation_error_rate: ", validation_error_rate

  testing_error_rate, testing_errors = get_classification_error_rate_slack(x_testing, y_testing, weight_vector, b)
  print "testing_error_rate: ", testing_error_rate

  # for error in validation_errors:
  #   casted_array = np.array([ float(val) for val in error ])
  #   reshaped_array = np.reshape(casted_array, (28, 28))
  #   plt.imshow(reshaped_array, cmap='Greys_r')
  #   plt.title("classification=1, actual=7")
  #   plt.show()


def run_kernel_svm(x_training, y_training, x_validate, y_validate, x_testing, y_testing, kernel_fn, C, threshold, b_threshold, gamma):

  ###### Training ######

  print "Solving for alphas..."
  alpha_vals = solve_dual_svm_kernel(x_training, y_training, C, kernel_fn)
  # weight_vector = train_model(x_training, y_training, alpha_vals, threshold)

  print "Solving for b..."
  b, num_support_vectors = calculate_b_kernel(x_training, y_training, alpha_vals, C, threshold, b_threshold, kernel_fn)
  print "b: ", b

  print "Plotting decision boundary..."
  # plotDecisionBoundary_kernel(x_training, y_training, predict_svm_kernel, [-1, 0, 1], alpha_vals, b, kernel_fn,
  #   title = 'SVM Training, data' + str(file_num) + ', C=' + str(C))

  training_error_rate = get_classification_error_rate_kernel(x_training, y_training, alpha_vals, b, kernel_fn)
  print "training_error_rate: ", training_error_rate

  ###### Validation ######

  # plotDecisionBoundary_kernel(x_training, y_training, predict_svm_kernel, [-1, 0, 1], alpha_vals, b, kernel_fn,
  #   title = 'SVM Validation, data' + str(file_num) + ', C=' + str(C))

  validation_error_rate = get_classification_error_rate_kernel(x_validate, y_validate, alpha_vals, b, kernel_fn)
  print "validation_error_rate: ", validation_error_rate

  testing_error_rate = get_classification_error_rate_kernel(x_testing, y_testing, alpha_vals, b, kernel_fn)
  print "testing_error_rate: ", testing_error_rate


def run_slack_var_svm_validation(x_training, y_training, x_validate, y_validate, x_testing, y_testing, threshold, b_threshold):
  C_vals = [ 0.2, 0.4, 0.6, 0.8, 1 ]
  C_opt = 0
  b_opt = 0
  weights_opt = []
  min_error_rate = float('inf')

  for C in C_vals:

    alpha_vals = solve_dual_svm_slack(x_training, y_training, C)
    weight_vector = train_model(x_training, y_training, alpha_vals, threshold)

    b = calculate_b_slack(weight_vector, x_training, y_training, alpha_vals, C, threshold, b_threshold)
    print "b: ", b

    if b is None:
      print "No support vectors, skipping C=", C
      continue

    validation_error_rate, validation_errors = get_classification_error_rate_slack(x_validate, y_validate, weight_vector, b)
    print "validation_error_rate: ", validation_error_rate

    if validation_error_rate < min_error_rate:
      print "Updating C..."
      C_opt = C
      b_opt = b
      weights_opt = weight_vector
      min_error_rate = validation_error_rate

  print "C_opt: ", C_opt

  testing_error_rate, testing_errors = get_classification_error_rate_slack(x_testing, y_testing, weights_opt, b_opt)
  print "testing_error_rate: ", testing_error_rate


  # for error in validation_errors:
  #   casted_array = np.array([ float(val) for val in error ])
  #   reshaped_array = np.reshape(casted_array, (28, 28))
  #   plt.imshow(reshaped_array, cmap='Greys_r')
  #   plt.title("classification=1, actual=7")
  #   plt.show()


def run_kernel_svm_validation(x_training, y_training, x_validate, y_validate, x_testing, y_testing, kernel_fn, threshold, b_threshold):
  C_vals = [ 0.2, 0.4, 0.6, 0.8, 1 ]
  gamma_vals = [ 0.5, 1, 5, 10]

  C_opt = 0
  gamma_opt = 0
  b_opt = 0
  alphas_opt = []
  min_error_rate = float('inf')

  for C in C_vals:
    for gamma in gamma_vals:
      print "C: ", C, "  gamma: ", gamma

      print "Solving for alphas..."
      alpha_vals = solve_dual_svm_kernel(x_training, y_training, C, kernel_fn)

      print "Solving for b..."
      b, num_support_vectors = calculate_b_kernel(x_training, y_training, alpha_vals, C, threshold, b_threshold, kernel_fn)
      print "b: ", b, " num_support_vectors: ", num_support_vectors

      if b is None:
        print "No support vectors, skipping C=", C, ", gamma=", gamma, "... "
        continue

      # print "Calculating training error..."
      # training_error_rate = get_classification_error_rate_kernel(x_training, y_training, alpha_vals, b, kernel_fn)
      # print "training_error_rate: ", training_error_rate

      validation_error_rate = get_classification_error_rate_kernel(x_validate, y_validate, alpha_vals, b, kernel_fn)
      print "validation_error_rate: ", validation_error_rate

      if validation_error_rate < min_error_rate:
        print "Updating C and gamma..."
        C_opt = C
        gamma_opt = gamma
        b_opt = b
        alphas_opt = alpha_vals
        min_error_rate = validation_error_rate

  print "C_opt: ", C_opt, "   gamma_opt: ", gamma_opt

  testing_error_rate = get_classification_error_rate_kernel(x_testing, y_testing, alphas_opt, b_opt, kernel_fn)
  print "testing_error_rate: ", testing_error_rate





###### MAIN ######
if __name__ == '__main__':
  file_num = sys.argv[1]
  C = float(sys.argv[2])
  threshold = 0.000001
  b_threshold = 3
  gamma = 1.

  train = loadtxt('../data/data'+file_num+'_train.csv')
  x_training = train[:, 0:2].copy()
  y_training = train[:, 2:3].copy()

  validate = loadtxt('../data/data'+file_num+'_validate.csv')
  x_validate = validate[:, 0:2]
  y_validate = validate[:, 2:3]

  # kernel_fn = linear_kernel_fn
  kernel_fn = make_gaussian_rbf_kernel_fn(gamma)


  # run_slack_var_svm(x_training, y_training, x_validate, y_validate, C, threshold, b_threshold, file_num)
  run_kernel_svm(x_training, y_training, x_validate, y_validate, kernel_fn, C, threshold, b_threshold, gamma, file_num)

  # train = loadtxt('../data/data'+file_num+'_train.csv')
  # x_training = train[:, 0:2].copy()
  # y_training = train[:, 2:3].copy()
  # kernel_fn = linear_kernel_fn
  # # kernel_fn = make_gaussian_rbf_kernel_fn(gamma)

  # for C_val in [0.01, 0.1, 1, 10, 100]:
  #   alpha_vals = solve_dual_svm_kernel(x_training, y_training, C_val, kernel_fn)
  #   weight_vector = train_model(x_training, y_training, alpha_vals, threshold)
  #   norm = sum([w_i ** 2 for w_i in weight_vector ])
  #   print "C: ", C_val, "   norm: ", norm, "  geometric margin: ", 1/float(norm)


# gaussian:

# 1:
# training: 0
# validation: 0

# 2: C = 1
# training: 0.06
# validation: 0.115

# 2: C = 0.1
# training: 0.0825
# validation: 0.115

# 2: C = 100
# training: 0.035
# validation: 0.135

# 3: C =1
# training: 0.0175
# validation: 0.065

# 3: C = 0.01
# training: 0.0575
# validation: 0.195

# 3: C = 100
# training: 0.0125
# validation: 0.065

# 4: C = 1
# training: 0.0325
# validation: 0.05

# 4: C = 0.1
# training: 0.035
# validation: 0.04
