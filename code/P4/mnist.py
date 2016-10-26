import sys
sys.path.append('../P1')
sys.path.append('../P2')
sys.path.append('../P3')
from generate_datasets    import *
from svm_test             import *
from logistic_regression  import *
from pegasos              import run_pegasos
from datetime             import datetime

def test_pegasos(x_training, y_training, x_validate, y_validate, x_testing, y_testing, weight_vector, b):
  training_error_rate, training_errors = get_classification_error_rate_slack(x_training, y_training, weight_vector, b)
  print "training_error_rate: ", training_error_rate

  validation_error_rate, validation_errors = get_classification_error_rate_slack(x_validate, y_validate, weight_vector, b)
  print "validation_error_rate: ", validation_error_rate

  testing_error_rate, testing_errors = get_classification_error_rate_slack(x_testing, y_testing, weight_vector, b)
  print "testing_error_rate: ", testing_error_rate

first_dataset = '4'
second_dataset = '9'

print "Constructing datasets..."
all_datasets = make_all_datasets()
normalized_datasets = normalize_datasets(all_datasets)

training_a, validation_a, testing_a = normalized_datasets[first_dataset][0], normalized_datasets[first_dataset][1], normalized_datasets[first_dataset][2]
training_b, validation_b, testing_b = normalized_datasets[second_dataset][0], normalized_datasets[second_dataset][1], normalized_datasets[second_dataset][2]

x_training = training_a + training_b
y_training = [1] * num_training + [-1] * num_training
x_validate = validation_a + validation_b
y_validate = [1] * num_validation + [-1] * num_validation
x_testing = testing_a + testing_b
y_testing = [1] * num_testing + [-1] * num_testing

# x_training, x_validate, x_testing = [], [], []
# for value in normalized_datasets.values():
#   x_training += value[0]
#   x_validate += value[1]
#   x_testing += value[2]
# y_training = ( [1] * num_training + [-1] * num_training ) * 5
# y_validate = ( [1] * num_validation + [-1] * num_validation ) * 5
# y_testing = ( [1] * num_testing + [-1] * num_testing ) * 5

C = 1.
threshold = 0.000001
b_threshold = 5
gamma = 1.
# kernel_fn = linear_kernel_fn
kernel_fn = make_gaussian_rbf_kernel_fn(gamma)
max_epochs = 100
lambda_val = 200
file_num = 'MNIST'

# print "Running SVM..."
# start = datetime.now()
# run_slack_var_svm(x_training, y_training, x_validate, y_validate, x_testing, y_testing, C, threshold, b_threshold)
# # run_slack_var_svm_validation(x_training, y_training, x_validate, y_validate, x_testing, y_testing, threshold, b_threshold)
# slack_duration = datetime.now() - start

# start = datetime.now()
# run_kernel_svm(x_training, y_training, x_validate, y_validate, x_testing, y_testing, kernel_fn, C, threshold, b_threshold, gamma)
# # run_kernel_svm_validation(x_training, y_training, x_validate, y_validate, x_testing, y_testing, kernel_fn, threshold, b_threshold)
# kernel_duration = datetime.now() - start


# print "Running logistic regression..."
# start = datetime.now()
# logistic_regression(x_training, y_training, x_validate, y_validate, x_testing, y_testing)
# logistic_duration = datetime.now() - start


print "Running pegasos..."
start = datetime.now()
bias, weight_vector = run_pegasos(x_training, y_training, lambda_val, max_epochs)
test_pegasos(x_training, y_training, x_validate, y_validate, x_testing, y_testing, weight_vector, bias)
pegasos_duration = datetime.now() - start


print "datasets: ", first_dataset, second_dataset
# print "slack duration: ", slack_duration
# print "kernel_duration: ", kernel_duration
# print "logistic_duration: ", logistic_duration
print "pegasos_duration: ", pegasos_duration


# SVM (slack):

# 1 vs 7:
# normalized:
#   training: 0.0
#   testing: 0.0133
# not normalized:
#   training: 0.0
#   testing: 0.0133

# 3 vs 5:
# normalized:
#   training: 0.0
#   testing: 0.0533
# normalized:
#   training: 0.0
#   testing: 0.0533

# 4 vs 9:
# normalized:
#   training: 0.0
#   testing: 0.056667
# normalized:
#   training: 0.0
#   testing: 0.05667

# even vs. odd:
# not normalized:
#   training: 0.0
#   testing: 0.1633
# normalized:
#   training: 0.0
#   testing: 0.1633
