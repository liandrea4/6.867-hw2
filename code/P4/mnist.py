import sys
sys.path.append('../P1')
sys.path.append('../P2')
from generate_datasets    import *
from svm_test             import *
from logistic_regression  import *

print "Constructing datasets..."
all_datasets = make_all_datasets()
normalized_datasets = normalize_datasets(all_datasets)

training_a, validation_a, testing_a = normalized_datasets['4'][0], normalized_datasets['4'][1], normalized_datasets['4'][2]
training_b, validation_b, testing_b = normalized_datasets['9'][0], normalized_datasets['9'][1], normalized_datasets['9'][2]

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

C = 100.
threshold = 0.000001
b_threshold = 1
gamma = 1.
# kernel_fn = linear_kernel_fn
kernel_fn = make_gaussian_rbf_kernel_fn(gamma)
file_num = 'MNIST'

print "Running slack SVM..."
# run_slack_var_svm(x_training, y_training, x_testing, y_testing, C, threshold, b_threshold, file_num)
run_kernel_svm_validation(x_training, y_training, x_validate, y_validate, x_testing, y_testing, threshold, b_threshold)

# print "Running logistic regression..."
# logistic_regression(x_training, y_training, x_validate, y_validate, x_testing, y_testing)



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
