import sys
sys.path.append('../P2')
from generate_datasets    import *
from svm_test             import *

all_datasets = make_all_datasets()
normalized_datasets = normalize_datasets(all_datasets)

training_1, validation_1 = normalized_datasets['1'][0], normalized_datasets['1'][1]
training_7, validation_7 = normalized_datasets['7'][0], normalized_datasets['7'][1]

x_training = training_1 + training_7
y_training = [1] * num_training + [-1] * num_training
x_validate = validation_1 + validation_7
y_validate = [1] * num_testing + [-1] * num_testing
print "Constructed datasets..."

C = 1.
threshold = 0.000001
b_threshold = 0.1
gamma = 1.
kernel_fn = linear_kernel_fn
file_num = 'MNIST'

print "Running slack SVM..."
run_slack_var_svm(x_training, y_training, x_validate, y_validate, C, threshold, b_threshold, file_num)
# run_kernel_svm(x_training, y_training, x_validate, y_validate, kernel_fn, C, threshold, b_threshold, gamma, file_num)


