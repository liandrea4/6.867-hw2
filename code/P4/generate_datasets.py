import os

num_training = 200
num_validation = 150
num_testing = 150
filepath = "../data/mnist_digit_"

def make_dataset(digit_num):
  dataset = []
  with open(filepath + str(digit_num) + ".csv", 'r') as f:
    for line in f:
      dataset.append(line.split(" "))

  training = dataset[:num_training]
  validation = dataset[num_training:(num_training + num_validation)]
  testing = dataset[(num_training + num_validation):(num_training + num_validation + num_testing)]

  return training, validation, testing

def make_all_datasets():
  all_datasets = {}

  for i in range(10):
    training, validation, testing = make_dataset(i)
    all_datasets[i] = [training, validation, testing]

    assert len(training) == num_training
    assert len(validation) == num_validation
    assert len(testing) == num_testing

  return all_datasets

def normalize_datasets(all_datasets):
  normalized_dict = {}

  for key in all_datasets.keys():
    normalized_dict[key] = []

    for number_datasets in all_datasets[key]:
      normalized_number_datasets = []
      for dataset in number_datasets:
        normalized = [ 2 * float(elem) / 255. - 1 for elem in dataset ]
      normalized_number_datasets.append(normalized)

    normalized_dict[key].append(normalized_number_datasets)

  return normalized_dict

all_datasets = make_all_datasets()
normalized_dataset = normalize_datasets(all_datasets)
for list_of_datasets in normalized_dataset.values():
  for dataset in list_of_datasets:
    for image in dataset:
      for elem in image:
        if elem > 1 or elem < -1:
          raise Exception(elem)


# format of dictionary:
# normalized = {
# 1: [training, validation, testing],
# 2: [ [image, image, image, ...], ...],
# 3: [ [ [1,2,3,4,5,...], [1,2,3,4,5,...], ...], ...]
# ...
# }

