"""
Parse MNIST gzip files such that the DigitIdentifier neural network
can effectively interpret them.
"""
import gzip
import numpy as np


def get_training_inputs():
    """Unpack MNIST training data and reshape """
    with gzip.open('train-images-idx3-ubyte.gz', 'r') as input_file:
        # collect training input data from gz file
        magic_num = int.from_bytes(input_file.read(4), 'big')
        num_images = int.from_bytes(input_file.read(4), 'big')
        num_rows = int.from_bytes(input_file.read(4), 'big')
        num_cols = int.from_bytes(input_file.read(4), 'big')
        data = input_file.read()
        images = np.frombuffer(data, dtype=np.uint8).reshape(
            (num_images, num_rows * num_cols))
    return images


def get_training_outputs():
    with gzip.open('train-labels-idx1-ubyte.gz', 'r') as output_file:
        # collect training output data from gz file
        magic_num = int.from_bytes(output_file.read(4), 'big')
        num_images = int.from_bytes(output_file.read(4), 'big')
        num_rows = int.from_bytes(output_file.read(4), 'big')
        num_cols = int.from_bytes(output_file.read(4), 'big')
        data = output_file.read()
        numbers = np.frombuffer(data, dtype=np.uint8)
    return numbers


def get_test_inputs():
    with gzip.open('t10k-images-idx3-ubyte.gz', 'r') as input_file:
        # collect training input data from gz file
        magic_num = int.from_bytes(input_file.read(4), 'big')
        num_images = int.from_bytes(input_file.read(4), 'big')
        num_rows = int.from_bytes(input_file.read(4), 'big')
        num_cols = int.from_bytes(input_file.read(4), 'big')
        data = input_file.read()
        images = np.frombuffer(data, dtype=np.uint8).reshape(
            (num_images, num_rows * num_cols))
    return images


def get_test_outputs():
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'r') as output_file:
        # collect training output data from gz file
        magic_num = int.from_bytes(output_file.read(4), 'big')
        num_images = int.from_bytes(output_file.read(4), 'big')
        num_rows = int.from_bytes(output_file.read(4), 'big')
        num_cols = int.from_bytes(output_file.read(4), 'big')
        data = output_file.read()
        numbers = np.frombuffer(data, dtype=np.uint8)
    return numbers


def organize_data(inputs, outputs):
    """takes resulting arrays from get_training_inputs and get_training_outputs
        and reorganizes them into a list of tuples to be passed into a neural network"""
    example_list = []
    for k, i in zip(inputs, outputs):
        x = k.reshape(784, 1)
        y = np.zeros((10, 1))
        y[i][0] = 1
        example_list.append((x, y))
    return example_list


def extract_samples(example_list, num):
    """extracts a specified number of examples form a data set"""
    data_set = []
    for k in range(num):
        data_set.append(example_list[k])
    return data_set
