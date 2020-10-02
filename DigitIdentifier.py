"""
This code consists of a 2-layer neural network that takes in a 784-D vector,
which corresponds to a grid, the parameter values of which correspond to cells
in that grid.  Each parameter is a number between zero and one which describes
that cell's grayscale value.  The output of the network is which digit the
pattern of cells in the grid looks the most like.
"""

import numpy as np


def get_result(output_vector: np.array((10, 1))):
    """return the digit according to the network"""
    max_index = 0
    max_val = -np.inf
    for k in range(10):
        num = output_vector[k][0]
        if num > max_val:
            max_index = k
            max_val = num
    return max_index


def ReLU(x):
    """implements ReLU"""
    return np.maximum(0, x)


def D_ReLu(x):
    """This is the derivative of the ReLu truncation function."""
    for row in x:
        for k in range(len(row)):
            if row[k] > 0:
                row[k] = 1
            else:
                row[k] = 0
    return x


def softmax(x):
    """implements softmax"""
    numerator = np.exp(x - np.maximum(0, x))
    denominator = np.sum(numerator)
    return numerator / denominator


def DL_Dz(y, y_hat):
    """implements derivative of loss with respect to (aw + b)"""
    return y_hat - y


'''
def make_delta_output(v: np.array((10, 1)), y: np.array((10, 1)), z: np.array((10, 1))):
    """return the vector of errors associated with a given layer"""
    nabla_a_C = (2 / 10) * (v - y)
    ReLU_prime_z = D_ReLu(z)
    delta = np.multiply(nabla_a_C, ReLU_prime_z)
    return delta'''


def make_delta(weights, delta, z):
    """weights in this particular case should be a """
    return np.multiply(np.dot(weights.T, delta), D_ReLu(z))


class DigitIdentifier:
    """This class defines a neural network to identify digits as they appear on a 28x28 grid."""

    def __init__(self):
        """inputs:
                examples: a set of examples, with input vector x with dimensions [1, 784]
                and output vector y with dimensions [1, 10]
                """

        # initialize the hidden layer (128 nodes) and the output layer(10-D vector)
        self.hidden_layer = np.random.rand(128, 1)
        self.output = np.random.rand(10, 1)

        # initialize weights and biases for hidden layer and output using He initialization
        self.hidden_weights = np.random.rand(128, 784) * np.sqrt(2 / 784)
        self.hidden_biases = np.random.rand(128, 1) * np.sqrt(2 / 128)
        self.output_weights = np.random.rand(10, 128) * np.sqrt(2 / 128)
        self.output_biases = np.random.rand(10, 1) * np.sqrt(2 / 10)

        # initialize list of results
        self.results = []
        self.accuracy = 0

    def mini_batch_GD(self, examples: list, epochs=20, learning_rate=1, batch_size=1):
        """Apply mini-batch GD back-propagation to network with set of examples
                inputs:
                    examples:   a list of examples with which to train the network
                    iterations: the number of times the user wants to train the network,
                        defaults to 1000
                    eta:    the numerator of the learning rate, defaults to 1000 as the
                        network is built for a training set of 60,000 examples
        """
        # reorganize data into batches to perform a mini_batch GD
        batches = []
        example_idx = 0
        for k in range(int(len(examples) / batch_size)):
            batch = []
            for i in range(batch_size):
                batch.append(examples[example_idx])
                example_idx += 1
            batches.append(batch)
        # train the network num_trainings times
        for k in range(0, epochs):
            for batch in batches:
                nabla_w_output = []
                nabla_w_hidden = []

                nabla_b_output = []
                nabla_b_hidden = []
                for example in batch:
                    # propagate the inputs forward to compute the outputs
                    (x, y) = example
                    z_hidden = np.dot(self.hidden_weights, x) + self.hidden_biases
                    self.hidden_layer = ReLU(z_hidden)

                    z_L = np.dot(self.output_weights, self.hidden_layer) + self.output_biases
                    self.output = softmax(z_L)

                    # find derivatives of loss function with respect to (aw + b)
                    delta_L = DL_Dz(y, self.output)  # delta_L is 10D vector
                    delta_hidden = make_delta(self.output_weights, delta_L, z_hidden)  # delta_hidden is 128D vector

                    # calculate cost gradient vector
                    nabla_w_output.append(np.dot(delta_L, self.hidden_layer.T))
                    nabla_w_hidden.append(np.dot(delta_hidden, x.T))

                    nabla_b_output.append(delta_L)
                    nabla_b_hidden.append(delta_hidden)

                # calculate average gradient over all examples for weights and biases
                avg_grad_output_w = sum(nabla_w_output) / len(nabla_w_output)
                avg_grad_hidden_w = sum(nabla_w_hidden) / len(nabla_w_hidden)

                avg_grad_output_b = sum(nabla_b_output) / len(nabla_b_output)
                avg_grad_hidden_b = sum(nabla_b_hidden) / len(nabla_b_hidden)

                # update weights and biases
                self.output_weights -= learning_rate * avg_grad_output_w
                self.output_biases -= learning_rate * avg_grad_output_b

                self.hidden_weights -= learning_rate * avg_grad_hidden_w
                self.hidden_biases -= learning_rate * avg_grad_hidden_b
        return

    def run(self, examples: list):
        """run the network on a set of examples"""
        self.results = [0] * len(examples)
        test_outputs = []
        for idx, example in enumerate(examples):
            # propagate the inputs forward to compute the outputs
            (vector, output_vec) = example
            z_hidden = np.dot(self.hidden_weights, vector) + self.hidden_biases
            self.hidden_layer = ReLU(z_hidden)

            z_L = np.dot(self.output_weights, self.hidden_layer) + self.output_biases
            self.output = softmax(z_L)

            # update result list
            self.results[idx] = get_result(self.output)
            output = get_result(output_vec)
            test_outputs.append(output)

        # calculate accuracy of the neural network
        right_answers = 0
        for k in range(len(test_outputs)):
            if test_outputs[k] == self.results[k]:
                right_answers += 1
        self.accuracy = right_answers / len(test_outputs)
        return
