import numpy as np

# Function to calculate the sigmoid activation function
# Gives the output in the range of 0 to 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    # constructor to initialize the weights and bias of the neuron
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # Function to perform the feedforward operation of the neuron
    # (w1 * x1) + (w2 * x2) + b = total 
    # then apply the sigmoid activation function to the total
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0, 1])  # w1 = 0, w2 = 1
bias = 4                    # b = 4
n = Neuron(weights, bias)   # creates the neuron with the specified weights and bias

x = np.array([2, 3])        # x1 = 2, x2 = 3
print(n.feedforward(x))     # Output: 0.9990889488055994
