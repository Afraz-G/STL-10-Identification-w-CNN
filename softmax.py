import numpy as np

class Softmax:
    def __init__(self, input_len, nodes):
        self.input_len = input_len
        self.nodes = nodes
        # Xavier Initialization for stable weight distribution
        self.weights = np.random.randn(input_len, nodes) * np.sqrt(1 / input_len)
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input  # Save input for backpropagation

        # Normalize input to prevent large values
        input = (input - np.mean(input)) / (np.std(input) + 1e-8)

        # Compute raw totals before softmax
        totals = np.dot(input, self.weights) + self.biases
        totals = (totals - np.mean(totals)) / (np.std(totals) + 1e-8)

        # Apply numerical stability trick
        exp = np.exp(totals - np.max(totals))

        self.last_output = exp / np.sum(exp, axis=0)  # Softmax output
        return self.last_output

    def backward(self, d_L_d_output, learning_rate):
        d_L_d_totals = np.zeros_like(self.last_output)

        for i, (output, d_L_d_output_i) in enumerate(zip(self.last_output, d_L_d_output)):
            d_L_d_totals[i] = output * (d_L_d_output_i - np.sum(self.last_output * d_L_d_output))

        # Compute gradients
        d_L_d_weights = np.outer(self.last_input, d_L_d_totals)
        d_L_d_biases = d_L_d_totals
        d_L_d_input = np.dot(d_L_d_totals, self.weights.T)

        # Gradient clipping to prevent instability
        np.clip(d_L_d_weights, -1, 1, out=d_L_d_weights)
        np.clip(d_L_d_biases, -1, 1, out=d_L_d_biases)


        # Update weights and biases
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_input.reshape(self.last_input_shape)
