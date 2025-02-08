import numpy as np

class Conv3x3:

    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3, 3) * np.sqrt(1 / (3 * 3 * 3))
        self.biases = np.zeros(num_filters)

    def iterate_regions(self, image):
        h, w, _ = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j
    
    def forward(self, input):
        input = input / 255.0  # Scale to [0, 1]
        self.last_input = input
        h,w, _ = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region * self.filters, axis=(1,2,3))
        
        return output
    
    def backward(self, d_L_d_out, learning_rate):
        """
        Backpropagation for Conv3x3 layer.
        - d_L_d_out: Gradient of the loss with respect to the output of this layer.
        - learning_rate: Learning rate for gradient descent.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_biases = np.zeros(self.biases.shape)
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Gradient of loss with respect to filters
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                # Gradient of loss with respect to biases
                d_L_d_biases[f] += d_L_d_out[i, j, f]
                # Gradient of loss with respect to input (for passing to previous layer)
                d_L_d_input[i:i+3, j:j+3, :] += d_L_d_out[i, j, f] * self.filters[f]

        np.clip(d_L_d_filters, -1, 1, out=d_L_d_filters)
        np.clip(d_L_d_biases, -1, 1, out=d_L_d_biases)
        self.filters -= learning_rate * d_L_d_filters
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_input
