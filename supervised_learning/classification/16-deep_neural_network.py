import numpy as np

class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network
        nx: number of input features
        layers: list representing the number of nodes in each layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            # Validate layer elements
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")
            
            # Determine input size for the current layer
            # If it's the first layer, input size is nx, else it's nodes in previous layer
            prev_layer_size = nx if l == 0 else layers[l-1]
            
            # He et al. initialization
            self.weights[f"W{l + 1}"] = np.random.randn(layers[l], prev_layer_size) * np.sqrt(2 / prev_layer_size)
            # Bias initialized to zeros
            self.weights[f"b{l + 1}"] = np.zeros((layers[l], 1))