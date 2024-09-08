import numpy as np

class add:
    def __init__(self, base_input, attentionvector) -> None:
        #print(len(base_input[0]), len(attentionvector[0][0]))
        self.modified_input = base_input + attentionvector[0]

    def get_output(self):
        return self.modified_input

class LayerNormalization:
    def __init__(self, no_of_nodes = 512, batch_size = 32, gamma = 1, beta = 0, feed_vector = None, norm_vector = None):
        input = add(feed_vector, norm_vector)
        self.batch_output = np.array(input)
        self.no_of_nodes = no_of_nodes
        self.batch_size = batch_size
        self.gamma = np.array(gamma)  # Scale parameter
        self.beta = np.array(beta)    # Shift parameter

    def normalize(self):
        # Calculate the mean and standard deviation across the nodes for each sample
        mean = np.mean(self.batch_output, axis=1, keepdims=True)
        stand_dev = np.std(self.batch_output, axis=1, keepdims=True)
        
        # Normalize
        normalized_output = (self.batch_output - mean) / (stand_dev + 1e-5)  # Adding epsilon for numerical stability
        
        # Scale and shift
        normalized_output = normalized_output * self.gamma + self.beta
        
        return normalized_output




