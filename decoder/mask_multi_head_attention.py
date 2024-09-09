import numpy as np
from math import sqrt
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')

class Mask_MultiHeadAttention:
    def __init__(self, num_heads=8, input = None):
        self.embeding_vectors = input
        self.no_words = len(self.embeding_vectors)
        self.dim = len(self.embeding_vectors[0][0])
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        
        assert self.dim % self.num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        # Initialize weights for each head
        self.query_weights = [np.random.rand(self.dim, self.head_dim) for _ in range(self.num_heads)]
        self.key_weights = [np.random.rand(self.dim, self.head_dim) for _ in range(self.num_heads)]
        self.value_weights = [np.random.rand(self.dim, self.head_dim) for _ in range(self.num_heads)]
        
        # Output projection weight
        self.fc_weight = np.random.rand(self.head_dim * self.num_heads, self.dim)
        self.calculate_new_values()
        self.calculate_dot()
        self.final_contextual_vector()

    def calculate_new_values(self):
        self.queries, self.keys, self.values = [], [], []
        for i in range(self.num_heads):
            query = np.dot(self.embeding_vectors, self.query_weights[i])
            key = np.dot(self.embeding_vectors, self.key_weights[i])
            value = np.dot(self.embeding_vectors, self.value_weights[i])
            self.queries.append(query)
            self.keys.append(key)
            self.values.append(value)

    def softmax(self, matrix, mask=None):
        if mask is not None:
            matrix += mask  # Apply mask
        for mat in matrix:
            s = np.sum(mat)
            for idx in range(len(mat)):
                mat[idx] = mat[idx] / s
        return matrix

    def calculate_dot(self):
        self.attention_outputs = []
        for i in range(self.num_heads):
            new_value = []
            for q in self.queries[i]:
                lis = []
                for k in self.keys[i]:
                    lis.append(np.dot(q, np.transpose(k)))
                new_value.append(lis)
            new_value = self.div(new_value, sqrt(self.head_dim))
            new_value = self.softmax(new_value, mask=self.create_mask())
            # Multiply with values to get attention output
            attention_output = np.dot(new_value, self.values[i])
            self.attention_outputs.append(attention_output)
    
    def create_mask(self):
        mask = np.triu(np.ones((self.no_words, self.no_words)) * -np.inf, 1)
        return mask

    def div(self, matrix, val):
        for row in matrix:
            for idx in range(len(row)):
                row[idx] = row[idx] / val
        return matrix

    def final_contextual_vector(self):
        # Concatenate the attention outputs from all heads
        concatenated_output = np.concatenate(self.attention_outputs, axis=-1)
        # Project the concatenated output
        final_output = np.dot(concatenated_output, self.fc_weight)
        return final_output

    def get_output(self):
        self.calculate_new_values()
        self.calculate_dot()
        return self.final_contextual_vector()

# Example usage

s = Mask_MultiHeadAttention()

print(s.get_output())
