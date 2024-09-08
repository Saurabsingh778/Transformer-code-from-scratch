import numpy as np
from math import sqrt

class CrossAttention:
    def __init__(self, input_vectors = None, output_vectors = None, num_heads = 8):
        self.input_vector = input_vectors
        self.output_vector = output_vectors
        self.num_heads = num_heads
        self.dim = len(input[0][0][0])
        self.head_dim = self.dim // self.num_heads

        assert self.dim % self.num_heads == 0

        self.nwords_in_input = len(self.input_vector)
        self.nwords_in_output = len(self.output_vector)

        #creating weights q, k, v
        self.query_weights = [np.random.rand(self.dim, self.head_dim) for _ in range(self.num_heads)]
        self.key_weights = [np.random.rand(self.dim, self.head_dim) for _ in range(self.num_heads)]
        self.value_weights = [np.random.rand(self.dim, self.head_dim) for _ in range(self.num_heads)]

        self.fc_weight = np.random.rand(self.head_dim * self.num_heads, self.dim)
        self.calculate_new_values()
        self.calculate_dot()
        self.final_contextual_vector()
    
    #creating query, key, value vectors
    def calculate_new_values(self):
        self.queries, self.keys, self.values = [], [], []
        for i in range(self.num_heads):
            query = np.dot(self.output_vector, self.query_weights[i])
            key = np.dot(self.input_vector, self.key_weights[i])
            value = np.dot(self.input_vector, self.value_weights[i])
            self.queries.append(query)
            self.keys.append(key)
            self.values.append(value)

    #clculating softmax by applying mask
    def softmax(self, matrix, mask=None):
        if mask is not None:
            matrix += mask  # Apply mask
        for mat in matrix:
            s = np.sum(mat)
            for idx in range(len(mat)):
                mat[idx] = mat[idx] / s
        return matrix

    #calculating new dot
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

    #creating mask
    def create_mask(self):
        mask = np.triu(np.ones((self.no_words, self.no_words)) * -np.inf, 1)
        return mask
    
    #calculating division of matrix
    def div(self, matrix, val):
        for row in matrix:
            for idx in range(len(row)):
                row[idx] = row[idx] / val
        return matrix

    #creating contexual vector
    def final_contextual_vector(self):
        # Concatenate the attention outputs from all heads
        concatenated_output = np.concatenate(self.attention_outputs, axis=-1)
        # Project the concatenated output
        final_output = np.dot(concatenated_output, self.fc_weight)
        return final_output
    
    #producting output
    def get_output(self):
        self.calculate_new_values()
        self.calculate_dot()
        return self.final_contextual_vector()