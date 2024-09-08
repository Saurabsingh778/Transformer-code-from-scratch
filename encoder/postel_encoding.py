import numpy as np
class PositionEncoding:
    def __init__(self, input):
        self.input = input
        self.vector_size = len(self.input[0])  # Assuming input is a list of vectors
        self.original_size = self.vector_size
        if self.vector_size % 2 == 1:
            self.vector_size += 1
        self.create_pos_vector()

    def create_pos_vector(self):
        position = np.arange(len(self.input))[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.vector_size, 2) * -(np.log(10000.0) / self.vector_size))

        pos_encodings = np.zeros((len(self.input), self.vector_size))
        pos_encodings[:, 0::2] = np.sin(position * div_term)
        pos_encodings[:, 1::2] = np.cos(position * div_term)

        if self.original_size % 2 == 1:
            pos_encodings = pos_encodings[:, :self.original_size]

        self.output = []
        
        for i in range(len(self.input)):
            lis = []
            for x, y in zip(self.input[i], pos_encodings[i]):
                lis.append(x + y)
            self.output.append(lis)

    def get_output(self):
        return self.output
    
    
